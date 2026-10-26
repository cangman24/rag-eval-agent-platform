# -*- coding: utf-8 -*-
"""rag1_knowledge_qa/bootstrap.py —— 生产装配（需要真机：GPU + BGE 模型 + DeepSeek Key）。

把已验证的 core 模块串成一个可服务的实例：

    core/embedder.py  ->  BgeEmbedder（已实现）
        encode_batch(texts: list[str]) -> list[list[float]]   # L2 归一化
        encode_one(text: str)         -> list[float]

    core/reranker.py  ->  BgeReranker（已实现）
        rerank(query: str, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]

参考实现（sentence-transformers / FlagEmbedding 路线）与 transformers 版本注意事项
见 core/embedder.py 与 core/reranker.py 的 docstring。
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from core.bm25 import BM25Index
from core.ingest import IngestConfig, ingest_dir, load_chunks
from core.llm import DiskCache, LLMClient
from core.pipeline import HybridRetriever, RetrievalConfig
from core.service import RagService, ServiceConfig
from core.vectorstore import InMemoryVectorStore

# --------------------------------------------------------------------- 环境
DEFAULTS = {
    "RAG_DOCS_DIR": "data/raw",
    "RAG_CHUNKS_FILE": "data/vectors/chunks.json",
    "RAG_VECTORS_FILE": "data/vectors/vectors.json",
    "BGE_M3_PATH": "models/bge-m3",
    "BGE_RERANKER_PATH": "models/bge-reranker-v2-m3",
    "RAG_CHUNK_SIZE": "512",
    "RAG_OVERLAP": "64",
    "RAG_PARENT_SIZE": "2048",
    "RAG_TOP_K": "5",
    "RAG_RECALL_K": "20",
    "RAG_REFUSAL_THRESHOLD": "0.3",
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    "DEEPSEEK_MODEL": "deepseek-chat",
}


def load_env_file(path: str = ".env") -> bool:
    """把 .env 读进 os.environ，返回是否做了填充（用于测试与自省）。

    ★ 这个函数必须放在 **bootstrap** 里，而不是 cli.py 里。

      踩坑记录：初版只在 cli.py 的 main() 里加载 .env，于是
      `python -m rag1_knowledge_qa.eval --judge` 走的是**另一条入口**，
      .env 没被读 -> bootstrap 里 `os.environ.get("DEEPSEEK_API_KEY")` 为空
      -> 服务静默降级成"未配置 LLM"模式 -> 评测报告里出现

          忠实度=0.00   幻觉率=0.00%

      这种**看起来完美、实际根本没测**的数字。比报错危险得多。

      教训：环境加载属于「装配」这一步，不属于某个命令行入口。
      放在装配层，任何入口（cli / eval / server / 你自己的脚本）都自动生效。

    ★ 惰性 + 容错，与 core/ 的零依赖可测设计保持一致：
      · 不写成模块顶层 `from dotenv import load_dotenv` ——
        那会让没装 dotenv 的解释器连 import 本模块都失败
      · 已存在的环境变量**不覆盖** —— 显式传入的优先于 .env 文件，
        这样测试里 `os.environ[...] = "fake"` 依然生效
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(path, override=False)
        return True
    except ImportError:
        pass

    # 没有 python-dotenv 就手写极简解析：KEY=VALUE，忽略注释与空行
    try:
        if not os.path.exists(path):
            return False
        touched = False
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    touched = True
        return touched
    except OSError:
        return False


def env(key: str) -> str:
    return os.environ.get(key, DEFAULTS.get(key, ""))


def env_int(key: str) -> int:
    return int(env(key))


def env_float(key: str) -> float:
    return float(env(key))


# --------------------------------------------------------------------- 装配
def build_embedder_and_reranker() -> Tuple[Any, Any]:
    """延迟导入，给缺失依赖一个可读的错误。"""
    try:
        from core.embedder import BgeEmbedder  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "缺少 core/embedder.py。请按本文件顶部 docstring 里的参考实现补全。"
        ) from exc
    try:
        from core.reranker import BgeReranker  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "缺少 core/reranker.py。请按本文件顶部 docstring 里的参考实现补全。"
        ) from exc

    embedder = BgeEmbedder(env("BGE_M3_PATH"))
    reranker = BgeReranker(env("BGE_RERANKER_PATH"))
    return embedder, reranker


def build_or_load_index(embedder: Any):
    """有缓存就加载，没有就现场建库并落盘。

    ★ 为什么要分两条路：建库（解析 + 向量化几千个块）在 GPU 上也要几分钟，
      评测迭代时每次都重建会严重拖慢节奏。落盘后 Day 3 的网格搜索才可行。
    """
    chunks_file = env("RAG_CHUNKS_FILE")
    vectors_file = env("RAG_VECTORS_FILE")

    if os.path.exists(chunks_file) and os.path.exists(vectors_file):
        chunks, parents = load_chunks(chunks_file)
        store = InMemoryVectorStore.load(vectors_file)
        bm25 = BM25Index()
        for c in chunks:
            bm25.add(c.chunk_id, f"{c.heading_path} {c.text}")
        bm25.finalize()
        parent_of = {c.chunk_id: c.parent_id for c in chunks}
        lookup = lambda cid: parents.get(parent_of.get(cid, ""), "")  # noqa: E731
        return bm25, store, lookup, {"loaded_from_cache": True,
                                     "chunks": len(chunks)}

    result = ingest_dir(
        env("RAG_DOCS_DIR"), embedder.encode_batch,
        config=IngestConfig(chunk_size=env_int("RAG_CHUNK_SIZE"),
                            overlap=env_int("RAG_OVERLAP"),
                            parent_size=env_int("RAG_PARENT_SIZE"),
                            embedding_model=os.path.basename(env("BGE_M3_PATH"))),
    )
    from core.ingest import save_chunks
    save_chunks(result, chunks_file)
    result.vector_store.persist(vectors_file)
    return result.bm25, result.vector_store, result.doc_text, result.stats


def build_production_service() -> RagService:
    """完整装配：embedder + reranker + 索引 + LLM + ServiceConfig。

    ★ 第一步就是加载 .env —— 这是「装配」的职责，不是某个入口的职责。
      放在这里之后，任何入口（cli / eval / server / 你的脚本）都自动生效，
      不会再出现"某个入口忘了加载、于是静默降级成无 LLM 模式"的情况。
    """
    load_env_file()

    embedder, reranker = build_embedder_and_reranker()
    bm25, store, lookup, stats = build_or_load_index(embedder)

    retriever = HybridRetriever(
        bm25=bm25, vector_store=store,
        embed_fn=embedder.encode_one,
        rerank_fn=reranker.rerank,
        doc_lookup=lookup,
    )

    llm: Optional[LLMClient] = None
    if os.environ.get("DEEPSEEK_API_KEY"):
        llm = LLMClient(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=env("DEEPSEEK_BASE_URL"),
            model=env("DEEPSEEK_MODEL"),
            cache=DiskCache("data/cache/llm"),
        )
    else:
        print("[warn] 未设置 DEEPSEEK_API_KEY -> 走无 LLM 降级模式（只回检索片段）")

    svc = RagService(
        # ★ 这里必须传 llm，不能写死 None。
        #   踩坑记录：初版这里写成了 `llm=None` —— 上面第 168 行明明把
        #   LLMClient 构造好了，却因为这一行硬编码而被丢弃，于是：
        #     · 服务静默降级成"未配置 LLM"模式（只回检索片段）
        #     · 耗时只有 ~190ms（没发网络请求），很容易误判成"模型答得不好"
        #     · 更糟的是 `if os.environ.get(...)` 那条分支走的是 else 吗？不是，
        #       Key 明明存在，所以连 [warn] 提示都不会打印 —— 完全静默。
        #   教训：构造完对象后，**检查它有没有真的被用上**。
        retriever=retriever, doc_lookup=lookup, llm=llm,
        config=ServiceConfig(
            top_k=env_int("RAG_TOP_K"),
            recall_k=env_int("RAG_RECALL_K"),
            refusal_threshold=env_float("RAG_REFUSAL_THRESHOLD"),
        ),
        retrieval_config=RetrievalConfig(
            top_k=env_int("RAG_TOP_K"),
            recall_k=env_int("RAG_RECALL_K"),
        ),
    )
    # ★ 把装配件挂在 service 上：评测要"实测"模型名与向量维度，
    #   不能靠读环境变量再抄一遍（那是文档声称，不是证据）。
    #   挂属性而不是改 RagService 构造签名，避免动到 core 层的公共接口。
    svc.embedder = embedder
    svc.reranker = reranker
    svc.index_meta = stats
    return svc
