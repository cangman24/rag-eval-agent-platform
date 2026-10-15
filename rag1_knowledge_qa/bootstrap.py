# -*- coding: utf-8 -*-
"""rag1_knowledge_qa/bootstrap.py —— 生产装配（需要真机：GPU + BGE 模型 + DeepSeek Key）。

★ 这个文件是给你**补完剩下 20% 代码**的地方。它把已验证的 core 模块
  串成一个可服务的实例，你只需要实现两个类：

      core/embedder.py  ->  BgeEmbedder
          encode_batch(texts: list[str]) -> list[list[float]]   # 必须 L2 归一化
          encode_one(text: str)         -> list[float]

      core/reranker.py  ->  BgeReranker
          rerank(query: str, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]

  参考实现（用 sentence-transformers / FlagEmbedding）：

      # core/embedder.py
      from sentence_transformers import SentenceTransformer
      from core.vectorstore import l2_normalize

      class BgeEmbedder:
          def __init__(self, model_path, device=None, batch_size=32):
              self.model = SentenceTransformer(model_path, device=device)
              self.batch_size = batch_size

          def encode_batch(self, texts):
              vecs = self.model.encode(texts, batch_size=self.batch_size,
                                       normalize_embeddings=True,
                                       show_progress_bar=len(texts) > 200)
              return [l2_normalize(list(map(float, v))) for v in vecs]

          def encode_one(self, text):
              return self.encode_batch([text])[0]

      # core/reranker.py
      from FlagEmbedding import FlagReranker

      class BgeReranker:
          def __init__(self, model_path, use_fp16=True):
              self.model = FlagReranker(model_path, use_fp16=use_fp16)

          def rerank(self, query, pairs):
              if not pairs:
                  return []
              scores = self.model.compute_score(
                  [[query, text] for _id, text in pairs], normalize=True)
              if not isinstance(scores, list):
                  scores = [scores]
              out = [(pairs[i][0], float(s)) for i, s in enumerate(scores)]
              out.sort(key=lambda kv: (-kv[1], kv[0]))
              return out

  ⚠️ 注意 transformers 版本：sentence-transformers / FlagEmbedding 目前主要
     兼容 4.x。若你的环境是 transformers 5.x（如 fieldai 环境是 5.19.0），
     先降到 4.44.2：
         pip install "transformers==4.44.2" --force-reinstall --no-deps
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
    """完整装配：embedder + reranker + 索引 + LLM + ServiceConfig。"""
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

    return RagService(
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
