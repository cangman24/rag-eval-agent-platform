# -*- coding: utf-8 -*-
"""core/ingest.py —— 建库流水线：文档解析 -> 切分 -> 向量化 -> 建索引。

对应简历里的「设计统一文档入库 pipeline……打通解析、清洗、语义分块、
向量化、父子块索引流程」。

这是把前几轮所有模块串起来的**装配层**：
    loaders（解析） -> recursive_chunk（切分） -> embed_fn（向量化）
        -> InMemoryVectorStore / ChromaVectorStore（向量索引）
        -> BM25Index（关键词索引）

★ 两个真实工程要点：

  1. **父子块索引**：检索用小块（精准），喂给模型用大块（上下文完整）。
     所以 chunk 要带 `parent_id`，检索命中子块后取父块正文。
     这是简历上"父子块双层索引"的落地点。

  2. **可复现的建库**：记录配置指纹（chunk_size/overlap/embedding 模型名/
     文档数），写进 `manifest.json`。否则你根本说不清"这份索引是
     哪套参数建的"——评测数字也就无法归因。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .bm25 import BM25Index
from .loaders import LoadedDoc, load_dir, summarize_load
from .retrieval import count_tokens_approx, recursive_chunk
from .vectorstore import InMemoryVectorStore


# --------------------------------------------------------------------- 配置
@dataclass
class IngestConfig:
    """建库参数。整套参数会算出指纹写进 manifest。"""

    chunk_size: int = 512
    overlap: int = 64
    parent_size: int = 2048          # 父块目标大小（≈4 倍子块）
    embedding_model: str = "unknown"
    min_chunk_tokens: int = 8        # 过短的块丢弃（避免噪声片段）
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None

    def fingerprint(self) -> str:
        payload = {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "parent_size": self.parent_size,
            "embedding_model": self.embedding_model,
            "min_chunk_tokens": self.min_chunk_tokens,
            "include": sorted(self.include or []),
            "exclude": sorted(self.exclude or []),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------- 结果
@dataclass
class Chunk:
    """一个检索单元（子块）。"""

    chunk_id: str
    text: str
    doc_id: str
    parent_id: str
    heading_path: str = ""
    token_len: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "text": self.text, "doc_id": self.doc_id,
            "parent_id": self.parent_id, "heading_path": self.heading_path,
            "token_len": self.token_len, "meta": self.meta,
        }


@dataclass
class IngestResult:
    """建库产物。"""

    chunks: List[Chunk]
    parents: Dict[str, str]                    # parent_id -> 完整正文
    vector_store: Any
    bm25: BM25Index
    config: IngestConfig
    stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def doc_text(self, chunk_id: str) -> str:
        """取某个 chunk 的**父块正文**（喂给模型用）。

        检索命中子块 -> 这里回吐父块，兼顾"召回精准"与"上下文完整"。
        """
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return self.parents.get(c.parent_id, c.text)
        return ""

    def child_text(self, chunk_id: str) -> str:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c.text
        return ""

    def metadata_for(self, chunk_id: str) -> Dict[str, Any]:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return {
                    "doc_id": c.doc_id, "parent_id": c.parent_id,
                    "heading_path": c.heading_path, "token_len": c.token_len,
                    **c.meta,
                }
        return {}

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.config.fingerprint(),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "chunk_size": self.config.chunk_size,
                "overlap": self.config.overlap,
                "parent_size": self.config.parent_size,
                "embedding_model": self.config.embedding_model,
                "min_chunk_tokens": self.config.min_chunk_tokens,
            },
            "stats": self.stats,
            "errors": self.errors,
        }


# --------------------------------------------------------------------- 切分
def build_parents(
    doc_id: str, text: str, parent_size: int
) -> List[Tuple[str, str, str]]:
    """把一篇文档切成父块，返回 [(parent_id, text, heading_path)]。

    父块用于喂模型（上下文完整），子块用于检索（召回精准）。

    ★ 必须一并返回父块自己的 heading_path：
      因为子块是**在父块内部**切的，`recursive_chunk` 在父块内只能看到
      父块内部的标题，看不到祖先章节。如果不把父块的层级前缀补回去，
      子块的 heading_path 就丢了顶层章节信息（这个 bug 是被
      test_headings_accumulate_within_parent 抓出来的）。
    """
    parents = recursive_chunk(text, chunk_size=parent_size, overlap=0)
    return [(f"{doc_id}#p{i}", p["text"], p.get("heading_path", ""))
            for i, p in enumerate(parents)]


def chunk_documents(
    docs: Sequence[LoadedDoc], config: IngestConfig
) -> Tuple[List[Chunk], Dict[str, str], Dict[str, int]]:
    """把文档切成带父子关系的 chunk 列表。

    ★ 关键实现选择：**先切父块，再在每个父块内部切子块**。

      一开始我写的是"在全文上切子块，再用字符串位置反查它属于哪个父块"——
      那是个脆弱启发式：文本可能出现多次、overlap 会让边界模糊，
      定位一错，父子关系就错，回吐的上下文全是别的章节。
      改成嵌套切分后，归属关系是**结构性确定**的，不依赖任何启发式。

      代价：父块边界会限制子块的最大跨度。对本项目（章节式文档）反而更好，
      因为子块永远不会跨章节——检索出的片段语义更聚焦。
    """
    chunks: List[Chunk] = []
    parents: Dict[str, str] = {}
    dropped_short = 0

    for doc in docs:
        parent_list = build_parents(doc.doc_id, doc.text, config.parent_size)
        if not parent_list:
            continue
        for pi, (pid, ptext, phpath) in enumerate(parent_list):
            parents[pid] = ptext
            subs = recursive_chunk(ptext, chunk_size=config.chunk_size,
                                   overlap=config.overlap)
            for ci, sub in enumerate(subs):
                if sub["token_len"] < config.min_chunk_tokens:
                    dropped_short += 1
                    continue
                sub_path = sub.get("heading_path", "")
                # 把父块的层级前缀补回子块，恢复"章 > 节"的完整路径
                if phpath and sub_path and not sub_path.startswith(phpath):
                    full_path = f"{phpath} > {sub_path}"
                else:
                    full_path = sub_path or phpath
                chunks.append(Chunk(
                    chunk_id=f"{pid}#c{ci}",
                    text=sub["text"],
                    doc_id=doc.doc_id,
                    parent_id=pid,
                    heading_path=full_path,
                    token_len=sub["token_len"],
                    meta={"format": doc.metadata.get("format", ""),
                          "filename": doc.metadata.get("filename", ""),
                          "parent_index": pi},
                ))

    return chunks, parents, {"dropped_short_chunks": dropped_short}


# --------------------------------------------------------------------- 建库
def ingest_dir(
    root: str,
    embed_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    config: Optional[IngestConfig] = None,
    vector_store: Any = None,
    recursive: bool = True,
    verbose: bool = False,
) -> IngestResult:
    """把一个目录的文档建成可检索索引。

    Args:
        root: 文档目录
        embed_fn: **批量**嵌入函数，签名 `(texts: list[str]) -> list[list[float]]`。
            必须返回 L2 归一化向量（与 InMemoryVectorStore 的约定一致）。
            真实项目替换成 BGE-M3 的批量编码；测试时注入假实现即可离线跑。
        config: 建库参数
        vector_store: 向量库实现；默认 InMemoryVectorStore
    """
    config = config or IngestConfig()
    vector_store = vector_store if vector_store is not None else InMemoryVectorStore()

    load_result = load_dir(root, recursive=recursive,
                           include=config.include, exclude=config.exclude)
    docs: List[LoadedDoc] = load_result["docs"]
    errors: List[Dict[str, str]] = list(load_result["errors"])

    chunks, parents, cstats = chunk_documents(docs, config)

    emb_errors: List[Dict[str, str]] = []
    if chunks:
        texts = [c.text for c in chunks]
        try:
            vectors = list(embed_fn(texts))
        except Exception as exc:  # noqa: BLE001 - 嵌入服务挂掉要有明确失败
            raise RuntimeError(f"嵌入失败（{len(texts)} 个块）: {exc}") from exc
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"嵌入返回数量不匹配：期望 {len(texts)}，实际 {len(vectors)}"
            )
        # 只对向量化成功的块建索引：坏块不进库，避免检索出空文本
        good_chunks, good_vectors = [], []
        for c, v in zip(chunks, vectors):
            if v is None:
                emb_errors.append({"chunk_id": c.chunk_id, "error": "嵌入为空"})
                continue
            good_chunks.append(c)
            good_vectors.append(v)
        chunks = good_chunks

        vector_store.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[list(v) for v in good_vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "parent_id": c.parent_id,
                        "heading_path": c.heading_path, "token_len": c.token_len}
                       for c in chunks],
        )

    # BM25 索引建在「heading_path + 正文」上：章节名本身是强信号
    bm25 = BM25Index()
    for c in chunks:
        bm25.add(c.chunk_id, f"{c.heading_path} {c.text}")
    bm25.finalize()

    stats = {
        "documents": len(docs),
        "chunks": len(chunks),
        "parents": len(parents),
        "total_chars": sum(len(d.text) for d in docs),
        "total_tokens": sum(c.token_len for c in chunks),
        "avg_chunk_tokens": round(
            (sum(c.token_len for c in chunks) / len(chunks)) if chunks else 0.0, 1
        ),
        "config_fingerprint": config.fingerprint(),
        # ★ 逐项参数也要落进 stats：
        #   只有指纹的话，看到指标波动时无法判断"到底是哪个参数变了"。
        #   （评测报告里曾经打印出 chunk_size=None，就是因为这里没带。）
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "parent_size": config.parent_size,
        "min_chunk_tokens": config.min_chunk_tokens,
        "embedding_model": config.embedding_model,
        "embedded": len(chunks),
        "skipped_files": len(load_result.get("skipped") or []),
        **cstats,
    }

    if verbose:
        print(summarize_load(load_result))
        print(f"切分 -> {stats['chunks']} 个子块 / {stats['parents']} 个父块；"
              f"平均 {stats['avg_chunk_tokens']} tokens")

    return IngestResult(chunks=chunks, parents=parents, vector_store=vector_store,
                        bm25=bm25, config=config, stats=stats,
                        errors=errors + emb_errors)


# --------------------------------------------------------------------- 落盘
def save_manifest(result: IngestResult, path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_manifest(), f, ensure_ascii=False, indent=2)
    return path


def save_chunks(result: IngestResult, path: str) -> str:
    """把 chunk 与父块落盘（评测脚本复用，避免每次重新解析）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"chunks": [c.to_dict() for c in result.chunks],
               "parents": result.parents}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def load_chunks(path: str) -> Tuple[List[Chunk], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    chunks = [Chunk(chunk_id=c["chunk_id"], text=c["text"], doc_id=c["doc_id"],
                    parent_id=c["parent_id"], heading_path=c.get("heading_path", ""),
                    token_len=c.get("token_len", 0), meta=c.get("meta") or {})
              for c in payload.get("chunks", [])]
    return chunks, dict(payload.get("parents") or {})
