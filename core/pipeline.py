# -*- coding: utf-8 -*-
"""core/pipeline.py —— 检索编排层：把 BM25 / 向量 / RRF / Rerank 串成可消融的链路。

★ 核心设计：**每一路都能单独开关**。
  这不是为了灵活，而是为了 Day 3 的消融实验能跑出下面这张表：

    | 配置        | Hit@5 | MRR@5 | ... |
    | 纯向量       |       |       |     |
    | +BM25/RRF   |       |       |     |
    | +Reranker   |       |       |     |
    | +规则召回    |       |       |     |

  如果链路写死了，就没法把每个模块的边际贡献拆开 —— 那简历上就只能写一个
  总分，面试官追问"到底是哪一步带来的提升"就答不上来。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .retrieval import rrf_fuse
from .text import extract_measures


@dataclass
class RetrievalConfig:
    """一档检索配置。Day 3 的消融就是遍历这些组合。"""

    name: str = "hybrid+rerank"
    use_bm25: bool = True
    use_vector: bool = True
    use_rerank: bool = True
    use_rules: bool = False          # 编号/型号规则召回
    recall_k: int = 20               # 每路先召回的候选数（给精排留空间）
    top_k: int = 5                   # 最终返回数
    rrf_k: int = 60
    bm25_min_score: float = 0.0
    rrf_weight: Optional[Dict[str, float]] = None

    def label(self) -> str:
        parts = []
        if self.use_vector:
            parts.append("向量")
        if self.use_bm25:
            parts.append("BM25")
        if self.use_rerank:
            parts.append("Rerank")
        if self.use_rules:
            parts.append("规则")
        return "+".join(parts) if parts else "空配置"


@dataclass
class RetrievalResult:
    doc_ids: List[str]
    scores: List[float]
    latency_ms: float
    trace: Dict[str, List[str]] = field(default_factory=dict)   # 各路原始召回，便于调试与解释


class HybridRetriever:
    """混合检索器。

    Args:
        bm25: BM25Index（可为 None）
        vector_store: BaseVectorStore（可为 None）
        embed_fn: text -> 归一化向量（可为 None；为 None 时关闭向量路）
        rerank_fn: (query, [(doc_id, text)]) -> [(doc_id, score)] 降序
        doc_lookup: doc_id -> 文本（规则召回与 Rerank 需要）
    """

    def __init__(
        self,
        bm25=None,
        vector_store=None,
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
        rerank_fn: Optional[Callable[[str, List[Tuple[str, str]]], List[Tuple[str, float]]]] = None,
        doc_lookup: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.doc_lookup = doc_lookup or (lambda _id: None)

    # ------------------------------------------------------------ 各路召回
    def _bm25_route(self, query: str, k: int, min_score: float) -> List[str]:
        if self.bm25 is None:
            return []
        return [d for d, _ in self.bm25.search(query, top_k=k, min_score=min_score)]

    def _vector_route(self, query: str, k: int) -> List[str]:
        if self.vector_store is None or self.embed_fn is None:
            return []
        return [d for d, _ in self.vector_store.query(self.embed_fn(query), top_k=k)]

    def rule_route(self, query: str, k: int = 5) -> List[str]:
        """规则召回：query 里的编号 / 型号 / 精度值，优先给含同样数值的文档。

        为什么要单独一路：这类 token 信息量极高但**低频**，
        向量模型对它们表达力弱，BM25 又可能被中文单字噪声淹没。
        直接用字符串匹配把它们捞出来，是最省事也最有效的一路。
        """
        measures = extract_measures(query)
        if not measures:
            return []
        matched: List[str] = []
        # 通过 bm25 的文档列表遍历（避免额外维护一份 doc store）
        doc_ids = getattr(self.bm25, "doc_ids", None)
        if not doc_ids:
            return []
        for doc_id in doc_ids:
            text = self.doc_lookup(doc_id) or ""
            if any(m in text for m in measures):
                matched.append(doc_id)
        return matched[:k]

    # ------------------------------------------------------------ 主入口
    def retrieve(self, query: str, cfg: Optional[RetrievalConfig] = None) -> RetrievalResult:
        cfg = cfg or RetrievalConfig()
        started = time.perf_counter()

        routes: Dict[str, List[str]] = {}
        if cfg.use_vector:
            routes["vector"] = self._vector_route(query, cfg.recall_k)
        if cfg.use_bm25:
            routes["bm25"] = self._bm25_route(query, cfg.recall_k, cfg.bm25_min_score)
        if cfg.use_rules:
            r = self.rule_route(query, cfg.recall_k)
            if r:
                routes["rules"] = r

        # 只剩一路时不必融合（融合会丢掉原始分数排序语义）
        if len(routes) == 1:
            ordered = [(d, 0.0) for d in next(iter(routes.values()))]
        elif len(routes) == 0:
            ordered = []
        else:
            if cfg.rrf_weight:
                # 加权 RRF：对指定路的排名贡献乘系数
                merged_score: Dict[str, float] = {}
                for name, ids in routes.items():
                    w = cfg.rrf_weight.get(name, 1.0)
                    for rank, doc_id in enumerate(ids, start=1):
                        merged_score[doc_id] = merged_score.get(doc_id, 0.0) + w / (cfg.rrf_k + rank)
                ordered = sorted(merged_score.items(), key=lambda kv: (-kv[1], kv[0]))
            else:
                ordered = rrf_fuse(routes, k=cfg.rrf_k)

        candidates = [d for d, _ in ordered][: cfg.recall_k]

        # ---- Rerank ----
        if cfg.use_rerank and self.rerank_fn is not None and candidates:
            pairs = [(d, self.doc_lookup(d) or "") for d in candidates]
            reranked = self.rerank_fn(query, pairs)
            if reranked:
                final = reranked[: cfg.top_k]
                latency = (time.perf_counter() - started) * 1000
                return RetrievalResult(
                    doc_ids=[d for d, _ in final],
                    scores=[float(s) for _, s in final],
                    latency_ms=latency,
                    trace=routes,
                )

        final2 = ordered[: cfg.top_k]
        latency = (time.perf_counter() - started) * 1000
        return RetrievalResult(
            doc_ids=[d for d, _ in final2],
            scores=[float(s) for _, s in final2],
            latency_ms=latency,
            trace=routes,
        )


# ------------------------------------------------------------------ 预置配置
def ablation_configs(top_k: int = 5, recall_k: int = 20) -> List[RetrievalConfig]:
    """Day 3 消融实验的四档配置（对应简历里那张表）。"""
    return [
        RetrievalConfig(name="A_vector_only", use_vector=True, use_bm25=False,
                        use_rerank=False, use_rules=False, top_k=top_k, recall_k=recall_k),
        RetrievalConfig(name="B_hybrid_rrf", use_vector=True, use_bm25=True,
                        use_rerank=False, use_rules=False, top_k=top_k, recall_k=recall_k),
        RetrievalConfig(name="C_hybrid_rerank", use_vector=True, use_bm25=True,
                        use_rerank=True, use_rules=False, top_k=top_k, recall_k=recall_k),
        RetrievalConfig(name="D_hybrid_rerank_rules", use_vector=True, use_bm25=True,
                        use_rerank=True, use_rules=True, top_k=top_k, recall_k=recall_k),
    ]
