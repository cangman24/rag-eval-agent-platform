# -*- coding: utf-8 -*-
"""core/metrics.py —— 检索与生成评测指标。

设计原则：
  1. 纯函数、零第三方依赖 -> 可以脱离 torch / chroma 单独单测
  2. 排名全部用 1-based（MRR 用 0-based 是最常见的低级错误）
  3. gold_ids 支持多标签（多跳任务需要 2 个片段才完整）
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


# --------------------------------------------------------------------- 工具
def _as_set(x: Iterable[str] | None) -> set:
    return set(x) if x else set()


def _first_hit_rank(retrieved: Sequence[str], gold: set, k: int) -> int | None:
    """返回第一个命中的 1-based 排名；未命中返回 None。"""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gold:
            return rank
    return None


# --------------------------------------------------------------------- 检索指标
def hit_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int = 5) -> float:
    """Hit@K：前 K 条中是否命中任一 gold 片段。

    单条查询返回 0.0 / 1.0；对整个评测集取均值即为报告的 Hit@K。
    """
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    return 1.0 if _first_hit_rank(retrieved_ids, gold, k) is not None else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int = 5) -> float:
    """MRR@K：第一个命中位置的倒数排名（1-based）。未命中记 0。"""
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    rank = _first_hit_rank(retrieved_ids, gold, k)
    return 1.0 / rank if rank is not None else 0.0


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int = 5) -> float:
    """Recall@K：gold 集合中被召回的比例（多跳任务 gold 有多个片段）。"""
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    return len(_as_set(retrieved_ids[:k]) & gold) / len(gold)


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int = 5) -> float:
    """Precision@K：前 K 条里命中 gold 的比例。分母固定为 K。"""
    if k <= 0:
        return 0.0
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    return len(_as_set(retrieved_ids[:k]) & gold) / k


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int = 5) -> float:
    """NDCG@K：二值相关性版本。"""
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def all_retrieval_metrics(
    retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int = 5
) -> Dict[str, float]:
    """一次算齐一档配置在单条 query 上的全部检索指标。"""
    return {
        f"hit@{k}": hit_at_k(retrieved_ids, gold_ids, k),
        f"mrr@{k}": mrr_at_k(retrieved_ids, gold_ids, k),
        f"recall@{k}": recall_at_k(retrieved_ids, gold_ids, k),
        f"precision@{k}": precision_at_k(retrieved_ids, gold_ids, k),
        f"ndcg@{k}": ndcg_at_k(retrieved_ids, gold_ids, k),
    }


# --------------------------------------------------------------------- 拒答指标
def refusal_correctness(
    is_answerable: bool, answer: str, refusal_marker: str = "未找到"
) -> float:
    """越界拒答正确率。

    不可答的问题，若回答里出现拒答标记 -> 1.0（正确的拒答）
    可答的问题，若出现拒答标记 -> 0.0（错误的拒答 = 漏答）
    """
    refused = refusal_marker in (answer or "")
    if not is_answerable:
        return 1.0 if refused else 0.0
    return 0.0 if refused else 1.0


def hallucination_rate(groundedness_scores: Sequence[float], threshold: float = 3.0) -> float:
    """幻觉率 = 忠实度低于阈值的条数占比。

    groundedness 由 LLM-as-Judge 按 1-5 分给出，专门衡量
    "回答中的每个结论是否都能在检索片段里找到支撑"。
    """
    if not groundedness_scores:
        return 0.0
    bad = sum(1 for s in groundedness_scores if s < threshold)
    return bad / len(groundedness_scores)


# --------------------------------------------------------------------- 聚合
def aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """把逐条指标聚合为报告用的均值（保留 4 位小数）。"""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: round(sum(q.get(key, 0.0) for q in per_query) / n, 4) for key in keys}


def latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """延迟统计。P95 是简历里最好用的指标（比均值更能体现工程能力）。"""
    if not latencies_ms:
        return {}
    xs = sorted(latencies_ms)
    def pct(p: float) -> float:
        idx = min(len(xs) - 1, max(0, int(round(p / 100 * len(xs) + 0.5)) - 1))
        return xs[idx]
    return {
        "mean_ms": round(sum(xs) / len(xs), 1),
        "p50_ms": round(pct(50), 1),
        "p95_ms": round(pct(95), 1),
        "max_ms": round(xs[-1], 1),
    }
