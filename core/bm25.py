# -*- coding: utf-8 -*-
"""core/bm25.py —— BM25 关键词检索（自实现，零第三方依赖）。

为什么自己实现而不直接用 rank_bm25：
  1. rank_bm25 不支持中文，必须先自己分词 —— 那不如把打分也一起做掉；
  2. 面试时"BM25 的 k1/b 是什么、为什么要 IDF"是高频追问，
     自己写过才答得清楚；
  3. 零依赖 = 可以脱离环境单测（本项目所有 core 模块都遵守这条）。

公式（Okapi BM25，与 rank_bm25 的 BM25Okapi 一致）：

    score(q, d) = Σ_{t∈q} IDF(t) · f(t,d)·(k1+1)
                            ─────────────────────────────
                            f(t,d) + k1·(1 - b + b·|d|/avgdl)

    IDF(t) = ln( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )      ← 加 1 防负值
    k1 = 1.5（词频饱和速度）   b = 0.75（长度归一化强度）
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .text import tokenize


class BM25Index:
    """内存 BM25 索引。适合本项目规模（几百~几千 chunk）。

    Args:
        k1: 词频饱和参数，越大越"看得起"重复出现的词（默认 1.5）
        b:  文档长度归一化强度，0 = 不归一化，1 = 完全归一化（默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.tf: List[Counter] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self._idf_cache: Dict[str, float] = {}

    # ------------------------------------------------------------ 建索引
    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        self.doc_ids.append(doc_id)
        self.doc_tokens.append(tokens)
        self.doc_len.append(len(tokens))
        counter = Counter(tokens)
        self.tf.append(counter)
        for term in counter:
            self.df[term] += 1
        self._idf_cache.clear()

    def add_many(self, docs: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)
        self.finalize()

    def finalize(self) -> None:
        """建完索引后调用，计算 avgdl 与 IDF。"""
        n = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self._idf_cache = {term: self._idf(term) for term in self.df}

    # ------------------------------------------------------------ 打分
    def _idf(self, term: str) -> float:
        """带 +1 平滑的 IDF。

        注意：未出现在语料中的词（n=0）IDF 最高，这对检索没有意义
        （因为它在任何文档里 f=0，贡献也是 0），所以无需特判。
        """
        n = len(self.doc_ids)
        if n == 0:
            return 0.0
        df = self.df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def idf(self, term: str) -> float:
        if not self._idf_cache:
            self.finalize()
        return self._idf_cache.get(term, self._idf(term))

    def score(self, query: str, index: int) -> float:
        """单文档得分。"""
        q_terms = tokenize(query)
        if not q_terms:
            return 0.0
        tf = self.tf[index]
        dl = self.doc_len[index]
        avgdl = self.avgdl or 1.0
        total = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf(term)
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            total += idf * (f * (self.k1 + 1.0)) / denom
        return total

    def search(
        self, query: str, top_k: int = 20, min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """返回 [(doc_id, score)] 降序。

        Args:
            min_score: 分数下限。**中文场景建议显式设置**（如 0.5）。
                ★ 原因：中文走单字 + bigram 切分时，任何一段中文查询几乎
                总能撞上几个高频单字（"的"、"不"、"是"…），这些字的 IDF
                很低但非零，会让无关文档拿到一个正分。若不过滤，
                "完全无关的查询"也会返回一堆低分文档，污染融合结果。
                阈值过滤等价于一次"低置信拒答"。
        """
        scored: List[Tuple[str, float]] = []
        for i, doc_id in enumerate(self.doc_ids):
            s = self.score(query, i)
            if s > min_score:
                scored.append((doc_id, s))
        # 分数相同则按 doc_id 稳定排序，保证结果可复现
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.doc_ids)
