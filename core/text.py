# -*- coding: utf-8 -*-
"""core/text.py —— 中文友好的分词与文本归一化。

设计取舍（面试可讲）：
  - 生产环境首选 jieba（若已安装则自动使用）。
  - 但没有 jieba 时不能"只能跑英文" —— 中文 BM25 的关键是
    **单字 + 双字（bigram）混合切分**：单字保证召回，双字提供区分度。
    这是中文信息检索里的经典做法（无需词典也有不错效果）。
  - 之所以要双字：只切单字时「水稻」和「水稻穗」共享「水」「稻」，
    BM25 的区分度会被高频单字稀释；bigram 让「水稻」成为独立词项。
"""
from __future__ import annotations

import re
from typing import Iterable, List

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
_CJK_RUN_RE = re.compile(f"[{_CJK}]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[._\-/][A-Za-z0-9]+)*")
# 度量词项：小数（0.946）、百分比（75%）、或多位整数（30600 / 2022）。
# ★ 特意不匹配单个数字：中文语料里 "3 个要点" 这种单数字极常见，
#   收进词表只会制造噪声、稀释 IDF。"7:1:2" 里的 7 和 1 同理不该收。
_MEASURE_SRC = r"\d+(?:\.\d+)+%?|\d{2,}%?|\d+(?:\.\d+)?%"
_MEASURE_RE = re.compile(_MEASURE_SRC)

try:  # 有 jieba 就用（生产环境推荐）
    import jieba as _jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:  # pragma: no cover - 取决于环境
    _jieba = None
    _HAS_JIEBA = False


def has_jieba() -> bool:
    return _HAS_JIEBA


def _cjk_terms(run: str) -> List[str]:
    """中文串 -> 单字 + bigram。

    "水稻穗" -> ['水','稻','穗','水稻','稻穗']
    """
    terms = list(run)
    terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def tokenize(text: str, keep_measure: bool = True) -> List[str]:
    """中英混合分词，返回小写词项列表。

    Args:
        text: 原始文本
        keep_measure: 是否额外保留度量词项（小数 / 百分比 / 多位整数）。
            纯 ASCII 词总会被保留（含 "2022" 这类年份），
            本参数只控制「从中文串里额外抽出的度量词」。
    """
    if not text:
        return []

    lowered = text.lower()
    measure_terms = _MEASURE_RE.findall(lowered) if keep_measure else []
    measure_set = set(measure_terms)
    terms: List[str] = []

    if _HAS_JIEBA:
        for tok in _jieba.lcut(lowered):
            tok = tok.strip()
            if not tok:
                continue
            if _CJK_RUN_RE.fullmatch(tok):
                terms.append(tok)
            elif _ASCII_WORD_RE.fullmatch(tok) or _MEASURE_RE.fullmatch(tok):
                terms.append(tok)
        terms.extend(measure_terms)
        return [t for t in terms if t.strip()]

    # 无 jieba：中文切单字 + bigram
    for run in _CJK_RUN_RE.findall(lowered):
        terms.extend(_cjk_terms(run))
    # ASCII 词（含数字串）。
    # keep_measure=False 时跳过"纯数值"词（整数与小数都要跳，否则 "0.90" 会漏进来）
    for word in _ASCII_WORD_RE.findall(lowered):
        if not keep_measure and re.fullmatch(r"\d+(?:[._]\d+)*", word):
            continue
        terms.append(word)
    terms.extend(measure_terms)
    # 去重但保序：bigram 与 ASCII 词可能重复命中同一 measure
    seen = set()
    out = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def normalize_whitespace(text: str) -> str:
    """折叠空白与全角空格，去掉零宽字符（PDF 解析后常见噪声）。

    ★ 必须先统一换行符：Windows 文本是 `\\r\\n`，
      如果只按 `\\n{3,}` 折叠空行，`\\r\\n\\r\\n\\r\\n` 永远不会被压缩，
      而且残留的 `\\r` 会混进 token（污染 BM25 词表）。
      这个 bug 是被 test_collapses_whitespace 抓出来的。
    """
    if not text:
        return ""
    cleaned = text.replace("\u3000", " ").replace("\u200b", "").replace("\ufeff", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # 去掉行尾空格（Markdown 硬换行除外，这里统一处理即可）
    cleaned = re.sub(r" *\n", "\n", cleaned)
    return cleaned.strip()


def extract_measures(text: str) -> List[str]:
    """抽取文本中的数值型词项（小数 / 百分比 / 多位整数）。

    用途：① BM25 精确匹配这些低频高信息量 token；
         ② 回答"为什么纯向量检索对型号/编号不灵"时能指出具体机制。

    注意：刻意排除单个数字（见 _MEASURE_SRC 的注释），
    否则 "3 个要点"、"7:1:2 划分" 会往词表里灌噪声。
    """
    if not text:
        return []
    return _MEASURE_RE.findall(text)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """集合 Jaccard 相似度（用于评测集去重、近重复 chunk 检测）。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0
