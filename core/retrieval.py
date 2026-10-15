# -*- coding: utf-8 -*-
"""core/retrieval.py —— 检索融合与切分。纯 Python，可独立单测。

包含：
  - rrf_fuse       : Reciprocal Rank Fusion（混合检索融合）
  - weighted_fuse  : 加权求和（用作对照，面试时解释"为什么不用它"）
  - dedup_keep_order
  - recursive_chunk: 递归切分，保留标题层级
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple


# --------------------------------------------------------------------- 融合
def rrf_fuse(
    rankings: Dict[str, Sequence[str]], k: int = 60
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion。

    Args:
        rankings: {"vector": [doc_id...], "bm25": [doc_id...]}，各列表已按相关性降序
        k: RRF 平滑常数，论文默认 60

    Returns:
        [(doc_id, score)] 按 score 降序

    为什么用 RRF 而不是加权求和：
        BM25 分数是无上界实数，向量相似度是 [-1,1] 或余弦 [0,1]，
        两者量纲不可比。归一化后加权对分数分布很敏感，需要反复调权重。
        RRF 只用「排名」不用「分数」，天然免疫量纲问题，且几乎不需调参。
    """
    scores: Dict[str, float] = {}
    for ids in rankings.values():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_fuse(
    rankings: Dict[str, Sequence[float]],
    weights: Dict[str, float] | None = None,
    normalize: bool = True,
) -> List[Tuple[str, float]]:
    """加权求和融合——仅作为 RRF 的对照组。

    Args:
        rankings: {"vector": [(doc_id, score)...]}（注意：这里是带分数的）
    """
    weights = weights or {}
    normed: Dict[str, List[Tuple[str, float]]] = {}
    for name, pairs in rankings.items():
        if not pairs:
            normed[name] = []
            continue
        if normalize:
            lo = min(s for _, s in pairs)
            hi = max(s for _, s in pairs)
            span = (hi - lo) or 1.0
            normed[name] = [(d, (s - lo) / span) for d, s in pairs]
        else:
            normed[name] = list(pairs)

    total: Dict[str, float] = {}
    for name, pairs in normed.items():
        w = weights.get(name, 1.0)
        for doc_id, score in pairs:
            total[doc_id] = total.get(doc_id, 0.0) + w * score
    return sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    """去重且保持首次出现顺序。"""
    seen: set = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def fuse_and_cut(rankings: Dict[str, Sequence[str]], top_k: int = 20, k: int = 60) -> List[str]:
    """RRF 融合后截断到 top_k（给 Reranker 的候选集）。"""
    return [doc_id for doc_id, _ in rrf_fuse(rankings, k=k)][:top_k]


# --------------------------------------------------------------------- 切分
# 句子边界：中英文句末标点
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；.!?;])\s*")

# 标题识别 + 层级推断（层级决定 heading_path 的累积方式）
_MD_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[、.\s)\]]?\s*(.*)$")
_CHAPTER_RE = re.compile(r"^\s*(第[一二三四五六七八九十百零\d]+[章篇编部])\s*(.*)$")
_SECTION_RE = re.compile(r"^\s*(第[一二三四五六七八九十百零\d]+[节])\s*(.*)$")
# 「第X条/第X款/第X项」是条文正文的锚点，不作为标题剥离
_ARTICLE_RE = re.compile(r"^\s*第[一二三四五六七八九十百零\d]+[条款项]")

# 章/节/条这些在中文文档里天然代表不同层级，不能互相覆盖
_HIERARCHICAL_CN = set("章篇编部")


def heading_level(line: str) -> int | None:
    """推断标题的层级（数字越小层级越高）；不是标题则返回 None。

    ★ 关键领域知识（本函数最容易写错的地方）：
      中文规范/标准文档里，`第X章`、`第X节` 是**真标题**，可以独占一行；
      但 `第X条` 后面通常**直接跟着条文正文**（"第一条 本规范适用于…"），
      它是**内容**而不是标题。若把 `第X条` 也当标题剥掉，
      正文会整句消失 —— 这是实测踩到的坑，务必保持这个区分。
    """
    stripped = line.strip()
    if not stripped:
        return None

    md = _MD_RE.match(line)
    if md:
        return len(md.group(1))

    m = _CHAPTER_RE.match(line)
    if m:
        return 1 if m.group(1)[-1] in _HIERARCHICAL_CN else 2

    # 独立的「第X节」当作标题；「第X条」交给正文处理（见上面的说明）
    if _SECTION_RE.match(line) and not _ARTICLE_RE.match(line):
        return 2

    m = _NUM_RE.match(line)
    if m:
        # "3" -> 1 级；"3.2" -> 2 级；"3.2.1" -> 3 级
        # 仅当编号后接短标题才认为是标题，避免把 "3 个要点如下" 之类误判
        title = m.group(2).strip()
        if title and count_tokens_approx(title) <= 20 and not title.endswith(("。", "！", "？")):
            return m.group(1).count(".") + 1
        return None

    return None


def build_heading_path(stack: Iterable[tuple[int, str]]) -> str:
    """把 [(level, title)] 栈拼成 "第三章 > 3.2 检测标准" 形式。"""
    return " > ".join(title for _level, title in stack if title)


def count_tokens_approx(text: str) -> int:
    """近似 token 计数，零依赖。

    规则：CJK 字符按 1 token，连续 ASCII 词按 1 token 计（对 BGE/DeepSeek
    的中文场景误差约 ±10%）。若装了 tiktoken 可替换为精确计数，
    但评测中「相对比较」才是关键，近似值完全够用。
    """
    if not text:
        return 0
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", text))
    ascii_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    other = len(re.findall(r"[^\sA-Za-z0-9_\u4e00-\u9fff\u3040-\u30ff]", text))
    return cjk + ascii_words + other


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    heading_path: str = "",
) -> List[dict]:
    """递归切分：先按标题分段，再按句子合并到 chunk_size，段落间保留 overlap。

    Returns:
        [{"text":..., "heading_path":..., "token_len":...}, ...]

    heading_path 按标题层级累积（如 "第三章 > 3.2 检测标准"），
    用于生成时给出更精确的引用 —— 面试常问「你怎么保留文档结构」。

    层级来自 heading_level()：遇到同级或更高级标题时出栈，
    保证 "第二章 > 第一条" 不会被写成 "第一章 > 第二章 > 第一条"。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: List[dict] = []
    stack: List[tuple[int, str]] = []
    if heading_path:
        stack.append((1, heading_path))

    buffer: List[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        body = "".join(buffer).strip()
        if not body:
            buffer, buffer_tokens = [], 0
            return
        chunks.append(
            {
                "text": body,
                "heading_path": build_heading_path(stack),
                "token_len": count_tokens_approx(body),
            }
        )
        # 保留尾部 overlap 个 token 作为下一块的前缀（按句子粒度）
        if overlap > 0:
            tail: List[str] = []
            tail_tokens = 0
            for sent in reversed(_SENT_SPLIT_RE.split(body)):
                if not sent.strip():
                    continue
                st = count_tokens_approx(sent)
                if tail_tokens + st > overlap:
                    break
                tail.insert(0, sent)
                tail_tokens += st
            buffer = tail
            buffer_tokens = tail_tokens
        else:
            buffer, buffer_tokens = [], 0

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue

        level = heading_level(raw_line)
        if level is not None:
            flush()
            title = raw_line.strip().lstrip("#").strip()
            # 同级或更深的标题出栈，再压入当前标题
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        for sent in _SENT_SPLIT_RE.split(raw_line):
            if not sent.strip():
                continue
            st = count_tokens_approx(sent)
            if buffer_tokens + st > chunk_size and buffer_tokens > 0:
                flush()
            buffer.append(sent)
            buffer_tokens += st

    # 收尾：把剩余缓冲写入最后一块（不能用 flush，否则会多出一个 overlap 块）
    body = "".join(buffer).strip()
    if body:
        chunks.append(
            {
                "text": body,
                "heading_path": build_heading_path(stack),
                "token_len": count_tokens_approx(body),
            }
        )
    return chunks
