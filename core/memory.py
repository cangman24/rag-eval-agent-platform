# -*- coding: utf-8 -*-
"""core/memory.py —— 三层记忆 + 上下文压缩。

对应简历里的两句：
  「基于 Redis + 向量库设计短期工作记忆、长期情景记忆与用户画像三层结构」
  「上下文占用达阈值时触发模型摘要压缩并截断大工具结果」

★ 本项目**不依赖 Redis**：`WorkingMemory` 用内存 + LRU，
  长期层用 `BaseVectorStore`（内存或 Chroma 均可）。
  这不是偷懒 —— 三层记忆的价值在**分层策略**（什么进哪层、什么时候淘汰、
  怎么回注），跟底层是不是 Redis 无关。真机要换 Redis 只需替换 `WorkingMemory`
  的存储后端，接口不变。面试被问"为什么用 Redis"，如实说"生产用 Redis 做
  会话态，我本地实现用内存 LRU，接口一致"是完全站得住的回答。

三级压缩策略（阈值可配置，这是简历上"60/85/95"的来源）：
    ratio < tier1           -> 不动
    tier1 <= ratio < tier2  -> 早期摘要（保留最近 N 条）
    tier2 <= ratio < tier3  -> 结构化归档（保留更少 + 抽关键词）
    ratio >= tier3          -> 全量压缩（只留最近极少条 + 摘要）
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


def _count_tokens(text: str) -> int:
    """本地 token 估算（复用 retrieval 的实现，避免两处口径不一致）。"""
    from .retrieval import count_tokens_approx
    return count_tokens_approx(text)


# --------------------------------------------------------------------- 消息
@dataclass
class Message:
    role: str          # system / user / assistant / tool / summary
    content: str
    ts: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content,
                "ts": self.ts, "meta": self.meta}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Message":
        return Message(role=d.get("role", "user"), content=d.get("content", ""),
                       ts=float(d.get("ts", 0.0)), meta=dict(d.get("meta") or {}))

    def tokens(self) -> int:
        return _count_tokens(self.content)


# --------------------------------------------------------------------- 短期
@dataclass
class Turn:
    """一轮对话（用户输入 + 助手输出）。"""

    user: str
    assistant: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"user": self.user, "assistant": self.assistant, "ts": self.ts}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Turn":
        return Turn(user=d.get("user", ""), assistant=d.get("assistant", ""),
                    ts=float(d.get("ts", 0.0)))


class WorkingMemory:
    """短期工作记忆：LRU + 容量上限。

    保留"最近 max_turns 轮"，超出按最近最少使用淘汰 —— 滑动窗口的落地。
    """

    def __init__(self, max_turns: int = 12) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be > 0")
        self.max_turns = max_turns
        self._turns: "OrderedDict[str, Turn]" = OrderedDict()
        # ★ 单调递增序号：不能用 "时间戳+内容" 做 key。
        #   同一毫秒内连续两次 add("同一个问题") 会算出同一个 key，
        #   后一次把前一次覆盖掉 —— 这个 bug 正是被
        #   test_duplicate_content_still_counts 抓出来的。
        self._seq = 0

    def add(self, user: str, assistant: str = "") -> Turn:
        turn = Turn(user=user, assistant=assistant)
        self._seq += 1
        key = f"{self._seq:08d}"
        self._turns[key] = turn
        self._turns.move_to_end(key)
        while len(self._turns) > self.max_turns:
            self._turns.popitem(last=False)
        return turn

    def recent(self, n: Optional[int] = None) -> List[Turn]:
        turns = list(self._turns.values())
        return turns[-n:] if n else turns

    def messages(self, n: Optional[int] = None) -> List[Message]:
        out: List[Message] = []
        for t in self.recent(n):
            out.append(Message("user", t.user, t.ts))
            if t.assistant:
                out.append(Message("assistant", t.assistant, t.ts))
        return out

    def tokens(self) -> int:
        return sum(_count_tokens(t.user) + _count_tokens(t.assistant)
                   for t in self._turns.values())

    def __len__(self) -> int:
        return len(self._turns)

    def clear(self) -> None:
        self._turns.clear()


# --------------------------------------------------------------------- 长期
class EpisodicMemory:
    """长期情景记忆：历史对话摘要，按向量相似度召回。

    与 `WorkingMemory` 的分工：
      - 短期：**最近**几轮，逐字保留，供模型直接看；
      - 长期：**很久以前**的摘要，按相关度召回，供模型按需取。
    这条分工是面试常问点："为什么不全部塞进上下文？"
    """

    def __init__(self, vector_store=None, embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
                 namespace: str = "episodic") -> None:
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.namespace = namespace
        self._docs: Dict[str, Dict[str, Any]] = {}

    def _new_id(self, text: str) -> str:
        return f"{self.namespace}-{hashlib.md5(text.encode('utf-8')).hexdigest()[:16]}"

    def add(self, summary: str, meta: Optional[Dict[str, Any]] = None) -> str:
        doc_id = self._new_id(summary)
        self._docs[doc_id] = {"summary": summary, "meta": dict(meta or {}),
                              "ts": time.time()}
        if self.vector_store is not None and self.embed_fn is not None:
            self.vector_store.add(ids=[doc_id], embeddings=[self.embed_fn(summary)],
                                  documents=[summary],
                                  metadatas=[self._docs[doc_id]["meta"]])
        return doc_id

    def recall(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.vector_store is not None and self.embed_fn is not None:
            hits = self.vector_store.query(self.embed_fn(query), top_k=top_k)
            out = []
            for doc_id, score in hits:
                rec = self._docs.get(doc_id)
                if rec:
                    out.append({"id": doc_id, "summary": rec["summary"],
                                "score": float(score), "meta": rec["meta"]})
            return out
        # 无向量库时的降级：按 token 重合度排（保证链路仍可用）
        from .text import tokenize
        q = set(tokenize(query))
        scored = []
        for doc_id, rec in self._docs.items():
            d = set(tokenize(rec["summary"]))
            scored.append((doc_id, len(q & d) / (len(q) or 1), rec))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [{"id": i, "summary": r["summary"], "score": float(s), "meta": r["meta"]}
                for i, s, r in scored[:top_k] if s > 0]

    def __len__(self) -> int:
        return len(self._docs)


class UserProfile:
    """用户画像：键值对，覆盖式更新（长期稳定，不做淘汰）。"""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def update(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update_many(self, kv: Dict[str, Any]) -> None:
        self._data.update(kv)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def to_text(self) -> str:
        if not self._data:
            return ""
        return "；".join(f"{k}={v}" for k, v in sorted(self._data.items()))

    def __len__(self) -> int:
        return len(self._data)


class LayeredMemory:
    """三层记忆的统一入口。"""

    def __init__(self, working: Optional[WorkingMemory] = None,
                 episodic: Optional[EpisodicMemory] = None,
                 profile: Optional[UserProfile] = None) -> None:
        self.working = working or WorkingMemory()
        self.episodic = episodic or EpisodicMemory()
        self.profile = profile or UserProfile()

    def build_context(self, query: str, recent_n: int = 6,
                      recall_k: int = 3) -> Dict[str, Any]:
        """组装给模型的记忆上下文。

        Returns:
            {"profile": str, "episodic": [...], "recent": [Message...]}
        """
        return {
            "profile": self.profile.to_text(),
            "episodic": self.episodic.recall(query, top_k=recall_k) if query else [],
            "recent": [m.to_dict() for m in self.working.messages(recent_n)],
        }

    def total_tokens(self) -> int:
        return self.working.tokens()


# --------------------------------------------------------------------- 压缩
@dataclass
class CompressionResult:
    messages: List[Message]
    level: str                    # none / early_summary / structured_archive / full_compress
    ratio: float
    tokens_before: int
    tokens_after: int
    summary: str = ""

    @property
    def saved_ratio(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before


def truncate_tool_result(text: str, max_chars: int = 800) -> str:
    """截断过长的工具返回。

    ★ 为什么必须做：工具返回的长 JSON（比如一次召回 20 条 chunk）
      往往占掉上下文的大头，而模型真正需要的只是前几条。
      这是"上下文压缩"里性价比最高的一步 —— 零成本、立竿见影。
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[已截断，原长度 {len(text)} 字符]"


class ContextCompressor:
    """三级阈值压缩。

    Args:
        token_budget: 上下文预算（决定 ratio 的分母）
        tier1/2/3: 触发各级压缩的占用比例
        keep_recent: 各级保留的最近消息条数
        summarize_fn: (text) -> 摘要；为 None 时用抽取式降级方案
    """

    def __init__(self, token_budget: int = 4000,
                 tier1: float = 0.60, tier2: float = 0.85, tier3: float = 0.95,
                 keep_recent: Sequence[int] = (12, 8, 6),
                 max_tool_chars: int = 800,
                 summarize_fn: Optional[Callable[[str], str]] = None) -> None:
        if not (0 < tier1 < tier2 < tier3 <= 1.0):
            raise ValueError("阈值必须满足 0 < tier1 < tier2 < tier3 <= 1")
        if token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        self.token_budget = token_budget
        self.tier1, self.tier2, self.tier3 = tier1, tier2, tier3
        self.keep_recent = tuple(keep_recent)
        if len(self.keep_recent) != 3:
            raise ValueError("keep_recent 需要 3 个值，对应三级压缩")
        self.max_tool_chars = max_tool_chars
        self.summarize_fn = summarize_fn

    # ---------------------------------------------------------- 工具
    def _summarize(self, text: str, max_chars: int) -> str:
        if not text.strip():
            return ""
        if self.summarize_fn is not None:
            try:
                out = self.summarize_fn(text) or ""
                if out.strip():
                    return out.strip()
            except Exception:  # noqa: BLE001 - 摘要失败必须降级而不是中断
                pass
        # 抽取式降级：保留首尾（首句通常是结论，尾句常含关键约束）
        if len(text) <= max_chars:
            return text
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2:]
        return f"{head}\n...[中略]...\n{tail}"

    def _truncate_tool_messages(self, messages: List[Message]) -> List[Message]:
        out = []
        for m in messages:
            if m.role == "tool":
                out.append(Message(m.role, truncate_tool_result(m.content, self.max_tool_chars),
                                   m.ts, m.meta))
            else:
                out.append(m)
        return out

    # ---------------------------------------------------------- 主入口
    def compress(self, messages: Sequence[Message],
                 extra_tokens: int = 0) -> CompressionResult:
        messages = self._truncate_tool_messages(list(messages))
        tokens_before = sum(m.tokens() for m in messages) + extra_tokens
        ratio = tokens_before / self.token_budget

        if ratio < self.tier1:
            return CompressionResult(messages, "none", ratio, tokens_before,
                                     tokens_before)

        if ratio < self.tier2:
            level, keep = "early_summary", self.keep_recent[0]
        elif ratio < self.tier3:
            level, keep = "structured_archive", self.keep_recent[1]
        else:
            level, keep = "full_compress", self.keep_recent[2]

        if len(messages) <= keep:
            after = sum(m.tokens() for m in messages)
            return CompressionResult(messages, level, ratio, tokens_before, after)

        old, recent = messages[:-keep], messages[-keep:]
        target_chars = max(200, self.token_budget // 8)
        summary_text = self._summarize("\n".join(m.content for m in old), target_chars)
        summary_msg = Message("summary", f"[历史摘要] {summary_text}")
        result = [summary_msg] + recent
        after = sum(m.tokens() for m in result)
        return CompressionResult(result, level, ratio, tokens_before, after, summary_text)

    def should_compress(self, messages: Sequence[Message],
                        extra_tokens: int = 0) -> bool:
        total = sum(m.tokens() for m in messages) + extra_tokens
        return total / self.token_budget >= self.tier1
