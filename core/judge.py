# -*- coding: utf-8 -*-
"""core/judge.py —— LLM-as-Judge 评测实现。

对应简历里的「实现 LLM-as-Judge 评测框架，从相关性、准确性、完整性、
有用性 4 维度评分」以及「基于 RAGAs 搭建自动化评估体系」的自实现版本。

★ 为什么不用 RAGAs 而自己实现（面试可讲）：
  ① 依赖少、可控、可离线单测（本模块零第三方依赖，注入 fake LLM 即可测）；
  ② 评测口径是我们自己的业务口径，第三方库的默认 prompt 不一定贴合；
  ③ 关键是要能讲清"评委模型有没有偏见、怎么防"——自己写过才答得上来。

★ 对抗评委模型偏见的三条措施（也是面试常问点）：
  1. **四维独立打分**，而不是让模型给一个笼统总分 —— 避免"整体印象"污染细分能力
  2. **提供参考答案与检索片段作为依据**，而不是让模型凭感觉
  3. **固定评委模型与 Prompt 版本**并记录在报告里，保证纵向可比
  另：位置偏见（先给谁高分）用"固定顺序 + 抽检复核"控制，不追求消除。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# 四个评分维度（与简历一致）
JUDGE_DIMENSIONS = ("relevance", "accuracy", "completeness", "groundedness")

DIMENSION_LABELS = {
    "relevance": "相关性",
    "accuracy": "准确性",
    "completeness": "完整性",
    "groundedness": "忠实度",
}

DIMENSION_HINTS = {
    "relevance": "回答是否直接针对问题，有无答非所问",
    "accuracy": "事实与数值是否与参考答案一致，有无错误",
    "completeness": "是否覆盖了参考答案里的所有要点",
    "groundedness": "每个结论是否都能在检索片段中找到支撑（★ 这一项专测幻觉）",
}

JUDGE_PROMPT_VERSION = "v1"

JUDGE_PROMPT = """你是严格的评测员。请按下面的材料给"模型回答"打分。

【问题】
{question}

【参考答案】
{reference}

【检索片段】
{context}

【模型回答】
{answer}

评分要求：
- {dims_desc}
- 每个维度打 1~5 分整数（1=很差，3=及格，5=优秀）。
- 必须基于材料判断，不要凭你的先验知识补充事实。
- 若【参考答案】为空，则 accuracy/completeness 以"是否与检索片段一致"为准。

只输出 JSON，不要解释：
{{"relevance":x,"accuracy":x,"completeness":x,"groundedness":x,"reason":"一句话理由"}}"""


# --------------------------------------------------------------------- 解析
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def clamp_score(value: Any, lo: int = 1, hi: int = 5) -> float:
    """把分数夹到合法区间；无法解析时返回 0（并被上层记为判分失败）。"""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0:                       # 0 或负数视为无效
        return 0.0
    return float(max(lo, min(hi, n)))


def parse_judge_output(text: str) -> Dict[str, Any]:
    """从评委模型输出里解析四维分。

    模型可能：① 干净 JSON；② ```json 包裹；③ JSON 前后带解释；
    ④ 中文键名；⑤ 分数写成 "4分" / "4/5"。全部要能容错。
    """
    raw = (text or "").strip()
    if not raw:
        return {}

    # 去 markdown 围栏
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    obj: Optional[dict] = None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            obj = parsed
    except json.JSONDecodeError:
        pass

    if obj is None:
        m = _JSON_BLOCK_RE.search(raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    obj = parsed
            except json.JSONDecodeError:
                obj = None

    if obj is None:
        # 最后兜底：用正则抓 "key: 数字"
        obj = {}
        for dim in JUDGE_DIMENSIONS:
            m = re.search(rf"{dim}\D{{0,8}}(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
            if m:
                obj[dim] = m.group(1)
        # 中文键名
        for dim, label in DIMENSION_LABELS.items():
            if dim in obj:
                continue
            m = re.search(rf"{label}\D{{0,8}}(\d+(?:\.\d+)?)", raw)
            if m:
                obj[dim] = m.group(1)

    out: Dict[str, Any] = {}
    for dim in JUDGE_DIMENSIONS:
        val = obj.get(dim)
        if val is None:
            val = obj.get(DIMENSION_LABELS[dim])
        if val is None:
            continue
        score = clamp_score(val)
        if score > 0:
            out[dim] = score
    reason = obj.get("reason") or obj.get("理由") or ""
    if reason:
        out["reason"] = str(reason)[:300]
    return out


def build_judge_prompt(question: str, answer: str, reference: str = "",
                       context: str = "", max_context_chars: int = 2000) -> str:
    dims_desc = "\n".join(
        f"  · {DIMENSION_LABELS[d]}（{d}）：{DIMENSION_HINTS[d]}"
        for d in JUDGE_DIMENSIONS
    )
    return JUDGE_PROMPT.format(
        question=question,
        reference=reference or "（无参考答案，请以检索片段为准）",
        context=(context or "（无检索片段）")[:max_context_chars],
        answer=answer,
        dims_desc=dims_desc,
    )


# --------------------------------------------------------------------- 结果
@dataclass
class JudgeResult:
    scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    raw: str = ""
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.scores)


# --------------------------------------------------------------------- 评委
class LLMJudge:
    """把 LLM 包装成 `core.evaluation.run_answer_eval` 需要的 judge_fn。

    Args:
        llm: 任意有 `chat(messages, **kw)` 的客户端（LLMClient 或假实现）
        context_provider: `(query) -> str`，提供该问题的检索片段文本
        version: Prompt 版本，会写进报告，保证纵向可比
    """

    def __init__(
        self,
        llm: Any,
        context_provider: Optional[Callable[[str], str]] = None,
        version: str = JUDGE_PROMPT_VERSION,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> None:
        self.llm = llm
        self.context_provider = context_provider
        self.version = version
        self.temperature = temperature
        self.use_cache = use_cache
        self.stats = {"calls": 0, "errors": 0, "parse_failures": 0}

    def judge_text(self, question: str, answer: str, reference: str = "",
                   context: str = "") -> JudgeResult:
        started = time.perf_counter()
        prompt = build_judge_prompt(question, answer, reference, context)
        messages = [
            {"role": "system", "content": "你是严谨、客观的评测员，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ]
        self.stats["calls"] += 1
        try:
            result = self.llm.chat(messages, temperature=self.temperature,
                                   use_cache=self.use_cache)
            text = getattr(result, "text", "") or ""
        except Exception as exc:  # noqa: BLE001 - 评委失败不该中断整轮评测
            self.stats["errors"] += 1
            return JudgeResult(error=f"{type(exc).__name__}: {exc}",
                               latency_ms=(time.perf_counter() - started) * 1000)

        scores = parse_judge_output(text)
        latency = (time.perf_counter() - started) * 1000
        if not scores:
            self.stats["parse_failures"] += 1
            return JudgeResult(error="判分输出无法解析", raw=text,
                               latency_ms=latency)

        reason = str(scores.pop("reason", ""))
        return JudgeResult(scores=scores, reason=reason, raw=text,
                           latency_ms=latency)

    def __call__(self, item: Any, answer: str) -> Dict[str, float]:
        """judge_fn 签名：(EvalItem, answer) -> {维度: 分数}"""
        context = ""
        if self.context_provider is not None:
            try:
                context = self.context_provider(getattr(item, "query", "")) or ""
            except Exception:  # noqa: BLE001
                context = ""
        reference = getattr(item, "gold_answer", "") or ""
        result = self.judge_text(getattr(item, "query", ""), answer,
                                 reference, context)
        return dict(result.scores)

    def report(self) -> Dict[str, Any]:
        return {
            "judge_prompt_version": self.version,
            "dimensions": list(JUDGE_DIMENSIONS),
            "calls": self.stats["calls"],
            "errors": self.stats["errors"],
            "parse_failures": self.stats["parse_failures"],
        }


def context_from_hits(hits: Sequence[Dict[str, Any]],
                      max_chars: int = 2000) -> str:
    """把检索命中拼成评委看的"检索片段"。"""
    parts: List[str] = []
    total = 0
    for i, h in enumerate(hits or [], start=1):
        text = (h.get("text") or "").strip()
        if not text:
            continue
        piece = f"[{i}] {text}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts)


def make_context_provider(rag_service: Any,
                          max_chars: int = 2000) -> Callable[[str], str]:
    """从 RagService 构造 context_provider（复用它的检索结果，不重复检索）。

    注意：这里会真的跑一次检索。为了不双倍开销，评测时建议
    用 `answer_eval_with_trace` 一次性拿到 (answer, hits, citations)。
    """
    def provider(query: str) -> str:
        try:
            raw = rag_service.retrieve(query)
        except Exception:  # noqa: BLE001
            return ""
        from core.service import _hits_from_retrieval
        hits = _hits_from_retrieval(raw, rag_service.doc_lookup,
                                    rag_service.config.top_k)
        return context_from_hits(hits, max_chars)
    return provider
