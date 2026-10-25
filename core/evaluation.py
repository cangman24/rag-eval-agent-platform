# -*- coding: utf-8 -*-
"""core/evaluation.py —— ★ Day 3 的核心：评测框架。

为什么它是整个项目最值钱的部分：
  简历上每一个检索指标都由这里产出。它的输出（reports/*.md）就是
  面试时你打开给面试官看的东西 —— 也是"你怎么知道它变好了"的答案。

设计要点：
  1. **一次评测产出多张表**：逐条明细、分类型汇总、消融对比、延迟分位。
  2. **分类型统计**（事实/多跳/术语/越界）—— 只报一个总分说服力弱，
     能说出"多跳类最差，因为…"才是真做过。
  3. **可复现**：记录配置哈希、时间戳，避免"这张表是哪次跑的"。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .metrics import (
    aggregate, all_retrieval_metrics, hallucination_rate, latency_stats,
    refusal_correctness,
)


# ------------------------------------------------------------------ 数据结构
@dataclass
class EvalItem:
    """一条评测样本。"""

    qid: str
    query: str
    gold_ids: List[str] = field(default_factory=list)
    gold_answer: str = ""
    qtype: str = "fact"            # fact / multi_hop / term / out_of_scope
    is_answerable: bool = True

    @staticmethod
    def from_dict(d: dict) -> "EvalItem":
        return EvalItem(
            qid=str(d.get("qid") or d.get("id") or ""),
            query=d.get("query", ""),
            gold_ids=list(d.get("gold_ids") or []),
            gold_answer=d.get("gold_answer", "") or "",
            qtype=d.get("type") or d.get("qtype") or "fact",
            is_answerable=bool(d.get("is_answerable", True)),
        )

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "query": self.query, "gold_ids": self.gold_ids,
            "gold_answer": self.gold_answer, "type": self.qtype,
            "is_answerable": self.is_answerable,
        }


def load_eval_set(path: str) -> List[EvalItem]:
    """读取 JSONL 或 JSON 评测集。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"评测集不存在: {path}")
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        return []
    if path.endswith(".json") or text.startswith("["):
        data = json.loads(text)
        return [EvalItem.from_dict(d) for d in data]
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(EvalItem.from_dict(json.loads(line)))
    return items


def save_eval_set(items: Sequence[EvalItem], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def validate_eval_set(items: Sequence[EvalItem]) -> Dict[str, object]:
    """体检评测集。★ 人工复核的辅助工具，别跳过。

    返回问题清单：qid 重复、query 为空、可答但没标 gold、不可答却有 gold 等。
    """
    problems: List[str] = []
    seen = set()
    for it in items:
        if not it.query.strip():
            problems.append(f"{it.qid}: query 为空")
        if it.qid in seen:
            problems.append(f"{it.qid}: qid 重复")
        seen.add(it.qid)
        if it.is_answerable and not it.gold_ids:
            problems.append(f"{it.qid}: 可答但未标注 gold_ids")
        if not it.is_answerable and it.gold_ids:
            problems.append(f"{it.qid}: 不可答却标了 gold_ids")
        if not it.is_answerable and it.qtype != "out_of_scope":
            problems.append(f"{it.qid}: 不可答样本的 type 建议设为 out_of_scope")

    by_type: Dict[str, int] = {}
    for it in items:
        by_type[it.qtype] = by_type.get(it.qtype, 0) + 1
    answerable = sum(1 for it in items if it.is_answerable)
    return {
        "total": len(items),
        "answerable": answerable,
        "unanswerable": len(items) - answerable,
        "by_type": by_type,
        "problems": problems,
        "ok": not problems,
    }


# ------------------------------------------------------------------ 评测执行
@dataclass
class EvalReport:
    config_name: str
    overall: Dict[str, float]
    by_type: Dict[str, Dict[str, float]]
    latency: Dict[str, float]
    details: List[dict]
    n: int
    answer_metrics: Optional[Dict[str, float]] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "config": self.config_name, "n": self.n, "overall": self.overall,
            "by_type": self.by_type, "latency": self.latency,
            "answer_metrics": self.answer_metrics, "timestamp": self.timestamp,
        }


def run_retrieval_eval(
    items: Sequence[EvalItem],
    retrieve_fn: Callable[[str], object],
    config_name: str = "default",
    k: int = 5,
    latency_key: str = "latency_ms",
    doc_ids_key: str = "doc_ids",
) -> EvalReport:
    """对检索器跑一遍评测。

    Args:
        retrieve_fn: query -> 带 doc_ids / latency_ms 的对象，或 (ids, latency) 元组，
            或直接是 id 列表。
    """
    per_query: List[Dict[str, float]] = []
    details: List[dict] = []
    latencies: List[float] = []
    per_type: Dict[str, List[Dict[str, float]]] = {}

    for it in items:
        if not it.is_answerable:
            # 不可答样本不计检索指标（没有 gold）；单独在答案层记拒答正确率
            continue
        raw = retrieve_fn(it.query)
        if isinstance(raw, tuple):
            ids, lat = list(raw[0]), float(raw[1]) if len(raw) > 1 else 0.0
        elif hasattr(raw, doc_ids_key):
            ids, lat = list(getattr(raw, doc_ids_key)), float(getattr(raw, latency_key, 0.0))
        elif isinstance(raw, dict):
            ids, lat = list(raw.get(doc_ids_key, [])), float(raw.get(latency_key, 0.0))
        else:
            ids, lat = list(raw), 0.0

        m = all_retrieval_metrics(ids, it.gold_ids, k=k)
        per_query.append(m)
        per_type.setdefault(it.qtype, []).append(m)
        latencies.append(lat)
        details.append({
            "qid": it.qid, "type": it.qtype, "query": it.query,
            "gold_ids": it.gold_ids, "retrieved": ids[:k],
            "hit": m[f"hit@{k}"], "mrr": m[f"mrr@{k}"], "latency_ms": round(lat, 2),
        })

    overall = aggregate(per_query)
    by_type = {t: aggregate(rows) for t, rows in sorted(per_type.items())}
    return EvalReport(
        config_name=config_name,
        overall=overall,
        by_type=by_type,
        latency=latency_stats(latencies),
        details=details,
        n=len(per_query),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def run_answer_eval(
    items: Sequence[EvalItem],
    answer_fn: Callable[[str], str],
    judge_fn: Callable[[EvalItem, str], Dict[str, float]],
    refusal_marker: str = "未找到",
) -> Dict[str, float]:
    """评测生成层：拒答正确率 + LLM-as-Judge 四维分 + 幻觉率。

    Args:
        answer_fn: query -> 模型回答
        judge_fn: (item, answer) -> {"relevance":..,"accuracy":..,
                                    "completeness":..,"groundedness":..}
    """
    refusal_scores: List[float] = []
    dims: Dict[str, List[float]] = {}
    groundedness: List[float] = []

    for it in items:
        answer = answer_fn(it.query) or ""
        refusal_scores.append(refusal_correctness(it.is_answerable, answer, refusal_marker))
        if not it.is_answerable or not answer.strip():
            continue
        try:
            scores = judge_fn(it, answer) or {}
        except Exception:  # noqa: BLE001 - 判分失败按 0 计，不中断整轮评测
            scores = {}
        for dim in ("relevance", "accuracy", "completeness", "groundedness"):
            if dim in scores:
                dims.setdefault(dim, []).append(float(scores[dim]))
        if "groundedness" in scores:
            groundedness.append(float(scores["groundedness"]))

    out: Dict[str, float] = {
        "refusal_accuracy": round(sum(refusal_scores) / len(refusal_scores), 4)
        if refusal_scores else 0.0,
        "judged": float(len(groundedness)),
    }
    for dim, vals in dims.items():
        out[f"{dim}_mean"] = round(sum(vals) / len(vals), 4) if vals else 0.0
    out["hallucination_rate"] = round(hallucination_rate(groundedness), 4)
    return out


# ------------------------------------------------------------------ 报告渲染
def render_ablation_table(reports: Sequence[EvalReport], k: int = 5) -> str:
    """把多档配置渲染成消融对比表（★ 直接贴进简历/README）。"""
    keys = [f"hit@{k}", f"mrr@{k}", f"recall@{k}", f"precision@{k}", f"ndcg@{k}"]
    header = "| 配置 | " + " | ".join(keys) + " | n | P95 延迟(ms) |"
    sep = "|" + "---|" * (len(keys) + 3)
    lines = [header, sep]
    for r in reports:
        cells = [f"{r.overall.get(key, 0.0):.4f}" for key in keys]
        lines.append(
            f"| {r.config_name} | " + " | ".join(cells)
            + f" | {r.n} | {r.latency.get('p95_ms', 0.0):.1f} |"
        )
    return "\n".join(lines)


def render_type_table(report: EvalReport, k: int = 5) -> str:
    """分类型汇总表 —— 体现"我知道哪类问题最弱"。"""
    lines = [f"| 问题类型 | n | hit@{k} | mrr@{k} | ndcg@{k} |", "|---|---|---|---|---|"]
    for qtype, m in report.by_type.items():
        n = sum(1 for d in report.details if d["type"] == qtype)
        lines.append(
            f"| {qtype} | {n} | {m.get(f'hit@{k}', 0):.4f} "
            f"| {m.get(f'mrr@{k}', 0):.4f} | {m.get(f'ndcg@{k}', 0):.4f} |"
        )
    return "\n".join(lines)


def render_failure_cases(report: EvalReport, limit: int = 15) -> str:
    """列出失败样本 —— Badcase 回流清单，也是面试时"我知道它哪里不行"的证据。"""
    bad = [d for d in report.details if d["hit"] == 0.0]
    if not bad:
        return "_本轮无完全失败样本（hit@k = 0）_"
    lines = [f"共 {len(bad)} 条，下列前 {min(limit, len(bad))} 条：", "",
             "| qid | 类型 | query | gold | 实际召回 top5 |", "|---|---|---|---|---|"]
    for d in bad[:limit]:
        lines.append(
            f"| {d['qid']} | {d['type']} | {d['query'][:28]} "
            f"| {','.join(d['gold_ids'])} | {','.join(d['retrieved'][:5])} |"
        )
    return "\n".join(lines)


def write_report(report: EvalReport, path: str, extra_sections: Optional[Dict[str, str]] = None,
                 k: int = 5) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    parts = [
        f"# 检索评测报告 · {report.config_name}",
        "",
        f"- 运行时间：{report.timestamp}",
        f"- 参与评测的样本数：{report.n}（不可答样本不计入检索指标）",
        f"- 指标口径：MRR 为 1-based 排名倒数；Precision 分母固定为 k={k}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for key, val in report.overall.items():
        parts.append(f"| {key} | {val:.4f} |")
    parts += ["", "## 延迟", "", "| 指标 | ms |", "|---|---|"]
    for key, val in report.latency.items():
        parts.append(f"| {key} | {val:.1f} |")
    parts += ["", "## 分类型汇总", "", render_type_table(report, k=k), "",
              "## 失败样本（Badcase 回流清单）", "", render_failure_cases(report)]
    if extra_sections:
        for title, body in extra_sections.items():
            parts += ["", f"## {title}", "", body]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
