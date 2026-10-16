# -*- coding: utf-8 -*-
"""core/eval_runner.py —— 一条命令跑完整评测，产出可复现的报告。

对应简历里的「自建 120 条评测集 + 逐层消融实验 + LLM-as-Judge 端到端评测」
以及「可复现的评测数据」。

★ 为什么需要这一层（而不是继续写 demo）：

  demo 的问题是**不可复现**：改了 chunk_size 再跑一次，没人知道这次的数字
  对应哪套参数。所以这里强制把**运行指纹**写进报告：

      eval_manifest.json = {
        "run_id":           唯一 ID
        "timestamp":        运行时间
        "eval_set":         {"path", "sha256", "n", "types"}
        "ingest":           {"chunk_size","overlap","parent_size","embedding_model",
                             "config_fingerprint"}
        "retrieval_configs": 每档的开关组合
        "judge":            {"prompt_version", "model", "calls", "errors"}
        "ablations":        逐档指标
      }

  这样任何一张表都能追溯回"哪一套参数 + 哪一份评测集"，面试时可以直接说
  "这些数字是在指纹 X 下跑的，报告在 reports/<run_id>/"。

  另外它把三件事串起来：检索消融 → 生成层评测（含 LLM-as-Judge）→ 汇总报告。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .evaluation import (
    EvalItem, EvalReport, load_eval_set, render_ablation_table,
    render_failure_cases, render_type_table, run_answer_eval,
    run_retrieval_eval, save_eval_set, validate_eval_set, write_report,
)
from .metrics import aggregate, latency_stats


# --------------------------------------------------------------------- 指纹
def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


# --------------------------------------------------------------------- 配置
@dataclass
class EvalRunConfig:
    """一次评测运行的完整参数。全部会被写进 manifest。"""

    eval_set_path: str = "data/eval/eval.jsonl"
    out_dir: str = "reports"
    run_id: Optional[str] = None
    top_k: int = 5
    recall_k: int = 20
    k_list: Sequence[int] = (5,)
    ingest_meta: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = "unknown"
    judge_enabled: bool = False
    judge_model: str = ""
    judge_prompt_version: str = ""
    notes: str = ""

    def resolved_run_id(self) -> str:
        return self.run_id or new_run_id()


# --------------------------------------------------------------------- 结果
@dataclass
class EvalRunResult:
    run_id: str
    out_dir: str
    manifest: Dict[str, Any]
    retrieval_reports: List[EvalReport] = field(default_factory=list)
    answer_metrics: Optional[Dict[str, float]] = None
    summary_path: str = ""
    manifest_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "out_dir": self.out_dir,
            "summary_path": self.summary_path,
            "manifest_path": self.manifest_path,
            "retrieval": [
                {"config": r.config_name, "n": r.n, "overall": r.overall,
                 "latency": r.latency}
                for r in self.retrieval_reports
            ],
            "answer_metrics": self.answer_metrics,
        }


# --------------------------------------------------------------------- 主流程
def run_eval(
    items: Sequence[EvalItem],
    configs: Sequence[Any],
    retrieve_fn_factory: Callable[[Any], Callable[[str], Any]],
    config: Optional[EvalRunConfig] = None,
    answer_fn: Optional[Callable[[str], str]] = None,
    judge_fn: Optional[Callable[[EvalItem, str], Dict[str, float]]] = None,
    judge_report: Optional[Callable[[], Dict[str, Any]]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> EvalRunResult:
    """跑完整评测并写出报告目录。

    Args:
        items: 评测集
        configs: 检索配置列表（见 `core.pipeline.ablation_configs`）
        retrieve_fn_factory: `cfg -> (query -> 检索结果)`，便于按档切换链路
        answer_fn: `query -> 回答`（生成层评测用；为 None 则跳过）
        judge_fn: `(item, answer) -> 四维分`（LLM-as-Judge；为 None 则跳过）
        judge_report: 返回评委统计（calls / errors / prompt_version）
        progress: 进度回调（CLI 打印用）
    """
    config = config or EvalRunConfig()
    run_id = config.resolved_run_id()
    run_dir = os.path.join(config.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    # ---------------- 评测集体检 ----------------
    health = validate_eval_set(items)
    say(f"评测集：{health['total']} 条（可答 {health['answerable']} / "
        f"不可答 {health['unanswerable']}），类型 {health['by_type']}")
    if health["problems"]:
        say(f"⚠️ 评测集体检发现 {len(health['problems'])} 个问题")
        for p in health["problems"][:10]:
            say(f"   · {p}")

    # ---------------- 检索消融 ----------------
    reports: List[EvalReport] = []
    for cfg in configs:
        name = getattr(cfg, "name", str(cfg))
        label = getattr(cfg, "label", lambda: "")()
        say(f"跑检索配置：{name}" + (f"（{label}）" if label else ""))
        rep = run_retrieval_eval(items, retrieve_fn_factory(cfg),
                                 config_name=(f"{name} ({label})" if label else name),
                                 k=config.top_k)
        reports.append(rep)

    # 逐档报告
    for rep in reports:
        safe = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in rep.config_name)
        write_report(rep, os.path.join(run_dir, f"retrieval_{safe}.md"),
                     k=config.top_k)

    # 多 k 值（可选）
    multi_k_tables: Dict[str, str] = {}
    for k in config.k_list:
        if k == config.top_k:
            continue
        extra = [run_retrieval_eval(items, retrieve_fn_factory(cfg),
                                    config_name=getattr(cfg, "name", str(cfg)), k=k)
                 for cfg in configs]
        multi_k_tables[f"k={k}"] = render_ablation_table(extra, k=k)
        say(f"补跑 k={k} 完成")

    # ---------------- 生成层评测 ----------------
    answer_metrics: Optional[Dict[str, float]] = None
    if answer_fn is not None and judge_fn is not None:
        say("跑生成层评测（含 LLM-as-Judge）")
        answer_metrics = run_answer_eval(items, answer_fn, judge_fn)
    elif answer_fn is not None:
        say("跑生成层评测（无评委，仅拒答正确率）")
        answer_metrics = run_answer_eval(items, answer_fn,
                                         lambda i, a: {})

    # ---------------- manifest ----------------
    eval_meta: Dict[str, Any] = {
        "n": health["total"],
        "answerable": health["answerable"],
        "unanswerable": health["unanswerable"],
        "by_type": health["by_type"],
    }
    if config.eval_set_path and os.path.exists(config.eval_set_path):
        eval_meta["path"] = config.eval_set_path
        eval_meta["sha256"] = file_sha256(config.eval_set_path)
        # 报告里存相对路径：绝对路径会把本机目录结构写进交付物，
        # 而且换机器后无法复现（报告是给别人看的）。
        try:
            eval_meta["path_rel"] = os.path.relpath(config.eval_set_path)
        except ValueError:
            eval_meta["path_rel"] = os.path.basename(config.eval_set_path)

    # ★ 合并顺序很重要：config.embedding_model 的默认值是 "unknown"，
    #   如果直接覆盖 ingest_meta，就会把建库时真实用的模型名冲掉
    #   （报告里出现过 embedding_model='unknown' 就是这个原因）。
    #   正确做法：显式值优先，其次用 ingest_meta 里的，最后才是 unknown。
    effective_model = (config.embedding_model
                       if config.embedding_model and config.embedding_model != "unknown"
                       else (config.ingest_meta or {}).get("embedding_model")
                       or config.embedding_model or "unknown")
    ingest_block = dict(config.ingest_meta or {})
    ingest_block["embedding_model"] = effective_model

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": config.notes,
        "eval_set": eval_meta,
        "ingest": ingest_block,
        "retrieval": {
            "top_k": config.top_k,
            "recall_k": config.recall_k,
            "k_list": list(config.k_list),
            "configs": [
                {"name": getattr(c, "name", str(c)),
                 "use_vector": getattr(c, "use_vector", None),
                 "use_bm25": getattr(c, "use_bm25", None),
                 "use_rerank": getattr(c, "use_rerank", None),
                 "use_rules": getattr(c, "use_rules", None)}
                for c in configs
            ],
        },
        "results": [
            {"config": r.config_name, "n": r.n, "overall": r.overall,
             "latency": r.latency, "by_type": r.by_type}
            for r in reports
        ],
        "answer_metrics": answer_metrics,
        "judge": (judge_report() if judge_report else None),
        "eval_set_health": {"problems": health["problems"]},
    }

    manifest_path = os.path.join(run_dir, "eval_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 评测集副本（保证"这份报告对应这一份评测集"）
    try:
        save_eval_set(items, os.path.join(run_dir, "eval_set.jsonl"))
    except OSError:
        pass

    # ---------------- 汇总报告 ----------------
    summary_path = os.path.join(run_dir, "SUMMARY.md")
    _write_summary(summary_path, manifest, reports, answer_metrics,
                   multi_k_tables, config)

    say(f"报告目录：{run_dir}")
    return EvalRunResult(run_id=run_id, out_dir=run_dir, manifest=manifest,
                         retrieval_reports=reports, answer_metrics=answer_metrics,
                         summary_path=summary_path, manifest_path=manifest_path)


def _write_summary(path: str, manifest: Dict[str, Any],
                   reports: Sequence[EvalReport],
                   answer_metrics: Optional[Dict[str, float]],
                   multi_k_tables: Dict[str, str],
                   config: EvalRunConfig) -> None:
    """写一份自包含的 SUMMARY —— 面试时直接打开这一个文件就够。"""
    m = manifest
    lines: List[str] = []
    lines.append(f"# 评测总结 · {m['run_id']}")
    lines.append("")
    lines.append(f"- 运行时间：**{m['timestamp']}**")
    lines.append(f"- 评测集：**{m['eval_set']['n']} 条**"
                 f"（可答 {m['eval_set']['answerable']} / "
                 f"不可答 {m['eval_set']['unanswerable']}）")
    lines.append(f"- 类型分布：`{m['eval_set']['by_type']}`")
    if m["eval_set"].get("sha256"):
        shown = m["eval_set"].get("path_rel") or m["eval_set"]["path"]
        lines.append(f"- 评测集指纹：`{m['eval_set']['sha256']}`（`{shown}`）")
    lines.append(f"- 嵌入模型：`{m['ingest'].get('embedding_model')}`")
    ing = m["ingest"]
    cfg_bits = [f"{k}={ing[k]}" for k in ("chunk_size", "overlap", "parent_size")
                if ing.get(k) is not None]
    if ing.get("config_fingerprint"):
        cfg_bits.append(f"指纹=`{ing['config_fingerprint']}`")
    if cfg_bits:
        lines.append(f"- 建库配置：{', '.join(cfg_bits)}")
    lines.append(f"- 检索口径：top_k={m['retrieval']['top_k']}, "
                 f"recall_k={m['retrieval']['recall_k']}")
    lines.append("")
    lines.append("> 指标口径：MRR 为 **1-based** 排名倒数；Precision 分母固定为 k；"
                 "幻觉率按忠实度 **严格小于** 阈值计。")
    lines.append("")

    lines.append("## 一、检索消融（核心结果）")
    lines.append("")
    lines.append(render_ablation_table(reports, k=config.top_k))
    lines.append("")

    if multi_k_tables:
        lines.append("### 多 k 值对比")
        lines.append("")
        for label, table in multi_k_tables.items():
            lines.append(f"**{label}**")
            lines.append("")
            lines.append(table)
            lines.append("")

    if answer_metrics:
        lines.append("## 二、生成层评测（含 LLM-as-Judge）")
        lines.append("")
        judged = answer_metrics.get("judged", 0)
        if not judged:
            # ★ 没跑评委时必须显式说明，**不能**把 0.00 当结果写进报告。
            #   报告是交付物、会被拿去讨论，一个"忠实度 0.00"或
            #   "幻觉率 0.00%" 会被读成完全相反的两个结论 ——
            #   而真相是"根本没测"。区分「测出来是 0」与「没测」
            #   是评测报告最基本的诚实要求。
            lines.append("> ⚠️ **本轮未启用 LLM-as-Judge**：忠实度与幻觉率"
                         "**未测评**（不是 0）。")
            lines.append("> 下表只有拒答正确率；四维分数需要 `--judge` "
                         "且配置 `DEEPSEEK_API_KEY`。")
            lines.append("")
        lines.append("| 指标 | 值 | 含义 |")
        lines.append("|---|---|---|")
        desc = {
            "refusal_accuracy": "拒答判定正确率（可答不拒答、不可答要拒答）",
            "relevance_mean": "相关性均分（1-5）",
            "accuracy_mean": "准确性均分（1-5）",
            "completeness_mean": "完整性均分（1-5）",
            "groundedness_mean": "忠实度均分（1-5，专测幻觉）",
            "hallucination_rate": "幻觉率 = 忠实度 < 阈值 的占比",
            "judged": "参与判分的样本数（0 = 未启用评委）",
        }
        unjudged = ("relevance_mean", "accuracy_mean", "completeness_mean",
                    "groundedness_mean", "hallucination_rate")
        for k, v in answer_metrics.items():
            if not judged and k in unjudged:
                lines.append(f"| {k} | _未测评_ | {desc.get(k, '')} |")
            else:
                lines.append(f"| {k} | {v} | {desc.get(k, '')} |")
        lines.append("")
        if m.get("judge"):
            j = m["judge"]
            lines.append(f"评委配置：prompt `{j.get('judge_prompt_version')}`，"
                         f"调用 {j.get('calls')} 次，失败 {j.get('errors')} 次，"
                         f"解析失败 {j.get('parse_failures')} 次。")
            lines.append("")
        if judged:
            lines.append("> ⚠️ LLM-as-Judge 存在位置偏见与自我偏好，本表用于"
                         "**回归筛选**（快速发现版本退化），关键结论仍需人工抽检。")
            lines.append("")

    lines.append("## 三、分类型表现（哪个类型最弱）")
    lines.append("")
    for r in reports:
        lines.append(f"**{r.config_name}**")
        lines.append("")
        lines.append(render_type_table(r, k=config.top_k))
        lines.append("")

    lines.append("## 四、失败样本（Badcase 回流清单）")
    lines.append("")
    for r in reports[-1:]:
        lines.append(f"**{r.config_name}**")
        lines.append("")
        lines.append(render_failure_cases(r, limit=20))
        lines.append("")

    problems = (m.get("eval_set_health") or {}).get("problems") or []
    lines.append("## 五、评测集体检")
    lines.append("")
    if problems:
        lines.append(f"发现 **{len(problems)}** 个问题（需人工复核）：")
        lines.append("")
        for p in problems[:30]:
            lines.append(f"- {p}")
    else:
        lines.append("✅ 未发现问题。")
    lines.append("")
    lines.append("## 六、复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append("# 本报告对应的完整参数见 eval_manifest.json")
    lines.append("python -m rag1_knowledge_qa.eval --eval-set "
                 f"{m['eval_set'].get('path_rel') or m['eval_set'].get('path', 'data/eval/eval.jsonl')} \\")
    lines.append("    --ablation --judge --out reports")
    lines.append("```")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------- 辅助
def load_items_or_raise(path: str) -> List[EvalItem]:
    items = load_eval_set(path)
    if not items:
        raise ValueError(f"评测集为空：{path}")
    return items


def latest_run_dir(out_dir: str = "reports") -> Optional[str]:
    """找最近一次 run（目录名形如 run-YYYYmmdd-HHMMSS-xxxxxx）。"""
    if not os.path.isdir(out_dir):
        return None
    runs = [d for d in os.listdir(out_dir)
            if d.startswith("run-") and os.path.isdir(os.path.join(out_dir, d))]
    if not runs:
        return None
    runs.sort()
    return os.path.join(out_dir, runs[-1])


def compare_runs(run_a: str, run_b: str) -> str:
    """对比两次 run 的指标差异 —— 回答"这次改动有没有让效果变好"。

    ★ 这是消融之外最有用的功能：改了 prompt / 分块 / 召回参数后，
      一眼看出哪档涨了、哪档跌了、跌了多少。
    """
    def load(dirpath: str) -> Dict[str, Any]:
        p = os.path.join(dirpath, "eval_manifest.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到 manifest: {p}")
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    ma, mb = load(run_a), load(run_b)
    keys = ["hit@5", "mrr@5", "recall@5", "precision@5", "ndcg@5"]
    lines = [f"# 两次评测对比", "",
             f"- A（基线）：`{ma['run_id']}` @ {ma['timestamp']}",
             f"- B（对照）：`{mb['run_id']}` @ {mb['timestamp']}", ""]
    ra = {r["config"]: r["overall"] for r in ma.get("results", [])}
    rb = {r["config"]: r["overall"] for r in mb.get("results", [])}
    common = [c for c in ra if c in rb]
    if not common:
        lines.append("_两次运行的配置名不重合，无法逐档对比。_")
        return "\n".join(lines)

    lines.append("| 配置 | 指标 | A | B | Δ |")
    lines.append("|---|---|---|---|---|")
    for cfg in common:
        for key in keys:
            va = ra[cfg].get(key)
            vb = rb[cfg].get(key)
            if va is None or vb is None:
                continue
            delta = vb - va
            mark = "🟢" if delta > 0.001 else ("🔴" if delta < -0.001 else "⚪")
            lines.append(f"| {cfg} | {key} | {va:.4f} | {vb:.4f} "
                         f"| {mark} {delta:+.4f} |")
    lines.append("")
    lines.append("> 没有统计显著性检验，小样本下 Δ < 0.02 不建议当作结论。")
    return "\n".join(lines)
