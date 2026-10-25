# -*- coding: utf-8 -*-
"""core/agent_eval.py —— Agent 层评测（项目二的消融与指标）。

对应简历里的「评测体系与消融实验：自建 150 条测试集，对 Baseline /
+混合检索 / +记忆 / +上下文压缩 / Full Harness 做消融实验；计算成功率、
Model Rounds、Tool Calls/Errors、Token、Latency 等 6 类指标」。

★ 为什么 Agent 层评测不能复用检索层的那套：

  检索层的指标是 Hit@5 / MRR —— 衡量"正确片段有没有进候选"。
  Agent 层关心的是完全不同的东西：

      任务成功率   —— 多步任务最终有没有闭环
      模型轮次     —— 每多一轮就是一次 API 调用（成本与延迟的直接来源）
      工具调用次数 —— 链路过长会显著抬高失败概率
      工具失败次数 —— 工具层稳定性
      Token 总量   —— 上下文预算是否失控
      P95 延迟     —— 用户体验

  把这两层混成一个"准确率"是最常见的错误（面试也爱问：召回提升≠准确率提升）。

★ 六类指标里最有信息量的是「轮次」和「工具失败」：
  成功率相同但轮次翻倍，说明任务的"过程质量"差很多 —— 这是只看成功率的
  评测永远发现不了的。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .metrics import latency_stats
from .retrieval import count_tokens_approx

# 六类指标（与简历一致）
AGENT_METRIC_KEYS = (
    "success_rate",      # 任务成功率
    "avg_rounds",        # 平均模型轮次
    "avg_tool_calls",    # 平均工具调用次数
    "avg_tool_errors",   # 平均工具失败次数
    "avg_tokens",        # 平均输入 Token 估算
    "p95_latency_ms",    # P95 延迟
)


# --------------------------------------------------------------------- 样本
@dataclass
class AgentTask:
    """一条 Agent 任务（多步性比检索评测更强）。"""

    tid: str
    query: str
    expect_tools: List[str] = field(default_factory=list)   # 期望调用的工具（子序列）
    must_contain: List[str] = field(default_factory=list)   # 回答必须包含的关键词
    must_not_contain: List[str] = field(default_factory=list)  # 回答必须不含（如幻觉词）
    task_type: str = "general"        # 期望意图（用于检查路由是否正确）
    max_rounds: int = 6               # 该任务允许的最大轮次（超了算低效）
    note: str = ""

    @staticmethod
    def from_dict(d: dict) -> "AgentTask":
        return AgentTask(
            tid=str(d.get("tid") or d.get("id") or ""),
            query=d.get("query", ""),
            expect_tools=list(d.get("expect_tools") or []),
            must_contain=list(d.get("must_contain") or []),
            must_not_contain=list(d.get("must_not_contain") or []),
            task_type=d.get("task_type", "general"),
            max_rounds=int(d.get("max_rounds", 6)),
            note=d.get("note", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"tid": self.tid, "query": self.query,
                "expect_tools": self.expect_tools,
                "must_contain": self.must_contain,
                "must_not_contain": self.must_not_contain,
                "task_type": self.task_type, "max_rounds": self.max_rounds,
                "note": self.note}


def load_agent_tasks(path: str) -> List[AgentTask]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent 任务集不存在: {path}")
    text = open(path, "r", encoding="utf-8").read().strip()
    if not text:
        return []
    if path.endswith(".json") or text.startswith("["):
        return [AgentTask.from_dict(d) for d in json.loads(text)]
    return [AgentTask.from_dict(json.loads(line))
            for line in text.splitlines() if line.strip()]


def save_agent_tasks(tasks: Sequence[AgentTask], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def validate_agent_tasks(tasks: Sequence[AgentTask]) -> Dict[str, Any]:
    """体检任务集（与检索集一样，人工复核前先跑一遍）。"""
    problems: List[str] = []
    seen = set()
    for t in tasks:
        if not t.query.strip():
            problems.append(f"{t.tid}: query 为空")
        if t.tid in seen:
            problems.append(f"{t.tid}: tid 重复")
        seen.add(t.tid)
        if not t.must_contain and not t.expect_tools:
            problems.append(f"{t.tid}: 既没有 must_contain 也没有 expect_tools，无法判定成功")
        if t.max_rounds <= 0:
            problems.append(f"{t.tid}: max_rounds 必须 > 0")
    by_type: Dict[str, int] = {}
    for t in tasks:
        by_type[t.task_type] = by_type.get(t.task_type, 0) + 1
    return {"total": len(tasks), "by_type": by_type,
            "problems": problems, "ok": not problems}


# --------------------------------------------------------------------- 判定
def judge_task_success(task: AgentTask, result: Dict[str, Any],
                       tool_sequence: Sequence[str]) -> tuple:
    """判定单条任务是否成功。返回 (ok, reason)。

    成功条件（全部满足）：
      ① Agent 正常收尾（finished，且不是 error / no_decision）
      ② expected_tools 是实际**成功执行**的工具序列的子序列
         （★ 顺序正确但允许穿插别的工具；被白名单拦下的调用不算数）
      ③ must_contain 的关键词都出现
      ④ must_not_contain 的关键词都不出现

    注意 `tool_sequence` 只应包含**真正执行成功**的工具。
    `run_agent_eval` 负责过滤掉失败/被拦的调用 —— 这个区分很重要：
    如果任务要求"必须检索过"，而那次检索被白名单拦下了，
    任务绝不能判定为成功（否则 baseline 消融会虚高，
    被问到就无法解释）。这个 bug 是被 test_tool_expectation_fails_when_tool_blocked 抓到的。
    """
    if not result.get("finished"):
        return False, f"未收尾（stop_reason={result.get('stop_reason')}）"
    if result.get("error"):
        return False, f"执行错误: {result['error']}"

    answer = result.get("answer") or ""

    # ② 工具子序列匹配（顺序敏感、允许穿插）
    if task.expect_tools:
        it = iter(tool_sequence)
        if not all(tool in it for tool in task.expect_tools):
            return False, (f"工具序列不符：期望子序列 {task.expect_tools}，"
                           f"实际成功执行 {list(tool_sequence)}")

    # ③ 关键词必须出现
    missing = [k for k in task.must_contain if k not in answer]
    if missing:
        return False, f"回答缺少关键内容 {missing}"

    # ④ 禁止词不能出现
    forbidden = [k for k in task.must_not_contain if k in answer]
    if forbidden:
        return False, f"回答包含禁止内容 {forbidden}"

    return True, ""


# --------------------------------------------------------------------- 结果
@dataclass
class AgentEvalRow:
    """单条任务的评测明细。"""

    tid: str
    task_type: str
    query: str
    success: bool
    reason: str
    rounds: int
    tool_calls: int
    tool_errors: int
    blocked_calls: int
    tokens: int
    latency_ms: float
    task_type_actual: str
    tool_sequence: List[str] = field(default_factory=list)
    answer: str = ""
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tid": self.tid, "type": self.task_type, "query": self.query,
            "success": self.success, "reason": self.reason,
            "rounds": self.rounds, "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors, "blocked_calls": self.blocked_calls,
            "tokens": self.tokens, "latency_ms": round(self.latency_ms, 1),
            "task_type_actual": self.task_type_actual,
            "tool_sequence": self.tool_sequence,
            "stop_reason": self.stop_reason,
            "answer": self.answer[:200],
        }


@dataclass
class AgentEvalReport:
    config_name: str
    n: int
    metrics: Dict[str, float]
    by_type: Dict[str, Dict[str, float]]
    rows: List[AgentEvalRow] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"config": self.config_name, "n": self.n,
                "metrics": self.metrics, "by_type": self.by_type,
                "timestamp": self.timestamp}


# --------------------------------------------------------------------- 采集
def extract_ok_tool_sequence(run: Any) -> List[str]:
    """只取**真正执行成功**的工具序列（用于成功判定）。

    与 `extract_tool_sequence` 的区别：
      - extract_tool_sequence    含失败与被拦的调用 -> 用于诊断/展示
      - extract_ok_tool_sequence 只含成功的         -> 用于成功判定
    混用会导致"工具被拦也算调用过"，让消融结果虚高。
    """
    if isinstance(run, dict):
        seq = run.get("ok_tool_sequence")
        if seq is not None:
            return [str(x) for x in seq]
        return [str(x) for x in (run.get("tool_sequence") or [])]
    out: List[str] = []
    for item in getattr(getattr(run, "state", None), "scratchpad", []) or []:
        if (item.get("type") == "tool_call" and item.get("tool")
                and item.get("ok") and not item.get("blocked")):
            out.append(str(item["tool"]))
    return out


def extract_tool_sequence(run: Any) -> List[str]:
    """取出实际工具调用序列（含被白名单拦下的）。

    兼容两种输入：
      ① AgentRun / 任何带 `.state.scratchpad` 的对象（生产路径）
      ② dict（当调用方用 `results_to_dict` 自己组装结果时，
         允许直接把 tool_sequence 放在 dict 里）
    """
    if isinstance(run, dict):
        seq = run.get("tool_sequence")
        return [str(x) for x in seq] if seq else []
    seq_out: List[str] = []
    for item in getattr(getattr(run, "state", None), "scratchpad", []) or []:
        if item.get("type") == "tool_call" and item.get("tool"):
            seq_out.append(str(item["tool"]))
    return seq_out


def count_tool_errors(run: Any) -> tuple:
    """返回 (失败次数, 被白名单拦截次数)。"""
    if isinstance(run, dict):
        return int(run.get("tool_errors", 0) or 0), int(run.get("blocked_calls", 0) or 0)
    errors = blocked = 0
    for item in getattr(getattr(run, "state", None), "scratchpad", []) or []:
        if item.get("type") != "tool_call":
            continue
        if item.get("blocked"):
            blocked += 1
        elif item.get("ok") is False:
            errors += 1
    return errors, blocked


def estimate_run_tokens(run: Any) -> int:
    """估算这次执行的输入 token 总量。

    ★ 口径说明：真实项目应累加每次 LLM 调用的 prompt_tokens
      （LLMResult 里有）。这里没有 LLM 时退化为"按 prompt 文本长度估算"，
      对**相对比较**（消融各档谁更省）足够用。
      面试被问口径时要如实说这一点。
    """
    if isinstance(run, dict):
        return int(run.get("tokens", 0) or 0)
    state = getattr(run, "state", None)
    total = 0
    # 若事件里带了 token 统计（真机接 LLMResult 时可补），优先用它
    for ev in getattr(run, "events", []) or []:
        data = ev.get("data") or {}
        if isinstance(data, dict) and data.get("prompt_tokens"):
            total += int(data["prompt_tokens"])
    if total:
        return total
    # 退化：按工具结果的文本量估算（越长的工具返回越吃上下文）
    for item in getattr(state, "scratchpad", []) or []:
        result = item.get("result")
        if result is None:
            continue
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        total += count_tokens_approx(text)
    total += count_tokens_approx(getattr(state, "query", "") or "")
    total += count_tokens_approx(getattr(state, "answer", "") or "")
    return total


# --------------------------------------------------------------------- 执行
def run_agent_eval(
    tasks: Sequence[AgentTask],
    run_fn: Callable[[str], Any],
    config_name: str = "agent",
    results_to_dict: Optional[Callable[[Any], Dict[str, Any]]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> AgentEvalReport:
    """对一批任务跑 Agent 评测。

    Args:
        run_fn: `query -> AgentRun`（或任何带 .state / .events 的对象）
        results_to_dict: 把 run 转成带 finished/answer/error 的 dict；
            默认直接从 run.state 读，所以传 AgentRun 即可
    """
    rows: List[AgentEvalRow] = []
    latencies: List[float] = []

    for task in tasks:
        started = time.perf_counter()
        try:
            run = run_fn(task.query)
            latency = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - 单条任务异常不该中断整批
            rows.append(AgentEvalRow(
                tid=task.tid, task_type=task.task_type, query=task.query,
                success=False, reason=f"run 抛异常: {type(exc).__name__}: {exc}",
                rounds=0, tool_calls=0, tool_errors=0, blocked_calls=0,
                tokens=0, latency_ms=(time.perf_counter() - started) * 1000,
                task_type_actual="", stop_reason="exception"))
            continue

        if results_to_dict is not None:
            info = results_to_dict(run)
        else:
            state = getattr(run, "state", None)
            info = {
                "finished": getattr(state, "finished", False),
                "answer": getattr(state, "answer", "") or "",
                "error": getattr(run, "error", None),
                "stop_reason": getattr(state, "stop_reason", "") or "",
                "rounds": getattr(state, "rounds", 0),
                "tool_calls": getattr(state, "tool_calls", 0),
                "task_type": getattr(state, "task_type", ""),
            }

        seq = extract_tool_sequence(run)
        ok_seq = extract_ok_tool_sequence(run)
        errors, blocked = count_tool_errors(run)
        # ★ 成功判定用 ok_seq（只含成功执行的工具），展示用 seq（含被拦的）
        ok, reason = judge_task_success(task, info, ok_seq)

        rows.append(AgentEvalRow(
            tid=task.tid, task_type=task.task_type, query=task.query,
            success=ok, reason=reason,
            rounds=int(info.get("rounds", 0)),
            tool_calls=int(info.get("tool_calls", 0)),
            tool_errors=errors, blocked_calls=blocked,
            tokens=estimate_run_tokens(run), latency_ms=latency,
            task_type_actual=str(info.get("task_type", "")),
            tool_sequence=seq, answer=str(info.get("answer", "")),
            stop_reason=str(info.get("stop_reason", "")),
        ))
        latencies.append(latency)
        if progress:
            flag = "OK " if ok else "FAIL"
            progress(f"  [{flag}] {task.tid} rounds={rows[-1].rounds} "
                     f"tools={rows[-1].tool_calls} {reason[:50]}")

    metrics = aggregate_agent_metrics(rows, latencies)
    by_type: Dict[str, Dict[str, float]] = {}
    for ttype in sorted({r.task_type for r in rows}):
        subset = [r for r in rows if r.task_type == ttype]
        by_type[ttype] = aggregate_agent_metrics(
            subset, [r.latency_ms for r in subset])

    return AgentEvalReport(config_name=config_name, n=len(rows), metrics=metrics,
                           by_type=by_type, rows=rows,
                           timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))


def aggregate_agent_metrics(rows: Sequence[AgentEvalRow],
                            latencies: Sequence[float]) -> Dict[str, float]:
    """把逐条明细聚合成六类指标。"""
    if not rows:
        return dict.fromkeys(AGENT_METRIC_KEYS, 0.0)
    n = len(rows)
    lat = latency_stats(latencies) if latencies else {}
    return {
        "success_rate": round(sum(1 for r in rows if r.success) / n, 4),
        "avg_rounds": round(sum(r.rounds for r in rows) / n, 2),
        "avg_tool_calls": round(sum(r.tool_calls for r in rows) / n, 2),
        "avg_tool_errors": round(sum(r.tool_errors for r in rows) / n, 2),
        "avg_tokens": round(sum(r.tokens for r in rows) / n, 1),
        "p95_latency_ms": round(lat.get("p95_ms", 0.0), 1),
    }


# --------------------------------------------------------------------- 渲染
def render_agent_ablation(reports: Sequence[AgentEvalReport]) -> str:
    """六类指标的消融对比表（★ 直接贴进 README / 简历）。"""
    header = ("| 配置 | 成功率 | 平均轮次 | 平均工具调用 | 平均工具失败 "
              "| 平均Token | P95延迟(ms) | n |")
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in reports:
        m = r.metrics
        lines.append(
            f"| {r.config_name} | {m.get('success_rate', 0):.4f} "
            f"| {m.get('avg_rounds', 0):.2f} | {m.get('avg_tool_calls', 0):.2f} "
            f"| {m.get('avg_tool_errors', 0):.2f} | {m.get('avg_tokens', 0):.1f} "
            f"| {m.get('p95_latency_ms', 0):.1f} | {r.n} |"
        )
    return "\n".join(lines)


def render_agent_type_table(report: AgentEvalReport) -> str:
    lines = ["| 任务类型 | 成功率 | 平均轮次 | 平均工具调用 | n |",
             "|---|---|---|---|---|"]
    for ttype, m in report.by_type.items():
        n = sum(1 for r in report.rows if r.task_type == ttype)
        lines.append(f"| {ttype} | {m.get('success_rate', 0):.4f} "
                     f"| {m.get('avg_rounds', 0):.2f} "
                     f"| {m.get('avg_tool_calls', 0):.2f} | {n} |")
    return "\n".join(lines)


def render_agent_failures(report: AgentEvalReport, limit: int = 20) -> str:
    bad = [r for r in report.rows if not r.success]
    if not bad:
        return "_本轮无失败任务_"
    lines = [f"共 {len(bad)} 条失败，下列前 {min(limit, len(bad))} 条：", "",
             "| tid | 类型 | query | 原因 | 轮次 | 工具序列 |",
             "|---|---|---|---|---|---|"]
    for r in bad[:limit]:
        lines.append(f"| {r.tid} | {r.task_type} | {r.query[:24]} "
                     f"| {r.reason[:60]} | {r.rounds} "
                     f"| {','.join(r.tool_sequence)} |")
    return "\n".join(lines)


def write_agent_report(report: AgentEvalReport, path: str,
                       extra_sections: Optional[Dict[str, str]] = None) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    parts = [
        f"# Agent 评测报告 · {report.config_name}",
        "",
        f"- 运行时间：{report.timestamp}",
        f"- 任务数：{report.n}",
        "- 指标口径：success_rate 要求 finished + 工具子序列匹配 + 关键词命中；"
        "avg_tokens 为**输入侧估算**（按工具结果与 query/answer 文本量），"
        "真机应改用 LLMResult.prompt_tokens 累加",
        "",
        "## 六类指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]
    for key in AGENT_METRIC_KEYS:
        parts.append(f"| {key} | {report.metrics.get(key, 0)} |")
    parts += ["", "## 分任务类型", "", render_agent_type_table(report),
              "", "## 失败任务（Badcase）", "", render_agent_failures(report)]
    if extra_sections:
        for title, body in extra_sections.items():
            parts += ["", f"## {title}", "", body]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


# --------------------------------------------------------------------- 消融配置
@dataclass
class AgentAblationConfig:
    """Agent 消融的一档配置。

    ★ 四档的差异必须落在**能力开关**上，而不是只改名字：
        baseline     —— 空白名单（模型拿不到工具，只能直答）
        +tools       —— 给工具但不做记忆/压缩
        +memory      —— 加三层记忆
        +compress    —— 再加三级上下文压缩（= Full Harness）
    """

    name: str
    use_tools: bool = True
    use_memory: bool = False
    use_compress: bool = False
    max_rounds: int = 6

    def label(self) -> str:
        bits = []
        if self.use_tools:
            bits.append("工具")
        if self.use_memory:
            bits.append("记忆")
        if self.use_compress:
            bits.append("压缩")
        return "+".join(bits) if bits else "直答"


def agent_ablation_configs() -> List[AgentAblationConfig]:
    """Plan 里写的那五档（这里给 4 档，Full 与 +compress 合并）。"""
    return [
        AgentAblationConfig(name="A_baseline_no_tools", use_tools=False,
                            use_memory=False, use_compress=False),
        AgentAblationConfig(name="B_plus_tools", use_tools=True,
                            use_memory=False, use_compress=False),
        AgentAblationConfig(name="C_plus_memory", use_tools=True,
                            use_memory=True, use_compress=False),
        AgentAblationConfig(name="D_full_harness", use_tools=True,
                            use_memory=True, use_compress=True),
    ]
