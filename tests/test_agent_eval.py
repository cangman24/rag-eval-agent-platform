# -*- coding: utf-8 -*-
"""tests/test_agent_eval.py —— Agent 层评测单测。全部离线。

覆盖：
  ① 任务成功判定（工具子序列 / 关键词 / 禁止词 / 未收尾 / 异常）
  ② 六类指标聚合（成功率 / 轮次 / 工具调用 / 工具失败 / Token / P95）
  ③ 分任务类型统计
  ④ 报告渲染与落盘
  ⑤ 四档消融配置的差异确实落在能力开关上
"""
import json
import os
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import AgentEngine, AgentState  # noqa: E402
from core.agent_eval import (  # noqa: E402
    AGENT_METRIC_KEYS, AgentEvalRow,
    AgentTask, agent_ablation_configs, aggregate_agent_metrics,
    count_tool_errors, estimate_run_tokens, extract_ok_tool_sequence,
    extract_tool_sequence, judge_task_success, load_agent_tasks,
    render_agent_ablation, render_agent_failures, render_agent_type_table,
    run_agent_eval, save_agent_tasks, validate_agent_tasks, write_agent_report,
)
from core.mcp import ToolRegistry  # noqa: E402

_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


def make_tmp() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    p = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(p, exist_ok=True)
    return p


# ================================================================== 假 run
class FakeRun:
    """最小可用的 run 替身（只需 .state / .events / .error）。"""

    def __init__(self, finished=True, answer="", rounds=2, tool_calls=1,
                 stop_reason="", task_type="spec_inquiry", error=None,
                 scratchpad=None, events=None):
        self.state = AgentState(
            query="q", task_type=task_type, rounds=rounds,
            tool_calls=tool_calls, finished=finished, answer=answer,
            stop_reason=stop_reason, verified=finished,
        )
        self.state.scratchpad = scratchpad if scratchpad is not None else []
        self.events = events or []
        self.error = error


def tool_item(tool, ok=True, result=None, blocked=False, error=None):
    item = {"type": "tool_call", "tool": tool, "ok": ok}
    if result is not None:
        item["result"] = result
    if blocked:
        item["blocked"] = True
    if error:
        item["error"] = error
    return item


# ================================================================== 成功判定
class TestJudgeTaskSuccess(unittest.TestCase):
    def test_simple_success(self):
        task = AgentTask("t1", "q", must_contain=["0.90"])
        ok, reason = judge_task_success(task, {"finished": True, "answer": "精度 0.90"}, [])
        self.assertTrue(ok, reason)

    def test_not_finished_fails(self):
        task = AgentTask("t1", "q", must_contain=["0.90"])
        ok, reason = judge_task_success(task, {"finished": False, "answer": "0.90",
                                               "stop_reason": "max_rounds"}, [])
        self.assertFalse(ok)
        self.assertIn("未收尾", reason)

    def test_error_fails(self):
        task = AgentTask("t1", "q", must_contain=["0.90"])
        ok, reason = judge_task_success(
            task, {"finished": True, "answer": "0.90", "error": "boom"}, [])
        self.assertFalse(ok)
        self.assertIn("boom", reason)

    def test_missing_keyword_fails(self):
        task = AgentTask("t1", "q", must_contain=["0.90", "0.88"])
        ok, reason = judge_task_success(task, {"finished": True, "answer": "0.90"}, [])
        self.assertFalse(ok)
        self.assertIn("0.88", reason)

    def test_forbidden_keyword_fails(self):
        task = AgentTask("t1", "q", must_not_contain=["我无法"])
        ok, reason = judge_task_success(
            task, {"finished": True, "answer": "我无法回答"}, [])
        self.assertFalse(ok)
        self.assertIn("禁止内容", reason)

    def test_tool_subsequence_match(self):
        task = AgentTask("t1", "q", expect_tools=["kb_search", "ticket_query"])
        ok, _ = judge_task_success(task, {"finished": True, "answer": "ok"},
                                   ["kb_search", "ticket_query"])
        self.assertTrue(ok)

    def test_tool_subsequence_allows_extra_in_between(self):
        """★ 子序列语义：允许中间插入别的工具调用。"""
        task = AgentTask("t1", "q", expect_tools=["kb_search", "ticket_query"])
        ok, _ = judge_task_success(task, {"finished": True, "answer": "ok"},
                                   ["kb_search", "metric_query", "ticket_query"])
        self.assertTrue(ok)

    def test_tool_order_matters(self):
        task = AgentTask("t1", "q", expect_tools=["kb_search", "ticket_query"])
        ok, reason = judge_task_success(task, {"finished": True, "answer": "ok"},
                                        ["ticket_query", "kb_search"])
        self.assertFalse(ok)
        self.assertIn("工具序列不符", reason)

    def test_missing_tool_fails(self):
        task = AgentTask("t1", "q", expect_tools=["kb_search", "ticket_query"])
        ok, _ = judge_task_success(task, {"finished": True, "answer": "ok"},
                                   ["kb_search"])
        self.assertFalse(ok)


# ================================================================== 采集
class TestCollectors(unittest.TestCase):
    def test_extract_tool_sequence_includes_blocked(self):
        run = FakeRun(scratchpad=[tool_item("kb_search"),
                                  tool_item("send_mail", blocked=True),
                                  {"type": "verify_feedback"}])
        self.assertEqual(extract_tool_sequence(run), ["kb_search", "send_mail"])

    def test_extract_empty(self):
        self.assertEqual(extract_tool_sequence(FakeRun()), [])

    def test_ok_sequence_excludes_blocked_and_failed(self):
        """★ 成功判定用的序列必须只含真正执行的工具。"""
        run = FakeRun(scratchpad=[
            tool_item("a"),
            tool_item("b", ok=False, error="超时"),
            tool_item("c", blocked=True),
        ])
        self.assertEqual(extract_tool_sequence(run), ["a", "b", "c"])
        self.assertEqual(extract_ok_tool_sequence(run), ["a"])

    def test_count_errors_and_blocked(self):
        run = FakeRun(scratchpad=[
            tool_item("a"), tool_item("b", ok=False, error="超时"),
            tool_item("c", blocked=True), tool_item("d", ok=False, error="x"),
        ])
        errors, blocked = count_tool_errors(run)
        self.assertEqual(errors, 2)
        self.assertEqual(blocked, 1)

    def test_estimate_tokens_uses_events_when_present(self):
        """★ 若事件里带了 prompt_tokens，应优先用它（真机口径）。"""
        run = FakeRun(events=[{"type": "llm", "data": {"prompt_tokens": 500}},
                              {"type": "llm", "data": {"prompt_tokens": 300}}])
        self.assertEqual(estimate_run_tokens(run), 800)

    def test_estimate_tokens_falls_back_to_text(self):
        run = FakeRun(scratchpad=[tool_item("kb", result={"hits": ["x" * 100]})],
                      answer="答案内容")
        self.assertGreater(estimate_run_tokens(run), 0)


# ================================================================== 聚合
class TestAggregateMetrics(unittest.TestCase):
    def _rows(self):
        return [
            AgentEvalRow("t1", "spec", "q1", True, "", 2, 1, 0, 0, 100, 50.0, "spec"),
            AgentEvalRow("t2", "spec", "q2", False, "缺关键词", 4, 2, 1, 0, 300, 150.0, "spec"),
            AgentEvalRow("t3", "record", "q3", True, "", 3, 2, 0, 1, 200, 100.0, "record"),
        ]

    def test_all_six_metrics_present(self):
        m = aggregate_agent_metrics(self._rows(), [50.0, 150.0, 100.0])
        for key in AGENT_METRIC_KEYS:
            self.assertIn(key, m)

    def test_success_rate(self):
        m = aggregate_agent_metrics(self._rows(), [50, 150, 100])
        # 指标做了 4 位小数舍入（报告要可读），断言用 delta
        self.assertAlmostEqual(m["success_rate"], 2 / 3, delta=1e-3)

    def test_averages(self):
        m = aggregate_agent_metrics(self._rows(), [50, 150, 100])
        self.assertAlmostEqual(m["avg_rounds"], 3.0, delta=1e-2)
        self.assertAlmostEqual(m["avg_tool_calls"], 5 / 3, delta=1e-2)
        self.assertAlmostEqual(m["avg_tool_errors"], 1 / 3, delta=1e-2)
        self.assertAlmostEqual(m["avg_tokens"], 200.0, delta=1e-1)

    def test_p95_present(self):
        m = aggregate_agent_metrics(self._rows(), [50, 150, 100])
        self.assertGreaterEqual(m["p95_latency_ms"], 0)

    def test_empty_rows(self):
        m = aggregate_agent_metrics([], [])
        for key in AGENT_METRIC_KEYS:
            self.assertEqual(m[key], 0.0)


# ================================================================== run_eval
class TestRunAgentEval(unittest.TestCase):
    def _tasks(self):
        return [
            AgentTask("t1", "查精度", expect_tools=["kb_search"],
                      must_contain=["0.90"], task_type="spec_inquiry"),
            AgentTask("t2", "查工单", expect_tools=["kb_search", "ticket_query"],
                      must_contain=["关闭"], task_type="record_lookup"),
            AgentTask("t3", "随便问问", must_contain=["不知道"],
                      task_type="general"),
        ]

    def test_happy_path(self):
        answers = {"查精度": ("0.90 是要求", ["kb_search"]),
                   "查工单": ("工单已关闭", ["kb_search", "ticket_query"]),
                   "随便问问": ("不知道", ["kb_search"])}

        def run_fn(query):
            answer, seq = answers[query]
            return FakeRun(finished=True, answer=answer, rounds=len(seq),
                           tool_calls=len(seq),
                           scratchpad=[tool_item(t) for t in seq])

        rep = run_agent_eval(self._tasks(), run_fn, config_name="X")
        self.assertEqual(rep.n, 3)
        self.assertEqual(rep.metrics["success_rate"], 1.0)
        self.assertIn("spec_inquiry", rep.by_type)

    def test_partial_failure_recorded(self):
        def run_fn(query):
            if query == "查工单":
                return FakeRun(finished=True, answer="没有相关信息",
                               scratchpad=[tool_item("kb_search")])
            return FakeRun(finished=True, answer="0.90 不知道",
                           scratchpad=[tool_item("kb_search")])

        rep = run_agent_eval(self._tasks(), run_fn)
        self.assertLess(rep.metrics["success_rate"], 1.0)
        self.assertTrue(any(not r.success for r in rep.rows))

    def test_exception_isolated(self):
        """★ 单条任务抛异常不能中断整批。"""
        def run_fn(query):
            if query == "查工单":
                raise RuntimeError("引擎炸了")
            return FakeRun(finished=True, answer="0.90 不知道",
                           scratchpad=[tool_item("kb_search")])

        rep = run_agent_eval(self._tasks(), run_fn)
        self.assertEqual(rep.n, 3)
        failed = [r for r in rep.rows if "抛异常" in r.reason]
        self.assertEqual(len(failed), 1)

    def test_tool_errors_counted(self):
        def run_fn(query):
            return FakeRun(finished=True, answer="0.90 关闭 不知道",
                           scratchpad=[tool_item("kb_search"),
                                       tool_item("ticket_query", ok=False,
                                                 error="超时")])

        rep = run_agent_eval(self._tasks(), run_fn)
        self.assertEqual(rep.rows[0].tool_errors, 1)
        self.assertGreater(rep.metrics["avg_tool_errors"], 0)

    def test_custom_results_to_dict(self):
        def run_fn(query):
            return {"finished": True, "answer": "0.90 关闭 不知道",
                    "rounds": 1, "tool_calls": 0, "task_type": "x"}

        rep = run_agent_eval(self._tasks(), run_fn,
                             results_to_dict=lambda r: r)
        self.assertEqual(rep.n, 3)

    def test_progress_callback(self):
        msgs = []
        run_agent_eval(self._tasks(),
                       lambda q: FakeRun(finished=True, answer="0.90 关闭 不知道"),
                       progress=msgs.append)
        self.assertEqual(len(msgs), 3)
        self.assertTrue(any("OK" in m or "FAIL" in m for m in msgs))

    def test_empty_tasks(self):
        rep = run_agent_eval([], lambda q: FakeRun())
        self.assertEqual(rep.n, 0)
        self.assertEqual(rep.metrics["success_rate"], 0.0)


# ================================================================== 报告
class TestRenderers(unittest.TestCase):
    def _reports(self):
        tasks = [AgentTask("t1", "q1", must_contain=["a"], task_type="spec"),
                 AgentTask("t2", "q2", must_contain=["b"], task_type="record")]
        out = []
        for name, answer in (("A_baseline", "a b"), ("B_full", "a b")):
            out.append(run_agent_eval(
                tasks,
                lambda q, ans=answer: FakeRun(finished=True, answer=ans,
                                              rounds=2, tool_calls=1),
                config_name=name))
        return out

    def test_ablation_table_has_all_metrics(self):
        table = render_agent_ablation(self._reports())
        for header in ("成功率", "平均轮次", "平均工具调用", "平均工具失败",
                       "平均Token", "P95延迟"):
            self.assertIn(header, table)
        self.assertEqual(len(table.strip().splitlines()), 2 + 2)

    def test_type_table(self):
        table = render_agent_type_table(self._reports()[0])
        self.assertIn("spec", table)
        self.assertIn("record", table)

    def test_failure_list(self):
        tasks = [AgentTask("t1", "q1", must_contain=["不存在的东西"])]
        rep = run_agent_eval(tasks, lambda q: FakeRun(finished=True, answer="别的"))
        text = render_agent_failures(rep)
        self.assertIn("t1", text)
        self.assertIn("缺少关键内容", text)

    def test_failure_list_when_none(self):
        tasks = [AgentTask("t1", "q1", must_contain=["a"])]
        rep = run_agent_eval(tasks, lambda q: FakeRun(finished=True, answer="a"))
        self.assertIn("无失败任务", render_agent_failures(rep))

    def test_write_report(self):
        tmp = make_tmp()
        try:
            tasks = [AgentTask("t1", "q1", must_contain=["a"])]
            rep = run_agent_eval(tasks, lambda q: FakeRun(finished=True, answer="a"),
                                 config_name="cfg")
            path = os.path.join(tmp, "sub", "agent.md")
            write_agent_report(rep, path, extra_sections={"消融对比": "| a |"})
            self.assertTrue(os.path.exists(path))
            content = open(path, encoding="utf-8").read()
            self.assertIn("Agent 评测报告", content)
            self.assertIn("success_rate", content)
            self.assertIn("消融对比", content)
            self.assertIn("输入侧估算", content)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ================================================================== 任务集 IO
class TestAgentTaskIO(unittest.TestCase):
    def test_roundtrip(self):
        tmp = make_tmp()
        try:
            path = os.path.join(tmp, "tasks.jsonl")
            tasks = [AgentTask("t1", "q1", ["kb_search"], ["0.90"], ["失败"], "spec", 4),
                     AgentTask("t2", "q2", [], ["a"], [], "general", 2)]
            save_agent_tasks(tasks, path)
            loaded = load_agent_tasks(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].expect_tools, ["kb_search"])
            self.assertEqual(loaded[0].max_rounds, 4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_json_array(self):
        tmp = make_tmp()
        try:
            path = os.path.join(tmp, "t.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"tid": "a", "query": "q", "must_contain": ["x"]}], f)
            self.assertEqual(len(load_agent_tasks(path)), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_agent_tasks(os.path.join(_TMP_ROOT, "nope.jsonl"))

    def test_validate_clean(self):
        rep = validate_agent_tasks([AgentTask("t1", "q", ["kb"], ["a"], [], "spec", 3)])
        self.assertTrue(rep["ok"])

    def test_validate_detects_unjugdeable(self):
        rep = validate_agent_tasks([AgentTask("t1", "q", [], [], [], "spec", 3)])
        self.assertFalse(rep["ok"])
        self.assertTrue(any("无法判定成功" in p for p in rep["problems"]))

    def test_validate_detects_dup_and_empty(self):
        rep = validate_agent_tasks([AgentTask("t1", "", ["kb"], [], [], "spec", 3),
                                    AgentTask("t1", "q", ["kb"], [], [], "spec", 3)])
        self.assertTrue(any("query 为空" in p for p in rep["problems"]))
        self.assertTrue(any("tid 重复" in p for p in rep["problems"]))

    def test_validate_bad_max_rounds(self):
        rep = validate_agent_tasks([AgentTask("t1", "q", ["kb"], [], [], "spec", 0)])
        self.assertTrue(any("max_rounds" in p for p in rep["problems"]))


# ================================================================== 消融配置
class TestAblationConfigs(unittest.TestCase):
    def test_four_configs(self):
        cfgs = agent_ablation_configs()
        self.assertEqual(len(cfgs), 4)
        self.assertEqual([c.name for c in cfgs],
                         ["A_baseline_no_tools", "B_plus_tools",
                          "C_plus_memory", "D_full_harness"])

    def test_baseline_has_no_tools(self):
        self.assertFalse(agent_ablation_configs()[0].use_tools)

    def test_full_has_everything(self):
        last = agent_ablation_configs()[-1]
        self.assertTrue(last.use_tools and last.use_memory and last.use_compress)

    def test_labels(self):
        self.assertEqual(agent_ablation_configs()[0].label(), "直答")
        self.assertIn("工具", agent_ablation_configs()[1].label())
        self.assertIn("压缩", agent_ablation_configs()[3].label())


# ================================================================== 真引擎集成
class TestAgentEvalWithRealEngine(unittest.TestCase):
    """用真 AgentEngine 跑评测 —— 验证指标采集确实来自引擎。"""

    def _registry(self):
        reg = ToolRegistry()

        @reg.tool(description="检索")
        def kb_search(query: str):
            """检索

            query: 检索词
            """
            return {"count": 1, "hits": [{"id": "c1", "text": "精度 0.90"}]}

        @reg.tool(description="查工单")
        def ticket_query(ticket_id: str):
            """查工单

            ticket_id: 单号
            """
            return {"ticket_id": ticket_id, "status": "closed"}

        return reg

    def _engine(self, decisions, route_table):
        reg = self._registry()
        cursor = {"i": 0}

        def act(state, allowed):
            d = decisions[min(cursor["i"], len(decisions) - 1)]
            cursor["i"] += 1
            return d

        return AgentEngine(registry=reg, intent_fn=lambda q: "spec",
                           plan_fn=lambda q, t: ["检索"], act_fn=act,
                           route_table=route_table, max_rounds=5)

    def test_metrics_come_from_engine(self):
        engine = self._engine(
            [{"tool": "kb_search", "args": {"query": "精度"}},
             {"final": "精度要求 0.90"}],
            {"spec": ["kb_search"]})
        tasks = [AgentTask("t1", "查精度", ["kb_search"], ["0.90"], [], "spec")]
        rep = run_agent_eval(tasks, lambda q: engine.run(q))
        self.assertEqual(rep.metrics["success_rate"], 1.0)
        self.assertEqual(rep.rows[0].tool_calls, 1)
        self.assertEqual(rep.rows[0].rounds, 2)
        self.assertEqual(rep.rows[0].tool_sequence, ["kb_search"])

    def test_baseline_with_empty_route_table_blocks_tools(self):
        """★ 消融关键验证：空白名单下模型想调工具会被引擎硬拦，
        而不是靠提示词「请求它别调」。"""
        engine = self._engine(
            [{"tool": "kb_search", "args": {"query": "精度"}},
             {"final": "我无法检索，只能直答"}],
            {"spec": []})            # 空白名单
        run = engine.run("查精度")
        self.assertEqual(run.state.tool_calls, 0, "被拦的调用不该计入预算")
        blocked = [s for s in run.state.scratchpad if s.get("blocked")]
        self.assertEqual(len(blocked), 1)
        # 工具不在白名单里，序列仍记录这次尝试（便于诊断"模型想干什么"）
        self.assertEqual(extract_tool_sequence(run), ["kb_search"])

    def test_tool_expectation_fails_when_tool_blocked(self):
        """若任务要求必须用某工具，而它被拦下 -> 任务应判失败。"""
        engine = self._engine(
            [{"tool": "kb_search", "args": {"query": "精度"}},
             {"final": "我无法检索"}],
            {"spec": []})
        tasks = [AgentTask("t1", "查精度", ["kb_search"], [], [], "spec")]
        rep = run_agent_eval(tasks, lambda q: engine.run(q))
        self.assertFalse(rep.rows[0].success)
        self.assertEqual(rep.rows[0].blocked_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
