# -*- coding: utf-8 -*-
"""tests/test_agent.py —— MCP 工具层 + Agent 主循环单测。全部离线。

重点验证四个工程机制（对应简历里的四条）：
  1. 工具白名单拦截越权调用
  2. Checkpoint 续跑（已完成节点不重跑）
  3. 超时熔断（最大轮次 / 工具次数 / 总耗时 / 死循环）
  4. Schema 自动生成与转换
"""
import json
import os
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import (  # noqa: E402
    AgentEngine, AgentState, EventType, InMemoryCheckpointer,
    JsonFileCheckpointer, json_safe, make_event,
)
from core.mcp import (  # noqa: E402
    InvalidArguments, ToolError, ToolNotAllowed, ToolNotFound, ToolRegistry,
    ToolSpec, build_input_schema, json_type_ok, validate_against_schema,
)

_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


def make_tmp() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    p = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(p, exist_ok=True)
    return p


# ================================================================== schema
class TestBuildInputSchema(unittest.TestCase):
    def test_required_and_optional(self):
        def f(query: str, top_k: int = 5):
            """检索知识库

            query: 检索词
            top_k: 返回条数
            """
        schema = build_input_schema(f)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["query"])
        self.assertEqual(schema["properties"]["query"]["type"], "string")
        self.assertEqual(schema["properties"]["query"]["description"], "检索词")
        self.assertEqual(schema["properties"]["top_k"]["type"], "integer")
        self.assertEqual(schema["properties"]["top_k"]["default"], 5)

    def test_list_annotation(self):
        def f(ids: list):
            pass
        schema = build_input_schema(f)
        self.assertEqual(schema["properties"]["ids"]["type"], "array")

    def test_dict_annotation(self):
        def f(meta: dict):
            pass
        self.assertEqual(build_input_schema(f)["properties"]["meta"]["type"], "object")

    def test_unannotated_becomes_string(self):
        def f(x):
            pass
        self.assertEqual(build_input_schema(f)["properties"]["x"]["type"], "string")

    def test_skips_self_and_kwargs(self):
        class C:
            def m(self, a: str, **kw):
                pass
        schema = build_input_schema(C.m)
        self.assertEqual(list(schema["properties"]), ["a"])


class TestJsonTypeOk(unittest.TestCase):
    def test_bool_is_not_integer(self):
        # ★ 顺序坑：Python 里 isinstance(True, int) == True
        self.assertTrue(json_type_ok(True, "boolean"))
        self.assertFalse(json_type_ok(True, "integer"))

    def test_integer_not_boolean(self):
        self.assertTrue(json_type_ok(3, "integer"))
        self.assertFalse(json_type_ok(3, "boolean"))

    def test_number_accepts_int(self):
        self.assertTrue(json_type_ok(3, "number"))
        self.assertTrue(json_type_ok(3.5, "number"))

    def test_string(self):
        self.assertTrue(json_type_ok("x", "string"))
        self.assertFalse(json_type_ok(1, "string"))


class TestValidateSchema(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"},
                       "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["query"],
    }

    def test_valid(self):
        self.assertEqual(validate_against_schema({"query": "x", "top_k": 3}, self.SCHEMA), [])

    def test_missing_required(self):
        errs = validate_against_schema({"top_k": 3}, self.SCHEMA)
        self.assertTrue(any("缺少必填参数" in e for e in errs))

    def test_wrong_type(self):
        errs = validate_against_schema({"query": 123}, self.SCHEMA)
        self.assertTrue(any("类型应为 string" in e for e in errs))

    def test_array_item_type(self):
        errs = validate_against_schema({"query": "x", "tags": ["a", 1]}, self.SCHEMA)
        self.assertTrue(any("tags[1]" in e for e in errs))

    def test_unknown_param_not_rejected(self):
        self.assertEqual(validate_against_schema({"query": "x", "junk": 1}, self.SCHEMA), [])

    def test_none_skipped(self):
        self.assertEqual(validate_against_schema({"query": "x", "top_k": None}, self.SCHEMA), [])


# ================================================================== registry
class TestToolSpec(unittest.TestCase):
    def _spec(self, func):
        return ToolSpec(name="t", description="d", input_schema=build_input_schema(func),
                        func=func)

    def test_call_strips_unknown_kwargs(self):
        seen = {}

        def f(query: str):
            seen.update({"query": query})
            return "ok"
        spec = self._spec(f)
        spec.call(query="x", junk=1)
        self.assertEqual(seen["query"], "x")

    def test_invalid_arguments_raises(self):
        def f(query: str):
            return "ok"
        with self.assertRaises(InvalidArguments):
            self._spec(f).call()

    def test_internal_exception_wrapped_as_tool_error(self):
        def f(query: str):
            raise RuntimeError("内部炸了")
        with self.assertRaises(ToolError):
            self._spec(f).call(query="x")

    def test_to_openai_tool_format(self):
        def f(query: str):
            """检索

            query: 检索词
            """
        d = self._spec(f).to_openai_tool()
        self.assertEqual(d["type"], "function")
        self.assertEqual(d["function"]["name"], "t")
        self.assertIn("parameters", d["function"])

    def test_to_mcp_dict_format(self):
        def f(query: str):
            pass
        d = self._spec(f).to_mcp_dict()
        self.assertIn("inputSchema", d)
        self.assertEqual(d["name"], "t")


class TestToolRegistry(unittest.TestCase):
    def _reg(self):
        reg = ToolRegistry()

        @reg.tool(description="检索知识库")
        def kb_search(query: str, top_k: int = 5):
            """检索知识库

            query: 检索词
            top_k: 返回条数
            """
            return {"hits": [f"{query}-{i}" for i in range(top_k)]}

        @reg.tool(description="查询工单", sensitive=True)
        def ticket_query(ticket_id: str):
            """查询工单

            ticket_id: 工单号
            """
            if ticket_id == "bad":
                raise ToolError("工单不存在")
            return {"ticket_id": ticket_id, "status": "open"}

        return reg

    def test_decorator_registers(self):
        reg = self._reg()
        self.assertEqual(reg.names(), ["kb_search", "ticket_query"])
        self.assertEqual(len(reg), 2)

    def test_sensitive_flag(self):
        self.assertTrue(self._reg().get("ticket_query").sensitive)

    def test_duplicate_register_raises(self):
        reg = self._reg()
        with self.assertRaises(ValueError):
            reg.register(reg.get("kb_search"))

    def test_replace_allowed(self):
        reg = self._reg()
        reg.register(reg.get("kb_search"), replace=True)
        self.assertEqual(len(reg), 2)

    def test_get_unknown_raises(self):
        with self.assertRaises(ToolNotFound):
            self._reg().get("nope")

    def test_has(self):
        reg = self._reg()
        self.assertTrue(reg.has("kb_search"))
        self.assertFalse(reg.has("nope"))

    def test_unregister(self):
        reg = self._reg()
        reg.unregister("kb_search")
        self.assertFalse(reg.has("kb_search"))

    def test_call_success(self):
        out = self._reg().call("kb_search", {"query": "水稻", "top_k": 2})
        self.assertEqual(out["hits"], ["水稻-0", "水稻-1"])

    def test_whitelist_blocks_unauthorized(self):
        """★ 工具白名单的核心验证。"""
        with self.assertRaises(ToolNotAllowed):
            self._reg().call("ticket_query", {"ticket_id": "T1"},
                             allowed=["kb_search"])

    def test_whitelist_allows_listed(self):
        out = self._reg().call("ticket_query", {"ticket_id": "T1"},
                               allowed=["kb_search", "ticket_query"])
        self.assertEqual(out["status"], "open")

    def test_none_whitelist_means_all(self):
        out = self._reg().call("ticket_query", {"ticket_id": "T1"}, allowed=None)
        self.assertEqual(out["status"], "open")

    def test_tool_internal_error_propagates(self):
        with self.assertRaises(ToolError):
            self._reg().call("ticket_query", {"ticket_id": "bad"})

    def test_resolve_allowed_ignores_unknown_names(self):
        """白名单里的拼写错误不该让整轮崩掉。"""
        specs = self._reg().resolve_allowed(["kb_search", "typo_name"])
        self.assertEqual([s.name for s in specs], ["kb_search"])

    def test_openai_tools_filtered(self):
        reg = self._reg()
        names = [t["function"]["name"] for t in reg.openai_tools(allowed=["kb_search"])]
        self.assertEqual(names, ["kb_search"])

    def test_mcp_tools_all(self):
        self.assertEqual(len(self._reg().mcp_tools()), 2)


# ================================================================== json_safe
class TestJsonSafe(unittest.TestCase):
    def test_primitives(self):
        for v in (None, "x", 1, 1.5, True):
            self.assertEqual(json_safe(v), v)

    def test_nested(self):
        self.assertEqual(json_safe({"a": [1, {"b": 2}]}), {"a": [1, {"b": 2}]})

    def test_set_becomes_list(self):
        self.assertEqual(sorted(json_safe({1, 2})), [1, 2])

    def test_tuple_becomes_list(self):
        self.assertEqual(json_safe((1, 2)), [1, 2])

    def test_to_dict_object(self):
        class O:
            def to_dict(self):
                return {"k": "v"}
        self.assertEqual(json_safe(O()), {"k": "v"})

    def test_arbitrary_object_falls_back_to_repr(self):
        class O:
            def __repr__(self):
                return "<O>"
        self.assertEqual(json_safe(O()), "<O>")

    def test_result_is_json_serializable(self):
        class O:
            pass
        payload = {"a": [1, O(), {2, 3}], "b": (4, 5)}
        json.dumps(json_safe(payload), ensure_ascii=False)  # 不抛异常即通过

    def test_depth_guard(self):
        deep = cur = {}
        for _ in range(20):
            cur["n"] = {}
            cur = cur["n"]
        json.dumps(json_safe(deep))  # 不应栈溢出


# ================================================================== state
class TestAgentState(unittest.TestCase):
    def test_roundtrip_json(self):
        s = AgentState(query="q", thread_id="t1", task_type="kb",
                       allowed_tools=["a"], plan=["p"], rounds=2)
        s2 = AgentState.from_dict(json.loads(s.to_json()))
        self.assertEqual(s2.query, "q")
        self.assertEqual(s2.rounds, 2)
        self.assertEqual(s2.allowed_tools, ["a"])

    def test_from_dict_ignores_unknown_keys(self):
        s = AgentState.from_dict({"query": "q", "bogus": 1})
        self.assertEqual(s.query, "q")
        self.assertFalse(hasattr(s, "bogus"))

    def test_from_dict_empty(self):
        self.assertEqual(AgentState.from_dict({}).query, "")

    def test_signature(self):
        s = AgentState(task_type="kb")
        s.scratchpad = [{"tool": "kb_search"}, {"tool": "kb_search"}]
        self.assertEqual(s.signature(), "kb|kb_search,kb_search")


# ================================================================== checkpointer
class TestInMemoryCheckpointer(unittest.TestCase):
    def test_save_load(self):
        cp = InMemoryCheckpointer()
        cp.save("t1", AgentState(query="q", rounds=3))
        self.assertEqual(cp.load("t1").rounds, 3)

    def test_load_missing(self):
        self.assertIsNone(InMemoryCheckpointer().load("nope"))

    def test_delete(self):
        cp = InMemoryCheckpointer()
        cp.save("t1", AgentState())
        cp.delete("t1")
        self.assertIsNone(cp.load("t1"))

    def test_list_threads(self):
        cp = InMemoryCheckpointer()
        cp.save("b", AgentState())
        cp.save("a", AgentState())
        self.assertEqual(cp.list_threads(), ["a", "b"])


class TestJsonFileCheckpointer(unittest.TestCase):
    def test_roundtrip(self):
        tmp = make_tmp()
        try:
            cp = JsonFileCheckpointer(tmp)
            cp.save("t1", AgentState(query="q", rounds=2, answer="a"))
            loaded = cp.load("t1")
            self.assertEqual(loaded.rounds, 2)
            self.assertEqual(loaded.answer, "a")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_returns_none(self):
        tmp = make_tmp()
        try:
            self.assertIsNone(JsonFileCheckpointer(tmp).load("nope"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_corrupt_file_returns_none(self):
        tmp = make_tmp()
        try:
            cp = JsonFileCheckpointer(tmp)
            cp.save("t1", AgentState())
            with open(os.path.join(tmp, "t1.json"), "w", encoding="utf-8") as f:
                f.write("{broken")
            self.assertIsNone(cp.load("t1"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unsafe_thread_id_is_sanitized(self):
        tmp = make_tmp()
        try:
            cp = JsonFileCheckpointer(tmp)
            cp.save("../../evil id", AgentState())
            # 不应逃出目录
            self.assertEqual(len(os.listdir(tmp)), 1)
            self.assertNotIn("..", os.listdir(tmp)[0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_delete_and_list(self):
        tmp = make_tmp()
        try:
            cp = JsonFileCheckpointer(tmp)
            cp.save("a", AgentState())
            cp.save("b", AgentState())
            self.assertEqual(cp.list_threads(), ["a", "b"])
            cp.delete("a")
            self.assertEqual(cp.list_threads(), ["b"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ================================================================== engine
def build_engine(**kw):
    """构造一个可离线跑的 Agent。act_fn 用脚本化的决策队列模拟"模型"。"""
    reg = ToolRegistry()

    @reg.tool(description="检索知识库")
    def kb_search(query: str, top_k: int = 5):
        """检索知识库

        query: 检索词
        top_k: 返回条数
        """
        return {"hits": [{"id": "c1", "text": f"{query} 的结果", "score": 0.9}],
                "count": 1}

    @reg.tool(description="查询工单")
    def ticket_query(ticket_id: str):
        """查询工单

        ticket_id: 工单号
        """
        return {"ticket_id": ticket_id, "status": "open"}

    @reg.tool(description="发送邮件（有副作用）", sensitive=True)
    def send_mail(to: str):
        """发送邮件

        to: 收件人
        """
        return {"sent": True, "to": to}

    decisions = kw.pop("decisions", [])

    def scripted_act(state, allowed):
        idx = min(state.rounds - 1, len(decisions) - 1)
        return decisions[idx] if decisions else {"final": "默认回答"}

    act_fn = kw.pop("act_fn", scripted_act)

    route_table = kw.pop("route_table", {
        "kb": ["kb_search"],
        "ticket": ["ticket_query"],
        "mail": ["send_mail"],
        "general": [],
    })

    engine = AgentEngine(
        registry=reg,
        intent_fn=kw.pop("intent_fn", lambda q: "kb" if "制度" in q else "ticket"),
        plan_fn=kw.pop("plan_fn", lambda q, t: ["检索", "汇总"]),
        act_fn=act_fn,
        verify_fn=kw.pop("verify_fn", None),
        route_table=route_table,
        **kw,
    )
    return engine, reg


class TestAgentEngineHappyPath(unittest.TestCase):
    def test_single_tool_then_answer(self):
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "水稻制度"}},
            {"final": "制度要求 mAP 0.90 [1]", "citations": ["c1"]},
        ])
        run = engine.run("查一下制度")
        self.assertTrue(run.ok, run.error)
        self.assertEqual(run.state.answer, "制度要求 mAP 0.90 [1]")
        self.assertEqual(run.state.citations, ["c1"])
        self.assertEqual(run.state.tool_calls, 1)
        self.assertTrue(run.state.verified)

    def test_events_emitted_in_order(self):
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "x"}},
            {"final": "答案"},
        ])
        run = engine.run("查制度")
        types = [e["type"] for e in run.events]
        for expected in (EventType.START, EventType.INTENT, EventType.PLAN,
                         EventType.TOOL_CALL, EventType.TOOL_RESULT,
                         EventType.VERIFY, EventType.ANSWER, EventType.DONE):
            self.assertIn(expected, types, f"缺少事件 {expected}")

    def test_intent_routing_assigns_whitelist(self):
        engine, _ = build_engine(decisions=[{"final": "ok"}])
        run = engine.run("查制度")
        self.assertEqual(run.state.task_type, "kb")
        self.assertEqual(run.state.allowed_tools, ["kb_search"])

    def test_tool_result_recorded_in_scratchpad(self):
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "x", "top_k": 1}},
            {"final": "答案"},
        ])
        run = engine.run("查制度")
        calls = [s for s in run.state.scratchpad if s.get("type") == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "kb_search")
        self.assertTrue(calls[0]["ok"])
        self.assertIn("result", calls[0])

    def test_finished_run_does_not_repeat_answer(self):
        """★ 回归测试：校验通过后必须立即结束，不能重复输出答案、虚增轮次。

        （demo_agent.py 曾暴露此问题：同一答案被打印 4 次、rounds 从 3 涨到 6。）
        """
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "x"}},
            {"final": "唯一答案"},
        ], max_rounds=10)
        run = engine.run("查制度")
        answers = [e for e in run.events if e["type"] == EventType.ANSWER]
        self.assertEqual(len(answers), 1, "答案只应输出一次")
        self.assertTrue(run.state.finished)
        # 2 轮：1 轮工具 + 1 轮作答
        self.assertEqual(run.state.rounds, 2)

    def test_default_plan_recorded(self):
        engine, _ = build_engine(decisions=[{"final": "ok"}])
        run = engine.run("查制度")
        self.assertEqual(run.state.plan, ["检索", "汇总"])


class TestAgentWhitelist(unittest.TestCase):
    def test_unauthorized_tool_blocked(self):
        """★ 核心：意图是 kb，模型却想发邮件 -> 必须被拦下。"""
        engine, _ = build_engine(decisions=[
            {"tool": "send_mail", "args": {"to": "a@b.com"}},
            {"final": "我没权限发邮件"},
        ])
        run = engine.run("查制度")   # -> task_type=kb -> 白名单只有 kb_search
        blocked = [e for e in run.events if e["type"] == EventType.TOOL_ERROR
                   and e["data"].get("blocked")]
        self.assertEqual(len(blocked), 1)
        self.assertIn("白名单", blocked[0]["data"]["error"])

    def test_blocked_call_does_not_count_toward_tool_budget(self):
        engine, _ = build_engine(decisions=[
            {"tool": "send_mail", "args": {"to": "a@b.com"}},
            {"final": "ok"},
        ])
        run = engine.run("查制度")
        self.assertEqual(run.state.tool_calls, 0)

    def test_unknown_intent_gets_empty_whitelist(self):
        """未在路由表里的意图 -> 空白名单（最保守策略）。"""
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "x"}},
            {"final": "兜底回答"},
        ], intent_fn=lambda q: "unknown_type")
        run = engine.run("随便问问")
        self.assertEqual(run.state.allowed_tools, [])
        blocked = [e for e in run.events
                   if e["type"] == EventType.TOOL_ERROR and e["data"].get("blocked")]
        self.assertEqual(len(blocked), 1)

    def test_route_method_direct(self):
        engine, _ = build_engine()
        self.assertEqual(engine.route("kb"), ["kb_search"])
        self.assertEqual(engine.route("nope"), [])


class TestAgentFuseGuards(unittest.TestCase):
    def test_max_rounds_stops_loop(self):
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "loop"}},
        ], max_rounds=3)
        run = engine.run("查制度")
        self.assertFalse(run.state.finished)
        self.assertEqual(run.state.rounds, 3)
        self.assertEqual(run.state.stop_reason, "max_rounds")

    def test_max_tool_calls_stops(self):
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "x"}},
        ], max_tool_calls=2, max_rounds=10)
        run = engine.run("查制度")
        self.assertEqual(run.state.tool_calls, 2)
        self.assertEqual(run.state.stop_reason, "max_tool_calls")

    def test_total_timeout_stops(self):
        """用可注入时钟模拟超时，避免真的 sleep。"""
        clock = {"t": 1000.0}

        def fake_clock():
            return clock["t"]

        def slow_act(state, allowed):
            clock["t"] += 40.0      # 每轮推进 40 秒：第 2 轮开始前累计 80s > 60s
            return {"tool": "kb_search", "args": {"query": "x"}}

        reg = ToolRegistry()

        @reg.tool(description="t")
        def kb_search(query: str):
            """检索

            query: q
            """
            return {"ok": True}

        engine = AgentEngine(
            registry=reg, intent_fn=lambda q: "kb",
            plan_fn=lambda q, t: [], act_fn=slow_act,
            route_table={"kb": ["kb_search"]},
            max_rounds=100, total_timeout_s=60.0, clock=fake_clock,
        )
        run = engine.run("查制度")
        self.assertEqual(run.state.stop_reason, "total_timeout")
        self.assertLess(run.state.rounds, 100)

    def test_verify_failure_loops_back(self):
        calls = {"n": 0}

        def verify(state, answer):
            calls["n"] += 1
            return (calls["n"] >= 2, "第一次不合格" if calls["n"] < 2 else "合格")

        engine, _ = build_engine(
            decisions=[{"final": "差答案"}, {"final": "好答案"}],
            verify_fn=verify, max_rounds=4,
        )
        run = engine.run("查制度")
        self.assertTrue(run.state.finished)
        self.assertEqual(run.state.answer, "好答案")
        # 校验反馈应进入 scratchpad
        self.assertTrue(any(s.get("type") == "verify_feedback"
                            for s in run.state.scratchpad))

    def test_detect_loop(self):
        engine, _ = build_engine(decisions=[{"tool": "kb_search", "args": {"query": "x"}}],
                                 max_rounds=8)
        run = engine.run("查制度")
        self.assertTrue(engine.detect_loop(run.state, window=2))

    def test_no_decision_recorded_as_error(self):
        engine, _ = build_engine(decisions=[{}], max_rounds=2)
        run = engine.run("查制度")
        self.assertEqual(run.state.stop_reason, "no_decision")
        self.assertTrue(any(e["type"] == EventType.ERROR for e in run.events))


class TestAgentCheckpointResume(unittest.TestCase):
    def test_checkpoint_saved_each_round(self):
        cp = InMemoryCheckpointer()
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "x"}},
            {"final": "答案"},
        ], checkpointer=cp)
        engine.run("查制度", thread_id="T1")
        self.assertGreater(cp.saves, 0)
        self.assertIsNotNone(cp.load("T1"))

    def test_resume_skips_completed(self):
        """★ 核心：已完成的任务续跑时不应重跑任何节点。"""
        cp = InMemoryCheckpointer()
        intent_calls = {"n": 0}

        def counting_intent(q):
            intent_calls["n"] += 1
            return "kb"

        engine, _ = build_engine(decisions=[{"final": "答案"}],
                                 checkpointer=cp, intent_fn=counting_intent)
        engine.run("查制度", thread_id="T2")
        self.assertEqual(intent_calls["n"], 1)

        # 续跑：应直接返回，不再调 intent_fn
        run2 = engine.resume("T2")
        self.assertEqual(intent_calls["n"], 1, "续跑不应重新执行意图识别")
        self.assertTrue(run2.state.finished)
        self.assertTrue(any(e["type"] == EventType.RESUME for e in run2.events))

    def test_resume_unfinished_continues_rounds(self):
        """中断在中间：续跑应从已有轮次继续，而不是从 0 开始。"""
        cp = InMemoryCheckpointer()
        decisions = [{"tool": "kb_search", "args": {"query": "x"}}] * 4

        # 第一段：只允许 1 轮就停
        engine1, _ = build_engine(decisions=[{"tool": "kb_search", "args": {"query": "x"}}],
                                  checkpointer=cp, max_rounds=1)
        run1 = engine1.run("查制度", thread_id="T3")
        self.assertEqual(run1.state.rounds, 1)
        self.assertEqual(run1.state.stop_reason, "max_rounds")

        # 第二段：同一 checkpointer，放宽轮次后继续
        engine2, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "y"}},
            {"final": "续跑后的答案"},
        ], checkpointer=cp, max_rounds=5)
        run2 = engine2.resume("T3")
        self.assertGreater(run2.state.rounds, 1, "应从检查点继续累加轮次")
        self.assertTrue(run2.state.finished)

    def test_resume_unknown_thread_starts_fresh(self):
        cp = InMemoryCheckpointer()
        engine, _ = build_engine(decisions=[{"final": "新答案"}], checkpointer=cp)
        run = engine.resume("不存在的线程")
        # 没有检查点 -> 当成新任务跑（query 为空但流程能走完）
        self.assertIsNotNone(run.state)

    def test_no_checkpointer_does_not_crash(self):
        engine, _ = build_engine(decisions=[{"final": "答案"}], checkpointer=None)
        run = engine.run("查制度")
        self.assertTrue(run.ok)

    def test_json_file_checkpoint_end_to_end(self):
        tmp = make_tmp()
        try:
            cp = JsonFileCheckpointer(tmp)
            engine, _ = build_engine(decisions=[{"final": "落盘答案"}],
                                     checkpointer=cp)
            engine.run("查制度", thread_id="T9")
            engine2, _ = build_engine(decisions=[{"final": "不该用到"}],
                                      checkpointer=JsonFileCheckpointer(tmp))
            run2 = engine2.resume("T9")
            self.assertEqual(run2.state.answer, "落盘答案")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestAgentRobustness(unittest.TestCase):
    def test_act_fn_exception_is_caught(self):
        def boom(state, allowed):
            raise RuntimeError("模型挂了")

        engine, _ = build_engine(act_fn=boom)
        run = engine.run("查制度")
        self.assertIsNotNone(run.error)
        self.assertIn("模型挂了", run.error)
        self.assertEqual(run.state.stop_reason, "error")

    def test_state_always_json_serializable_after_run(self):
        engine, _ = build_engine(decisions=[
            {"tool": "kb_search", "args": {"query": "x"}},
            {"final": "答案"},
        ])
        run = engine.run("查制度")
        json.dumps(run.state.to_dict(), ensure_ascii=False)

    def test_empty_query_still_runs(self):
        engine, _ = build_engine(decisions=[{"final": "兜底"}])
        run = engine.run("")
        self.assertTrue(run.state.finished)

    def test_thread_id_autogenerated(self):
        engine, _ = build_engine(decisions=[{"final": "x"}])
        run = engine.run("查制度")
        self.assertTrue(run.state.thread_id)


class TestMakeEvent(unittest.TestCase):
    def test_shape(self):
        e = make_event(EventType.TOOL_CALL, tool="kb_search")
        self.assertEqual(e["type"], "tool_call")
        self.assertEqual(e["data"]["tool"], "kb_search")
        self.assertIn("ts", e)


class TestFullMultiToolFlow(unittest.TestCase):
    """一个更接近真实的多步任务：先检索 -> 再查工单 -> 汇总作答。"""

    def test_three_step_task(self):
        reg = ToolRegistry()

        @reg.tool(description="检索知识库")
        def kb_search(query: str):
            """检索

            query: 检索词
            """
            return {"hits": [{"id": "c1", "text": "制度: 需人工复核"}]}

        @reg.tool(description="查询工单")
        def ticket_query(ticket_id: str):
            """查工单

            ticket_id: 工单号
            """
            return {"ticket_id": ticket_id, "status": "closed"}

        decisions = [
            {"tool": "kb_search", "args": {"query": "复核制度"}},
            {"tool": "ticket_query", "args": {"ticket_id": "T-100"}},
            {"final": "制度要求人工复核 [1]，工单 T-100 已关闭 [2]",
             "citations": ["c1", "T-100"]},
        ]
        calls = {"n": 0}

        def act_fn(state, allowed):
            d = decisions[min(calls["n"], len(decisions) - 1)]
            calls["n"] += 1
            return d

        cp = InMemoryCheckpointer()
        engine = AgentEngine(
            registry=reg,
            intent_fn=lambda q: "mixed",
            plan_fn=lambda q, t: ["检索制度", "查工单", "汇总"],
            act_fn=act_fn,
            route_table={"mixed": ["kb_search", "ticket_query"]},
            checkpointer=cp,
        )
        run = engine.run("查一下复核制度并看看工单 T-100")
        self.assertTrue(run.ok, run.error)
        self.assertEqual(run.state.tool_calls, 2)
        self.assertEqual(len(run.state.citations), 2)
        tools_used = [s["tool"] for s in run.state.scratchpad if s.get("tool")]
        self.assertEqual(tools_used, ["kb_search", "ticket_query"])
        # 事件流完整可回放
        types = [e["type"] for e in run.events]
        self.assertEqual(types.count(EventType.TOOL_CALL), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
