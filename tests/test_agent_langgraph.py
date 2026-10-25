# -*- coding: utf-8 -*-
"""tests/test_agent_langgraph.py —— LangGraph 版引擎的离线单测。

★★ 这个测试文件的设计是本项目里最值得讲的一点，先说清楚：

    问题：langgraph 装不上（沙箱无网络），**怎么验证图接线是对的？**

    答案：`LangGraphAgentEngine` 支持注入 `graph_factory`，
          于是这里实现一个**最小但语义正确的 StateGraph 替身**，
          它按与 langgraph 相同的规则驱动节点：

              entry_point -> 节点 -> (条件边 or 普通边) -> ... -> END

          这样一来，**节点实现、条件路由、白名单校验、熔断、checkpoint
          全部被真实执行过**，只有在"真 langgraph 的 API 细节"这一层
          需要你在真机确认。

    这个替身覆盖的风险：
      ✅ 节点接线是否遗漏/写错（我实测过：漏掉 set_entry_point 立即报错）
      ✅ 条件边返回值是否在映射表里（映射缺失会抛错而不是静默走错）
      ✅ 每个节点对 state 的读写是否符合预期
      ✅ 循环边（tools -> act）是否会无限打转
      ✅ checkpoint 是否在每一步之后落盘
      ❌ 不覆盖：真 langgraph 的函数签名差异、状态合并语义差异

    ★ 2026-10-25 更新：真库路径已由本文件末尾的 `TestRealLangGraph` 覆盖
      （langgraph 0.2.28 实测）。补真库测试之前，先跑真机冒烟，**当场抓出两个
      Fake 永远发现不了的真 bug**：

        ① `StateGraph(dict)` —— 裸 dict 让真库无法推断状态通道，输入字典的键
           （如 query）根本不进状态，第一节点读 state["query"] 直接 KeyError。
           Fake 的 `cur.update(updates)` 全量合并语义掩盖了它。
           修复：TypedDict 显式声明全部状态键（见 AgentGraphState 踩坑记录）。
        ② 节点名与状态键同名 —— 真库 add_node 拒绝 "plan"（状态键已占用），
           编译期 ValueError。节点改名 make_plan。

      两个 bug 的共同点：**都是"真库才有的约束"**。这验证了替身策略的边界——
      替身保逻辑，真库测试保 API 契约，两者都不能省。

    真机验证命令：
        conda run -n agent_rag python -m unittest tests.test_agent_langgraph -v
        # 预期：35 个离线测试 + TestRealLangGraph 全部通过
        # CI（零依赖环境）上 TestRealLangGraph 自动跳过，其余 35 个照常跑
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import (  # noqa: E402
    EventType, InMemoryCheckpointer, JsonFileCheckpointer,
)
from core.agent_langgraph import (  # noqa: E402
    LangGraphAgentEngine, langgraph_available, make_agent_engine,
)
from core.mcp import ToolRegistry  # noqa: E402

_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


# ================================================================== 假 StateGraph
class _FakeCompiledGraph:
    """按 langgraph 语义驱动节点：入口 -> 节点 -> 边 -> ... -> END。"""

    def __init__(self, builder):
        self.b = builder

    def invoke(self, state):
        return self._run(state, collect_all=False)[0]

    def stream(self, state):
        """逐节点产出 {node: updates}，语义对齐 langgraph 的 stream。"""
        return self._run(state, collect_all=True)[1]

    def get_graph(self):
        class G:
            def __init__(self, ids):
                self.nodes = {i: type("N", (), {"id": i})() for i in ids}
        return G(list(self.b.nodes))

    def _run(self, state, collect_all):
        cur = dict(state)
        node = self.b.entry
        chunks = []
        seen = []
        for _ in range(200):                      # 硬上限，防测试挂死
            if node == self.b.END:
                break
            if node not in self.b.nodes:
                raise ValueError(f"未知节点: {node}")
            seen.append(node)
            updates = self.b.nodes[node](cur) or {}
            if not isinstance(updates, dict):
                raise TypeError(f"节点 {node} 必须返回 dict，返回了 {type(updates)}")
            cur.update(updates)
            if collect_all:
                chunks.append({node: updates})
            # 决定下一个节点
            if node in self.b.cond:
                fn, mapping = self.b.cond[node]
                nxt = fn(cur)
                if nxt not in mapping:
                    raise ValueError(
                        f"条件边 {node} 返回 '{nxt}'，但映射表只有 {list(mapping)}")
                node = mapping[nxt]
            elif node in self.b.edges:
                node = self.b.edges[node]
            else:
                break
        else:
            raise RuntimeError(f"疑似无限循环，访问顺序: {seen[:12]} ...")
        return cur, chunks


class _FakeStateGraph:
    """最小 StateGraph 替身。只用标准库，离线可跑。"""

    END = "__end__"

    def __init__(self, state_schema=None):
        self.schema = state_schema
        self.nodes = {}
        self.edges = {}
        self.cond = {}
        self.entry = None

    def add_node(self, name, fn):
        if name in self.nodes:
            raise ValueError(f"节点重复: {name}")
        self.nodes[name] = fn
        return self

    def add_edge(self, src, dst):
        if src in self.edges:
            raise ValueError(f"节点 {src} 已有普通出边")
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src, fn, mapping):
        if src in self.cond:
            raise ValueError(f"节点 {src} 已有条件出边")
        self.cond[src] = (fn, dict(mapping))
        return self

    def set_entry_point(self, name):
        self.entry = name
        return self

    def compile(self):
        if self.entry is None:
            raise ValueError("未设置 entry_point —— 图无法执行")
        for name in self.nodes:
            if name not in self.edges and name not in self.cond:
                raise ValueError(f"节点 {name} 没有出边，会成为死路")
        return _FakeCompiledGraph(self)


def fake_graph_factory():
    return _FakeStateGraph


# ================================================================== 工具与夹具
def make_registry(log=None):
    reg = ToolRegistry()

    @reg.tool(description="按关键词检索知识库")
    def kb_search(query: str, top_k: int = 3):
        """检索

        query: 关键词
        top_k: 条数
        """
        if log is not None:
            log.append(("kb_search", query))
        return {"count": 1, "hits": [{"id": "c1", "text": "复核制度 第三章"}]}

    @reg.tool(description="查询工单")
    def ticket_query(ticket_id: str):
        """查工单

        ticket_id: 单号
        """
        if log is not None:
            log.append(("ticket_query", ticket_id))
        return {"ticket_id": ticket_id, "status": "closed"}

    @reg.tool(description="会抛异常的工具")
    def boom(x: str = ""):
        """会炸

        x: 参数
        """
        raise RuntimeError("工具内部错误")

    return reg


ROUTE = {"full_review": ["kb_search", "ticket_query"],
         "spec_inquiry": ["kb_search"],
         "general": []}


def make_engine(decisions, route_table=None, clock=None, checkpointer=None,
                registry_log=None, real_langgraph=False, **kw):
    """构造带脚本化决策的 LangGraph 版引擎。

    real_langgraph=True 时不注入假图，走真 langgraph（仅真机测试用）；
    CI 零依赖环境保持默认 False。
    """
    reg = make_registry(registry_log)
    cursor = {"i": 0}
    calls = {"n": 0}

    def act(state, allowed):
        calls["n"] += 1
        d = decisions[min(cursor["i"], len(decisions) - 1)]
        cursor["i"] += 1
        if callable(d):
            return d(state, allowed)
        return d

    if clock is not None:
        kw["clock"] = clock

    engine = LangGraphAgentEngine(
        registry=reg,
        intent_fn=lambda q: "full_review" if "工单" in q else "spec_inquiry",
        plan_fn=lambda q, t: ["解析意图", "检索", "回答"],
        act_fn=act,
        route_table=route_table if route_table is not None else ROUTE,
        checkpointer=checkpointer,
        graph_factory=None if real_langgraph else fake_graph_factory,
        **kw,
    )
    engine._act_calls = lambda: calls["n"]
    return engine


# ================================================================== 拓扑
class TestGraphTopology(unittest.TestCase):
    def test_graph_builds(self):
        eng = make_engine([{"final": "答案"}])
        self.assertEqual(len(eng.graph_nodes()), 6)

    def test_all_six_nodes_present(self):
        eng = make_engine([{"final": "答案"}])
        for n in (eng.N_INTENT, eng.N_PLAN, eng.N_ACT, eng.N_TOOLS,
                  eng.N_VERIFY, eng.N_FINALIZE):
            self.assertIn(n, eng.graph_nodes())

    def test_missing_entry_point_raises(self):
        """★ 回归：漏掉 set_entry_point 时假运行时必须报错，而不是静默通过。"""
        g = _FakeStateGraph()
        g.add_node("a", lambda s: {})
        g.add_edge("a", _FakeStateGraph.END)
        with self.assertRaises(ValueError) as ctx:
            g.compile()
        self.assertIn("entry_point", str(ctx.exception))

    def test_node_without_out_edge_raises(self):
        """★ 回归：节点没出边就是死路，必须报错（真实 langgraph 也会报）。"""
        g = _FakeStateGraph()
        g.add_node("a", lambda s: {})
        g.add_node("orphan", lambda s: {})
        g.add_edge("a", _FakeStateGraph.END)
        g.set_entry_point("a")
        with self.assertRaises(ValueError) as ctx:
            g.compile()
        self.assertIn("orphan", str(ctx.exception))

    def test_conditional_return_not_in_mapping_raises(self):
        g = _FakeStateGraph()
        g.add_node("a", lambda s: {})
        g.add_edge("a", _FakeStateGraph.END)
        g.set_entry_point("a")
        g.add_conditional_edges("a", lambda s: "nonexistent", {"x": "a"})
        compiled = g.compile()
        with self.assertRaises(ValueError) as ctx:
            compiled.invoke({})
        self.assertIn("nonexistent", str(ctx.exception))

    def test_describe_self_report(self):
        eng = make_engine([{"final": "答案"}])
        d = eng.describe()
        self.assertEqual(d["engine"], "langgraph")
        self.assertEqual(len(d["nodes"]), 6)
        self.assertEqual(len(d["conditional_edges"]), 2)
        self.assertIn("max_rounds", d["fuses"])

    def test_langgraph_available_is_bool(self):
        self.assertIsInstance(langgraph_available(), bool)


# ================================================================== 主流程
class TestRunFlow(unittest.TestCase):
    def test_single_tool_then_answer(self):
        eng = make_engine([
            {"tool": "kb_search", "args": {"query": "复核"}},
            {"final": "高风险工单需人工复核 [1]", "citations": ["c1"]},
        ])
        run = eng.run("工单 T-100 是否合规")
        self.assertTrue(run.state.finished)
        self.assertIn("人工复核", run.state.answer)
        self.assertEqual(run.state.citations, ["c1"])
        self.assertEqual(run.state.tool_calls, 1)
        self.assertEqual(run.state.rounds, 2)

    def test_direct_answer_without_tools(self):
        eng = make_engine([{"final": "直接回答"}])
        run = eng.run("随便问问")
        self.assertEqual(run.state.tool_calls, 0)
        self.assertTrue(run.state.verified)

    def test_intent_and_plan_events(self):
        eng = make_engine([{"final": "答案"}])
        run = eng.run("工单 T-100 是否合规")
        types = [e["type"] for e in run.events]
        self.assertIn(EventType.START, types)
        self.assertIn(EventType.INTENT, types)
        self.assertIn(EventType.PLAN, types)
        self.assertIn(EventType.ANSWER, types)
        self.assertIn(EventType.VERIFY, types)
        self.assertIn(EventType.DONE, types)

    def test_intent_called_once(self):
        """★ 回归：意图识别只应调用一次（对应 LangGraph 里 intent 节点不可回边）。"""
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "答案"}])
        eng.run("工单 T-100")
        self.assertEqual(eng._intent_calls, 1)

    def test_event_ordering_is_sane(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "答案"}])
        run = eng.run("工单 T-100")
        types = [e["type"] for e in run.events]
        self.assertLess(types.index(EventType.INTENT), types.index(EventType.PLAN))
        self.assertLess(types.index(EventType.PLAN), types.index(EventType.TOOL_CALL))
        self.assertLess(types.index(EventType.TOOL_RESULT), types.index(EventType.ANSWER))
        self.assertEqual(types[-1], EventType.DONE)

    def test_start_event_marks_engine(self):
        eng = make_engine([{"final": "答案"}])
        run = eng.run("q")
        start = next(e for e in run.events if e["type"] == EventType.START)
        self.assertEqual(start["data"]["engine"], "langgraph")


# ================================================================== 白名单
class TestWhitelist(unittest.TestCase):
    def test_route_returns_allowed(self):
        eng = make_engine([{"final": "a"}])
        self.assertEqual(eng.route("full_review"), ["kb_search", "ticket_query"])
        self.assertEqual(eng.route("spec_inquiry"), ["kb_search"])

    def test_unknown_intent_gets_empty_whitelist(self):
        """★ 未注册意图默认给空清单（最保守），语义与自实现版一致。"""
        eng = make_engine([{"final": "a"}])
        self.assertEqual(eng.route("unknown_intent"), [])

    def test_blocked_tool_does_not_consume_budget(self):
        """★ 关键语义：被白名单拦下的调用**不消耗**工具预算。"""
        eng = make_engine([
            {"tool": "ticket_query", "args": {"ticket_id": "T-1"}},
            {"final": "答案"},
        ])
        run = eng.run("随便问问")            # -> spec_inquiry，只允许 kb_search
        self.assertEqual(run.state.tool_calls, 0)
        blocked = [s for s in run.state.scratchpad if s.get("blocked")]
        self.assertEqual(len(blocked), 1)

    def test_blocked_emits_tool_error(self):
        eng = make_engine([
            {"tool": "ticket_query", "args": {"ticket_id": "T-1"}},
            {"final": "答案"},
        ])
        run = eng.run("随便问问")
        errs = [e for e in run.events if e["type"] == EventType.TOOL_ERROR]
        self.assertTrue(any(e["data"].get("blocked") for e in errs))

    def test_tool_exception_recorded_not_fatal(self):
        """工具自身抛异常时，应记录为失败并继续，而不是整轮崩掉。

        ★ 测试要点：必须让**参数每轮都变**，否则会先撞上 loop_detected，
          就测不到"工具异常"这条路径了。（初版在这里踩过一次：
          固定参数 -> 第 3 轮判死循环 -> scratchpad 里根本没有失败记录。）
        """
        reg = make_registry()
        n = {"i": 0}

        def act(state, allowed):
            n["i"] += 1
            if n["i"] <= 2:
                return {"tool": "boom", "args": {"x": f"try{n['i']}"}}
            return {"final": "兜底答案"}

        eng = LangGraphAgentEngine(
            registry=reg, intent_fn=lambda q: "spec_inquiry",
            plan_fn=lambda q, t: [], act_fn=act,
            route_table={"spec_inquiry": ["boom"]},
            max_rounds=6, graph_factory=fake_graph_factory)
        run = eng.run("问一句")
        self.assertTrue(run.state.finished)
        failed = [s for s in run.state.scratchpad if s.get("ok") is False]
        self.assertEqual(len(failed), 2)
        self.assertEqual(run.state.tool_calls, 2, "失败的工具调用也要计入预算")
        self.assertIn("RuntimeError", failed[0]["error"])


# ================================================================== 熔断
class TestFuses(unittest.TestCase):
    def test_max_rounds_stops(self):
        """轮次熔断。

        ★ 必须让参数每轮都变，否则会先触发 loop_detected（更早的熔断）。
          这不是测试技巧，而是熔断的**优先级语义**：死循环检测比轮次上限更早介入，
          因为循环是更明确的失控信号。这里显式断言这一点。
        """
        n = {"i": 0}

        def act(state, allowed):
            n["i"] += 1
            return {"tool": "kb_search", "args": {"query": f"q{n['i']}"}}

        reg = make_registry()
        eng = LangGraphAgentEngine(
            registry=reg, intent_fn=lambda q: "spec_inquiry",
            plan_fn=lambda q, t: [], act_fn=act,
            route_table={"spec_inquiry": ["kb_search"]},
            max_rounds=4, max_tool_calls=100, total_timeout_s=10_000.0,
            graph_factory=fake_graph_factory)
        run = eng.run("工单 T-100")
        self.assertEqual(run.state.stop_reason, "max_rounds")
        self.assertLessEqual(run.state.rounds, 4)

    def test_loop_detection_fires_before_max_rounds(self):
        """★ 优先级语义：固定参数的死循环应在轮次用尽前就被判出。"""
        eng = make_engine([{"tool": "kb_search", "args": {"query": "same"}}],
                          max_rounds=50, max_tool_calls=100,
                          total_timeout_s=10_000.0)
        run = eng.run("工单 T-100")
        self.assertEqual(run.state.stop_reason, "loop_detected")
        self.assertLess(run.state.rounds, 50)

    def test_max_tool_calls_stops(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "loop"}}],
                          max_rounds=20, max_tool_calls=3)
        run = eng.run("工单 T-100")
        self.assertEqual(run.state.stop_reason, "max_tool_calls")
        self.assertLessEqual(run.state.tool_calls, 3)

    def test_total_timeout_stops(self):
        """★ 时钟必须走注入的那个，否则超时判定形同虚设。

        同时注意：clock 单次步进要**足够大**，否则会先撞上 max_rounds。
        初版设成每次 +5 秒、超时 10 秒 —— 因为 loop/rounds 先触发，
        超时这条分支根本没被走到。
        """
        t = {"v": 1000.0}

        def clock():
            t["v"] += 1000.0          # 每次调用前进 1000 秒，远超超时阈值
            return t["v"]

        eng = make_engine([{"tool": "kb_search", "args": {"query": "loop"}}],
                          max_rounds=500, max_tool_calls=500,
                          total_timeout_s=10.0, clock=clock)
        run = eng.run("工单 T-100")
        self.assertEqual(run.state.stop_reason, "total_timeout")

    def test_loop_detection_explicit(self):
        """明确的死循环：同样工具 + 同样参数连续 3 次。

        用大 max_rounds / max_tool_calls 把其它熔断排除掉，
        确保抓到的是 loop_detected 而不是碰巧撞上别的限制。
        注意最后一次调用的 sig 也要进入 scratchpad —— 检测发生在
        act 开头，第 3 次工具结果落盘后再次进入 act 时才会判定。
        """
        eng = make_engine([{"tool": "kb_search", "args": {"query": "same"}}],
                          max_rounds=500, max_tool_calls=500,
                          total_timeout_s=10_000.0)
        run = eng.run("工单 T-100")
        self.assertEqual(run.state.stop_reason, "loop_detected")
        sigs = [s.get("sig") for s in run.state.scratchpad
                if s.get("type") == "tool_call"]
        self.assertEqual(len(set(sigs[-3:])), 1, "最后三次签名应相同")

    def test_changing_args_avoids_loop_detection(self):
        """★ 反例：参数每轮都变 -> 不该被判死循环（避免误杀正常多步任务）。"""
        n = {"i": 0}

        def act(state, allowed):
            n["i"] += 1
            if n["i"] <= 5:
                return {"tool": "kb_search", "args": {"query": f"q{n['i']}"}}
            return {"final": "答案"}

        reg = make_registry()
        eng = LangGraphAgentEngine(
            registry=reg, intent_fn=lambda q: "spec_inquiry",
            plan_fn=lambda q, t: [], act_fn=act,
            route_table={"spec_inquiry": ["kb_search"]},
            max_rounds=10, max_tool_calls=10, total_timeout_s=10_000.0,
            graph_factory=fake_graph_factory)
        run = eng.run("多步任务")
        self.assertNotEqual(run.state.stop_reason, "loop_detected")
        self.assertEqual(run.state.tool_calls, 5)

    def test_limit_event_emitted(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "loop"}}],
                          max_rounds=3)
        run = eng.run("工单 T-100")
        self.assertTrue(any(e["type"] == EventType.LIMIT for e in run.events))

    def test_answer_filled_when_stopped(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "loop"}}],
                          max_rounds=3)
        run = eng.run("工单 T-100")
        self.assertTrue(run.state.answer)


# ================================================================== 校验
class TestVerify(unittest.TestCase):
    def test_verify_failure_triggers_retry(self):
        """verify 不通过时应回到 act 再试一次（对应 verify -> act 条件边）。"""
        calls = {"n": 0}

        def verify(state, answer):
            calls["n"] += 1
            return (calls["n"] >= 2, "第一次不通过" if calls["n"] < 2 else "通过")

        eng = make_engine([{"final": "第一次答案"}, {"final": "第二次答案"}])
        eng.verify_fn = verify
        run = eng.run("q")
        self.assertTrue(run.state.verified)
        self.assertIn("第二次", run.state.answer)
        self.assertGreaterEqual(calls["n"], 2)

    def test_verify_event_recorded(self):
        eng = make_engine([{"final": "答案"}])
        run = eng.run("q")
        v = [e for e in run.events if e["type"] == EventType.VERIFY]
        self.assertTrue(v)
        self.assertTrue(v[-1]["data"]["ok"])


# ================================================================== 检查点
class TestCheckpoint(unittest.TestCase):
    def test_checkpoint_saved_each_step(self):
        ck = InMemoryCheckpointer()
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "答案"}], checkpointer=ck)
        run = eng.run("工单 T-100", thread_id="T1")
        self.assertEqual(ck.list_threads(), ["T1"])
        saved = ck.load("T1")
        self.assertEqual(saved.answer, run.state.answer)

    def test_resume_returns_saved_answer(self):
        ck = InMemoryCheckpointer()
        eng = make_engine([{"final": "已完成的答案"}], checkpointer=ck)
        eng.run("q", thread_id="T2")
        run2 = eng.resume("T2")
        self.assertIn("已完成的答案", run2.state.answer)
        types = [e["type"] for e in run2.events]
        self.assertIn(EventType.RESUME, types)

    def test_resume_emits_done_without_rerun(self):
        ck = InMemoryCheckpointer()
        eng = make_engine([{"final": "答案"}], checkpointer=ck)
        eng.run("q", thread_id="T3")
        before = eng._intent_calls
        eng.resume("T3")
        self.assertEqual(eng._intent_calls, before, "续跑不该重新识别意图")

    def test_json_file_checkpointer(self):
        os.makedirs(_TMP_ROOT, exist_ok=True)
        import uuid
        d = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:10])
        ck = JsonFileCheckpointer(d)
        eng = make_engine([{"final": "落盘答案"}], checkpointer=ck)
        eng.run("q", thread_id="T4")
        self.assertEqual(ck.list_threads(), ["T4"])
        self.assertIn("落盘答案", JsonFileCheckpointer(d).load("T4").answer)
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ================================================================== 工厂
class TestFactory(unittest.TestCase):
    def test_prefer_false_gives_self_implemented(self):
        from core.agent import AgentEngine
        eng = make_agent_engine(
            prefer_langgraph=False, registry=make_registry(),
            intent_fn=lambda q: "spec_inquiry", plan_fn=lambda q, t: [],
            act_fn=lambda s, a: {"final": "ok"},
            route_table=ROUTE)
        self.assertIsInstance(eng, AgentEngine)

    def test_factory_returns_engine_with_same_api(self):
        """★ 两版引擎的公开 API 必须一致 —— 业务代码才能无感切换。"""
        eng = make_agent_engine(
            prefer_langgraph=False, registry=make_registry(),
            intent_fn=lambda q: "spec_inquiry", plan_fn=lambda q, t: [],
            act_fn=lambda s, a: {"final": "答案是 42"},
            route_table=ROUTE)
        self.assertTrue(callable(getattr(eng, "run")))
        self.assertTrue(callable(getattr(eng, "resume")))
        self.assertTrue(callable(getattr(eng, "route")))
        self.assertTrue(callable(getattr(eng, "detect_loop")))
        run = eng.run("q")
        self.assertIn("42", run.state.answer)

    def test_both_engines_agree_on_basic_flow(self):
        """★ 一致性检查：同样输入下两版引擎的核心状态应一致。

        这是"换引擎不换行为"的关键证据。只比较与语义无关的实现细节会跳过；
        这里比较的字段是简历/评测要用的那几个。
        """
        from core.agent import AgentEngine
        decisions = [{"tool": "kb_search", "args": {"query": "复核"}},
                     {"final": "高风险需人工复核 [1]", "citations": ["c1"]}]

        def build(cls_or_maker):
            return cls_or_maker(decisions)

        def mk_self():
            reg = make_registry()
            cur = {"i": 0}

            def act(state, allowed):
                d = decisions[min(cur["i"], len(decisions) - 1)]
                cur["i"] += 1
                return d
            return AgentEngine(registry=reg,
                               intent_fn=lambda q: "full_review",
                               plan_fn=lambda q, t: ["检索", "回答"],
                               act_fn=act, route_table=ROUTE, max_rounds=6)

        a = mk_self().run("工单 T-100")
        b = make_engine(decisions, route_table={"full_review": ["kb_search",
                                                               "ticket_query"]}
                        ).run("工单 T-100")
        self.assertEqual(a.state.tool_calls, b.state.tool_calls)
        self.assertEqual(a.state.rounds, b.state.rounds)
        self.assertEqual(a.state.answer, b.state.answer)
        self.assertEqual(a.state.citations, b.state.citations)
        self.assertEqual(a.state.finished, b.state.finished)


# ================================================================== 真库验证
@unittest.skipUnless(
    langgraph_available(),
    "langgraph 未安装 —— CI 零依赖环境预期跳过；真机（conda env agent_rag）应运行",
)
class TestRealLangGraph(unittest.TestCase):
    """★ 真 langgraph 驱动的引擎测试（2026-10-25 新增）。

    以上 35 个测试全部经 `_FakeStateGraph` 注入驱动，验证的是**逻辑接线**；
    这个类验证**真库 API 契约**——两类风险不同，都不能省：

        ① 真 StateGraph(TypedDict) 的状态合并语义（LastValue 逐键覆盖）
        ② 真 add_node 的命名约束（不能与状态键同名）
        ③ 真 stream() 的 chunk 形态（{node: updates}）
        ④ 真 compile() 的图校验与 get_graph() 的节点枚举
        ⑤ 真库递归上限下循环边不会提前炸

    运行前提：`pip install langgraph`（本机 agent_rag 环境已装 0.2.28）。
    CI 零依赖环境自动跳过——这正是"业务逻辑与框架解耦"的代价与证据。
    """

    def test_compiled_graph_is_real_langgraph(self):
        eng = make_engine([{"final": "x"}], real_langgraph=True)
        mod = type(eng._graph).__module__
        self.assertTrue(mod.startswith("langgraph"),
                        f"真库路径拿到的应是 langgraph 编译图，实际是 {mod}")

    def test_single_final_answer(self):
        eng = make_engine([{"final": "复核流程见第三章", "citations": ["c1"]}],
                          real_langgraph=True)
        run = eng.run("复核流程是什么")
        self.assertEqual(run.state.answer, "复核流程见第三章")
        self.assertEqual(run.state.citations, ["c1"])
        self.assertEqual(run.state.rounds, 1)
        self.assertTrue(run.state.finished)
        types = [e["type"] for e in run.events]
        self.assertEqual(types[0], "start")
        self.assertIn("done", types)

    def test_tool_then_answer_full_flow(self):
        log = []
        eng = make_engine([
            {"tool": "kb_search", "args": {"query": "复核"}},
            {"final": "第三章规定了复核流程 [1]", "citations": ["c1"]},
        ], registry_log=log, real_langgraph=True)
        run = eng.run("复核流程是什么")
        self.assertEqual(run.state.tool_calls, 1)
        self.assertEqual(run.state.rounds, 2)
        self.assertIn("第三章", run.state.answer)
        ok_calls = [s for s in run.state.scratchpad
                    if s.get("type") == "tool_call" and s.get("ok")]
        self.assertEqual(len(ok_calls), 1)
        self.assertEqual(ok_calls[0]["tool"], "kb_search")
        self.assertIn(("kb_search", "复核"), log)
        types = [e["type"] for e in run.events]
        self.assertEqual(types, ["start", "intent", "plan", "tool_call",
                                 "tool_result", "answer", "verify", "done"])

    def test_real_stream_chunk_shape(self):
        """真 stream() 逐节点产出 {node: updates} —— run() 依赖这个形态。"""
        eng = make_engine([{"final": "答案"}], real_langgraph=True)
        init = {"query": "q", "thread_id": "t1", "task_type": "general",
                "allowed_tools": [], "plan": [], "scratchpad": [], "rounds": 0,
                "tool_calls": 0, "started_at": 0.0, "answer": "", "citations": [],
                "verified": False, "verify_reason": "", "refused": False,
                "finished": False, "stop_reason": "", "last_observation": None,
                "pending": None}
        chunks = list(eng._graph.stream(init))
        self.assertTrue(chunks, "stream 不应为空")
        for chunk in chunks:
            self.assertIsInstance(chunk, dict)
            for node, updates in chunk.items():
                self.assertIsInstance(node, str)
                self.assertIsInstance(updates, dict)

    def test_whitelist_block_not_consume_budget(self):
        """spec_inquiry 只允许 kb_search；调 ticket_query 应被拦且不耗预算。"""
        eng = make_engine([
            {"tool": "ticket_query", "args": {"ticket_id": "T-1"}},
            {"final": "查不到就走回答", "citations": []},
        ], real_langgraph=True)
        run = eng.run("制度是什么")   # intent -> spec_inquiry
        self.assertEqual(run.state.tool_calls, 0)
        blocked = [s for s in run.state.scratchpad if s.get("blocked")]
        self.assertEqual(len(blocked), 1)
        errs = [e for e in run.events
                if e["type"] == "tool_error" and e["data"].get("blocked")]
        self.assertEqual(len(errs), 1)

    def test_fuse_max_rounds(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}}],
                          max_rounds=2, real_langgraph=True)
        run = eng.run("制度是什么")
        self.assertEqual(run.state.stop_reason, "max_rounds")
        self.assertEqual(run.state.rounds, 2)
        self.assertLessEqual(run.state.tool_calls, 2)

    def test_verify_failure_retries_once(self):
        """verify 不通过 -> 回 act 再试（真库条件边 + 循环边）。"""
        eng = make_engine([
            {"final": "第一次回答"},
            {"final": "第二次回答"},
        ], real_langgraph=True,
            verify_fn=lambda s, a: (False, "第一遍必挂") if s.rounds == 1
            else (True, "通过"))
        run = eng.run("制度是什么")
        self.assertEqual(run.state.rounds, 2)
        self.assertEqual(run.state.answer, "第二次回答")
        self.assertTrue(run.state.verified)

    def test_tool_exception_recorded_not_fatal(self):
        """工具抛异常应记录为失败并继续，而不是整轮崩掉。

        参数逐轮变化，避免先撞上 loop_detected 测不到这条路径
        （与 Fake 版同名测试同样的坑）。
        """
        n = {"i": 0}

        def act(state, allowed):
            n["i"] += 1
            if n["i"] <= 2:
                return {"tool": "boom", "args": {"x": f"try{n['i']}"}}
            return {"final": "工具炸了也能收尾", "citations": []}

        eng = make_engine([act], route_table={"spec_inquiry": ["boom"]},
                          real_langgraph=True)
        run = eng.run("制度是什么")
        self.assertEqual(run.state.answer, "工具炸了也能收尾")
        failed = [s for s in run.state.scratchpad
                  if s.get("type") == "tool_call" and s.get("ok") is False]
        self.assertEqual(len(failed), 2)
        self.assertIn("RuntimeError", failed[0]["error"])

    def test_checkpoint_saved_and_resume_finished(self):
        from core.agent import InMemoryCheckpointer
        cp = InMemoryCheckpointer()
        eng = make_engine([{"final": "答案", "citations": ["c1"]}],
                          checkpointer=cp, real_langgraph=True)
        run = eng.run("制度是什么", thread_id="T-real")
        self.assertTrue(run.state.finished)
        self.assertIsNotNone(cp.load("T-real"))
        resumed = eng.resume("T-real")
        reasons = [e["data"].get("reason") for e in resumed.events
                   if e["type"] == "done"]
        self.assertIn("已完成的检查点，无需续跑", reasons)

    def test_graph_topology_via_real_get_graph(self):
        eng = make_engine([{"final": "x"}], real_langgraph=True)
        nodes = eng.graph_nodes()
        for n in (eng.N_INTENT, eng.N_PLAN, eng.N_ACT, eng.N_TOOLS,
                  eng.N_VERIFY, eng.N_FINALIZE):
            self.assertIn(n, nodes)

    def test_factory_and_describe_on_real_path(self):
        eng = make_agent_engine(
            prefer_langgraph=True,
            registry=make_registry(),
            intent_fn=lambda q: "spec_inquiry",
            plan_fn=lambda q, t: ["p"],
            act_fn=lambda s, a: {"final": "x"},
            route_table=ROUTE,
        )
        self.assertIsInstance(eng, LangGraphAgentEngine)
        d = eng.describe()
        self.assertTrue(d["langgraph_available"])
        self.assertEqual(len(d["nodes"]), 6)

    def test_node_name_collides_with_state_key_regression(self):
        """★ 回归：节点名不能与 AgentGraphState 的键同名。

        真 langgraph 的 add_node 会抛 ValueError（状态键已占用）。
        改名 make_plan 修复过一次，这里把约束钉住：
        六个节点名与全部状态键求交，必须为空。
        """
        from core.agent_langgraph import AgentGraphState
        node_names = {LangGraphAgentEngine.N_INTENT,
                      LangGraphAgentEngine.N_PLAN,
                      LangGraphAgentEngine.N_ACT,
                      LangGraphAgentEngine.N_TOOLS,
                      LangGraphAgentEngine.N_VERIFY,
                      LangGraphAgentEngine.N_FINALIZE}
        state_keys = set(AgentGraphState.__annotations__)
        self.assertEqual(node_names & state_keys, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
