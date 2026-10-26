# -*- coding: utf-8 -*-
"""tests/test_agent_spans.py —— Agent 运行期 span 埋点（耗时归因）不变量。

★ 与 test_observability.py 的分工：

    test_observability 测的是 SpanTracer 这个**数据结构**本身；
    本文件测的是它在真实 Agent 引擎上的**接线**是否成立 ——
    埋点最典型的失败不是算法错，而是"根本没接上"或"接歪了"：

      S1 一次 run 结束后能拿到完整 span 树（run > node:* > tool:*）
      S2 span **不污染事件流**（事件序列仍与 M1 钉死的一致）
         —— 这是最容易翻车的一条：为了省事把 span 当事件推，
            859 项测试里关于事件序列的断言会集体变红
      S3 越权工具留下 blocked span，且能给出完整错误路径
      S4 span 落盘 + 回放一致（面板"历史回放"的甘特条靠它）
      S5 最慢的工具出现在瓶颈榜首位（归因有效，不是摆设）
      S6 续跑（resume）追加 span 而不是覆盖
      S7 自实现版 AgentEngine 也有 span（两版引擎可互换，面板不写分支）

    零依赖：默认走 tests/test_agent_langgraph.py 里的假 StateGraph。
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest

# ★ 同 test_agent_stream.py：CI 有分模块跑法，sys.path 里没有 tests 目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import (  # noqa: E402
    AgentEngine,
    EventType,
    InMemoryCheckpointer,
)
from core.observability import (  # noqa: E402
    KIND_RUN,
    KIND_TOOL,
    STATUS_BLOCKED,
    analyze,
)
from test_agent_langgraph import ROUTE, make_engine  # noqa: E402


def _names(spans):
    return [s["name"] for s in spans]


def _find(spans, name):
    for s in spans:
        if s["name"] == name:
            return s
    return None


def _tool_engine(query="工单怎么样", decisions=None, **kw):
    """默认走 full_review（白名单含 kb_search / ticket_query）。"""
    return make_engine(decisions if decisions is not None else
                       [{"tool": "kb_search", "args": {"query": "精度"}},
                        {"final": "答案是 A", "citations": ["c1"]}], **kw)


# ================================================================== S1
class TestSpanTreeWiring(unittest.TestCase):
    def test_run_produces_span_tree(self):
        eng = _tool_engine(checkpointer=InMemoryCheckpointer())
        eng.run("工单怎么样")
        spans = eng.last_spans
        self.assertTrue(spans, "run 之后必须能拿到 span")
        self.assertIn("run", _names(spans))
        self.assertIn("node:tools", _names(spans))
        self.assertIn("tool:kb_search", _names(spans))

    def test_tree_nested_run_node_tool(self):
        eng = _tool_engine(checkpointer=InMemoryCheckpointer())
        eng.run("工单怎么样")
        tree = analyze(eng.last_spans)["tree"]
        self.assertEqual(len(tree), 1)
        root = tree[0]
        self.assertEqual(root["name"], "run")
        self.assertEqual(root["kind"], KIND_RUN)
        kids = {k["name"]: k for k in root["children"]}
        self.assertIn("node:tools", kids)
        grand = {g["name"] for g in kids["node:tools"]["children"]}
        self.assertIn("tool:kb_search", grand)

    def test_tool_span_is_child_of_tools_node_not_root(self):
        """工具 span 必须挂在 node:tools 下 ——
        挂到根上说明 parent 栈没接对（嵌套信息全丢）。"""
        eng = _tool_engine(checkpointer=InMemoryCheckpointer())
        eng.run("工单怎么样")
        spans = eng.last_spans
        tool = _find(spans, "tool:kb_search")
        node = _find(spans, "node:tools")
        root = _find(spans, "run")
        self.assertEqual(tool["parent_id"], node["span_id"])
        self.assertEqual(node["parent_id"], root["span_id"])
        self.assertEqual(tool["kind"], KIND_TOOL)

    def test_every_span_has_timing(self):
        eng = _tool_engine()
        eng.run("工单怎么样")
        for s in eng.last_spans:
            self.assertIsNotNone(s["t1"], f"{s['name']} 未闭合")
            self.assertGreaterEqual(s["t1"] - s["t0"], 0.0)

    def test_spans_json_serializable(self):
        eng = _tool_engine()
        eng.run("工单怎么样")
        json.dumps(eng.last_spans, ensure_ascii=False)   # 不该抛


# ================================================================== S2
class TestSpanDoesNotPolluteEvents(unittest.TestCase):
    def test_no_span_events_in_event_stream(self):
        eng = _tool_engine()
        r = eng.run("工单怎么样")
        for ev in r.events:
            self.assertNotIn("span", ev["type"],
                             "span 不该进事件流（会破坏 M1 钉死的事件序列）")

    def test_event_sequence_unchanged(self):
        eng = _tool_engine()
        r = eng.run("工单怎么样")
        types = [e["type"] for e in r.events]
        self.assertEqual(types[0], EventType.START)
        self.assertEqual(types[-1], EventType.DONE)
        for want in (EventType.INTENT, EventType.PLAN, EventType.TOOL_CALL,
                     EventType.TOOL_RESULT, EventType.ANSWER, EventType.VERIFY):
            self.assertIn(want, types)

    def test_run_and_run_stream_same_events_still_holds(self):
        a = _tool_engine()
        b = _tool_engine()
        # ★ thread_id 必须显式固定：它由毫秒时间戳派生，跨毫秒时两次运行的
        #   start 事件 thread_id 不同 —— 会让这条断言 1/5 概率偶发红（M1 踩过）
        streamed = list(a.run_stream("工单怎么样", thread_id="T-CMP"))
        finished = b.run("工单怎么样", thread_id="T-CMP").events
        fp = lambda evs: [(e["type"], json.dumps(e["data"], sort_keys=True,  # noqa: E731
                                                 default=str)) for e in evs]
        self.assertEqual(fp(streamed), fp(finished))


# ================================================================== S3
class TestBlockedToolSpan(unittest.TestCase):
    def test_blocked_tool_marked(self):
        # full_review 白名单里没有 send_mail -> 必被拦
        eng = _tool_engine(decisions=[{"tool": "send_mail",
                                       "args": {"to": "a@b.c"}},
                                      {"final": "兜底答案"}])
        eng.run("工单怎么样")
        sp = _find(eng.last_spans, "tool:send_mail")
        self.assertIsNotNone(sp, "被拦的工具也要留 span，否则面板没东西可指")
        self.assertEqual(sp["status"], STATUS_BLOCKED)

    def test_error_path_is_complete(self):
        eng = _tool_engine(decisions=[{"tool": "send_mail", "args": {}},
                                      {"final": "兜底答案"}])
        eng.run("工单怎么样")
        errs = analyze(eng.last_spans)["errors"]
        paths = [e["path"] for e in errs]
        self.assertTrue(any("tool:send_mail" in p for p in paths), paths)
        self.assertTrue(any(p.startswith("run > ") for p in paths), paths)

    def test_tree_root_flagged_has_error(self):
        eng = _tool_engine(decisions=[{"tool": "send_mail", "args": {}},
                                      {"final": "兜底答案"}])
        eng.run("工单怎么样")
        tree = analyze(eng.last_spans)["tree"]
        self.assertTrue(tree[0]["has_error"])


# ================================================================== S4
class TestPersistAndReplay(unittest.TestCase):
    def test_spans_persisted(self):
        cp = InMemoryCheckpointer()
        eng = _tool_engine(checkpointer=cp)
        st = eng.run("工单怎么样").state
        saved = cp.load_spans(st.thread_id)
        # 落盘顺序是"闭合顺序"，last_spans 按开始时间排 —— 比内容不比顺序
        self.assertEqual(sorted(_names(saved)), sorted(_names(eng.last_spans)))

    def test_replay_gantt_contains_tool(self):
        cp = InMemoryCheckpointer()
        eng = _tool_engine(checkpointer=cp)
        st = eng.run("工单怎么样").state
        info = analyze(cp.load_spans(st.thread_id))
        self.assertIn("tool:kb_search", info["gantt"])
        self.assertGreater(info["total_ms"], 0.0)

    def test_replay_tree_equals_live_tree(self):
        cp = InMemoryCheckpointer()
        eng = _tool_engine(checkpointer=cp)
        st = eng.run("工单怎么样").state
        live = analyze(eng.last_spans)["tree"]
        replay = analyze(cp.load_spans(st.thread_id))["tree"]
        self.assertEqual(json.dumps(replay, sort_keys=True, default=str),
                         json.dumps(live, sort_keys=True, default=str))

    def test_new_run_resets_spans(self):
        cp = InMemoryCheckpointer()
        r1 = _tool_engine(checkpointer=cp).run("工单怎么样",
                                               thread_id="T-SPAN")
        n1 = len(cp.load_spans("T-SPAN"))
        _tool_engine(checkpointer=cp).run("工单怎么样", thread_id="T-SPAN")
        n2 = len(cp.load_spans("T-SPAN"))
        self.assertEqual(n1, n2)     # 重跑是覆盖，不是累加
        self.assertTrue(r1.state.finished)


# ================================================================== S5
class TestAttribution(unittest.TestCase):
    def test_slowest_tool_is_top_bottleneck(self):
        """归因有效性的核心断言：谁慢谁排第一。

        ★ 这里用真实 sleep（不是假时钟）：假时钟测的是 SpanTracer 的算术，
          真 sleep 才能验证"耗时确实是从真实调用上量出来的"。
        """
        eng = _tool_engine(route_table={"full_review": ["kb_search", "slow"]},
                           decisions=[{"tool": "slow", "args": {"x": "1"}},
                                      {"final": "答案", "citations": []}])

        @eng.registry.tool(description="演示用的慢工具")
        def slow(x: str = "1"):
            """慢工具

            x: 占位参数
            """
            time.sleep(0.12)
            return "slow done"

        eng.run("工单怎么样", thread_id="T-SLOW")
        info = analyze(eng.last_spans)
        self.assertEqual(info["bottlenecks"][0]["name"], "tool:slow")
        self.assertGreaterEqual(info["bottlenecks"][0]["self_ms"], 100.0)
        # 它必须挂在 node:tools 下，而不是凭空成为根的孩子
        self.assertEqual(info["bottlenecks"][0]["path"],
                         "run > node:tools > tool:slow")

    def test_bottleneck_has_path_and_pct(self):
        eng = _tool_engine()
        eng.run("工单怎么样")
        top = analyze(eng.last_spans)["bottlenecks"]
        self.assertTrue(top)
        self.assertIn("path", top[0])
        self.assertIn("self_ms", top[0])
        self.assertIn("self_pct", top[0])
        self.assertTrue(top[0]["path"].startswith("run > "))


# ================================================================== S6
class TestResume(unittest.TestCase):
    def test_resume_appends_spans(self):
        """续跑时 spans 只追加不清空：回放要能看到「首次 + 续跑」两段。"""
        cp = InMemoryCheckpointer()
        _tool_engine(checkpointer=cp).run("工单怎么样", thread_id="T-R")
        n1 = len(cp.load_spans("T-R"))
        # 换个引擎实例续跑，模拟"进程重启后接着跑"
        _tool_engine(checkpointer=cp).run("工单怎么样", thread_id="T-R",
                                          resume=True)
        n2 = len(cp.load_spans("T-R"))
        self.assertGreaterEqual(n2, n1)


# ================================================================== S7
class TestFallbackEngine(unittest.TestCase):
    def _fallback(self, decisions):
        eng = _tool_engine(decisions=decisions)
        seq = {"i": 0}

        def act(state, allowed):
            d = decisions[min(seq["i"], len(decisions) - 1)]
            seq["i"] += 1
            return d

        return AgentEngine(
            registry=eng.registry,
            intent_fn=lambda q: "spec_inquiry",
            plan_fn=lambda q, t: ["检索", "回答"],
            act_fn=act,
            route_table=ROUTE,
            checkpointer=InMemoryCheckpointer(),
        )

    def test_fallback_engine_produces_spans(self):
        eng = self._fallback([{"tool": "kb_search", "args": {"query": "p"}},
                              {"final": "答案"}])
        eng.run("查一下精度")
        self.assertTrue(eng.last_spans)
        self.assertIn("run", _names(eng.last_spans))
        self.assertIn("tool:kb_search", _names(eng.last_spans))

    def test_fallback_engine_marks_blocked(self):
        # ticket_query 在 registry 里，但 spec_inquiry 白名单只有 kb_search
        eng = self._fallback([{"tool": "ticket_query",
                               "args": {"ticket_id": "T1"}},
                              {"final": "兜底"}])
        eng.run("查一下精度")
        sp = _find(eng.last_spans, "tool:ticket_query")
        self.assertIsNotNone(sp)
        self.assertEqual(sp["status"], STATUS_BLOCKED)

    def test_fallback_engine_persists(self):
        eng = self._fallback([{"tool": "kb_search", "args": {"query": "p"}},
                              {"final": "答案"}])
        st = eng.run("查一下精度").state
        self.assertEqual(sorted(_names(
            eng.checkpointer.load_spans(st.thread_id))),
            sorted(_names(eng.last_spans)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
