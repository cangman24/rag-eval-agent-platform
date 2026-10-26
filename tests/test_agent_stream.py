# -*- coding: utf-8 -*-
"""tests/test_agent_stream.py —— Agent 轨迹事件「实时流式」不变量。

★ 为什么单独一个文件（而不是并进 test_agent_langgraph.py）：

    简历声称「SSE 实时输出 Agent 推理轨迹，支撑前端可视化面板定位错误节点」。
    要兑现它，`run()` 那种「跑完一次性返回 events」的形态是不够的 ——
    必须在**每个图节点后**就把已产生的事件增量推给调用方。

    本文件锁死 5 条不变量：

      I1 run_stream() 与 run() 产生**完全相同**的事件序列
         （重构 run() 时最容易破坏的东西，必须钉死）
      I2 事件是**增量**产出的：拿到第一个事件时工具还没被调用
         （这条区分「真流式」与「跑完再回放」）
      I3 熔断 / 越权 / 工具异常都能在事件流里被观测到
         （面板「定位错误节点」就靠这些事件）
      I4 每个事件带 ts，且单调不减（面板画时间线用）
      I5 消费完后 last_state 可用（HTTP 层要拿最终答案）

    零依赖：默认走 tests/test_agent_langgraph.py 里的假 StateGraph。
"""
from __future__ import annotations

import json
import os
import sys
import unittest

# ★ 本文件复用 test_agent_langgraph.py 里的 make_engine 与假 StateGraph。
#   `python -m unittest discover -s tests` 会把 tests 目录加进 sys.path，
#   但 CI 还有一种分模块跑法 `python -m unittest tests.test_agent_stream`，
#   那种模式下 sys.path 里只有仓库根 —— 没有这两行就会出现
#   「本地 discover 全绿、CI 分模块跑 ImportError」的鬼故事。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import EventType  # noqa: E402
from test_agent_langgraph import make_engine  # noqa: E402


def _types(events):
    return [e["type"] for e in events]


def _fingerprint(events):
    """事件指纹：类型 + 数据（不含 ts，两次运行 ts 必然不同）。"""
    return [(e["type"],
             json.dumps(e["data"], sort_keys=True, ensure_ascii=False,
                        default=str))
            for e in events]


def _drain(gen):
    return list(gen)


class TestStreamParity(unittest.TestCase):
    """I1：run_stream 与 run 的事件必须完全一致。"""

    def test_same_event_sequence_as_run(self):
        decisions = [{"tool": "kb_search", "args": {"query": "精度"}},
                     {"final": "答案是 A", "citations": ["c1"]}]
        a = make_engine(list(decisions))
        b = make_engine(list(decisions))
        # ★ 必须显式指定 thread_id：它由毫秒时间戳派生，两次运行跨毫秒时
        #   start 事件里的 thread_id 就不同 —— 测试会变成 1/40 概率的偶发红。
        streamed = _drain(a.run_stream("倒伏面积怎么评", thread_id="T-FIX"))
        finished = b.run("倒伏面积怎么评", thread_id="T-FIX")
        self.assertEqual(_fingerprint(streamed), _fingerprint(finished.events))
        self.assertEqual(_types(streamed), _types(finished.events))

    def test_run_still_returns_agent_run(self):
        """重构不能改 run() 的返回契约。"""
        eng = make_engine([{"final": "完成"}])
        r = eng.run("查点东西")
        self.assertTrue(hasattr(r, "state"))
        self.assertTrue(hasattr(r, "events"))
        self.assertEqual(r.state.answer, "完成")
        self.assertEqual(_types(r.events)[0], EventType.START)
        self.assertEqual(_types(r.events)[-1], EventType.DONE)

    def test_events_cover_full_trajectory(self):
        """一次带工具调用的完整轨迹应当出现这些节点级事件。"""
        decisions = [{"tool": "kb_search", "args": {"query": "精度"}},
                     {"final": "答案是 A"}]
        eng = make_engine(decisions)
        types = _types(_drain(eng.run_stream("查精度")))
        for want in (EventType.START, EventType.INTENT, EventType.PLAN,
                     EventType.TOOL_CALL, EventType.TOOL_RESULT,
                     EventType.ANSWER, EventType.VERIFY, EventType.DONE):
            self.assertIn(want, types, f"轨迹缺少事件 {want}：{types}")


class TestStreamIsIncremental(unittest.TestCase):
    """I2：这是「真流式」与「跑完回放」的分水岭。"""

    def test_first_event_arrives_before_any_tool_call(self):
        log = []
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "done"}],
                          registry_log=log)
        gen = eng.run_stream("查一下")
        first = next(gen)
        self.assertEqual(first["type"], EventType.START)
        # ★ 关键断言：拿到 start 时，工具一次都还没被调过
        self.assertEqual(log, [], "start 事件到达时工具已被调用 —— 不是流式")

    def test_tool_result_arrives_before_final_answer(self):
        """事件按发生顺序到达，不是攒到最后。"""
        log = []
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "done"}],
                          registry_log=log)
        gen = eng.run_stream("查一下")
        seen = []
        for ev in gen:
            seen.append((ev["type"], len(log)))
            if ev["type"] == EventType.TOOL_RESULT:
                # 此刻恰好调过 1 次工具，且还没到 answer
                self.assertEqual(len(log), 1)
                self.assertNotIn(EventType.ANSWER, [s[0] for s in seen])
        self.assertIn(EventType.ANSWER, [s[0] for s in seen])

    def test_generator_can_be_abandoned_midway(self):
        """调用方可以中途放弃（客户端断开）—— 不能因为没跑完就报错。"""
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "done"}])
        gen = eng.run_stream("查一下")
        self.assertEqual(next(gen)["type"], EventType.START)
        gen.close()
        # 不要求任何副作用；能安全关闭即可


class TestStreamErrorObservability(unittest.TestCase):
    """I3：面板要「定位错误节点」，这些事件必须能被观测到。"""

    def test_blocked_tool_emits_tool_error(self):
        # query 不含「工单」→ spec_inquiry → 只允许 kb_search
        eng = make_engine([{"tool": "ticket_query", "args": {"ticket_id": "T1"}},
                           {"final": "兜底"}])
        types = _types(_drain(eng.run_stream("查个东西")))
        self.assertIn(EventType.TOOL_ERROR, types)

    def test_tool_error_payload_marks_blocked(self):
        eng = make_engine([{"tool": "ticket_query", "args": {"ticket_id": "T1"}},
                           {"final": "兜底"}])
        errs = [e for e in _drain(eng.run_stream("查个东西"))
                if e["type"] == EventType.TOOL_ERROR]
        self.assertTrue(errs)
        self.assertTrue(errs[0]["data"]["blocked"])
        self.assertIn("ticket_query", errs[0]["data"]["tool"])

    def test_raising_tool_emits_tool_error_not_crash(self):
        # ★ boom 必须在白名单内，否则走的是「越权拦截」分支而不是「工具异常」分支
        eng = make_engine([{"tool": "boom", "args": {"x": "1"}},
                           {"final": "兜底"}],
                          route_table={"full_review": ["kb_search",
                                                       "ticket_query", "boom"]})
        evs = _drain(eng.run_stream("工单怎么样"))
        errs = [e for e in evs if e["type"] == EventType.TOOL_ERROR]
        self.assertTrue(errs)
        self.assertFalse(errs[0]["data"]["blocked"],
                         "白名单内工具抛异常不该被标成 blocked")
        self.assertIn("RuntimeError", errs[0]["data"]["message"])

    def test_max_rounds_emits_limit(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}}],
                          max_rounds=1)
        limits = [e for e in _drain(eng.run_stream("工单怎么样"))
                  if e["type"] == EventType.LIMIT]
        self.assertTrue(limits)
        self.assertIn(limits[0]["data"]["reason"],
                      ("max_rounds", "loop_detected", "max_tool_calls"))


class TestStreamTimestamps(unittest.TestCase):
    """I4：面板画时间线依赖 ts。"""

    def test_every_event_has_ts(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "done"}])
        for ev in eng.run_stream("查一下"):
            self.assertIn("ts", ev)
            self.assertIsInstance(ev["ts"], float)

    def test_ts_monotonic(self):
        ticks = {"t": 1000.0}

        def clock():
            ticks["t"] += 0.001
            return ticks["t"]

        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "done"}],
                          clock=clock)
        ts = [e["ts"] for e in eng.run_stream("查一下")]
        self.assertEqual(ts, sorted(ts))
        self.assertTrue(len(ts) > 1)


class TestStreamFinalState(unittest.TestCase):
    """I5：HTTP 层推完事件后还要拿最终答案。"""

    def test_last_state_available_after_drain(self):
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "最终答案"}])
        _drain(eng.run_stream("查一下"))
        self.assertIsNotNone(eng.last_state)
        self.assertEqual(eng.last_state.answer, "最终答案")
        self.assertTrue(eng.last_state.finished)

    def test_last_state_matches_run(self):
        decisions = [{"tool": "kb_search", "args": {"query": "x"}},
                     {"final": "最终答案"}]
        a = make_engine(list(decisions))
        b = make_engine(list(decisions))
        _drain(a.run_stream("查一下", thread_id="T-FIX"))
        r = b.run("查一下", thread_id="T-FIX")
        self.assertEqual(a.last_state.answer, r.state.answer)
        self.assertEqual(a.last_state.rounds, r.state.rounds)
        self.assertEqual(a.last_state.tool_calls, r.state.tool_calls)


class TestStreamResume(unittest.TestCase):
    """续跑路径也要能流式输出。"""

    def test_resume_stream_emits_resume_event(self):
        from core.agent import InMemoryCheckpointer
        cp = InMemoryCheckpointer()
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "最终答案"}],
                          checkpointer=cp)
        _drain(eng.run_stream("查一下", thread_id="T-1"))
        eng2 = make_engine([{"final": "续跑答案"}], checkpointer=cp)
        types = _types(_drain(eng2.run_stream("查一下", thread_id="T-1",
                                              resume=True)))
        self.assertEqual(types[0], EventType.START)
        self.assertIn(EventType.RESUME, types)


# ================================================================== 轨迹落盘
class TestEventLog(unittest.TestCase):
    """M1-2：轨迹要能落盘、能回放，否则"事后定位错误节点"无从谈起。"""

    def _json_cp(self):
        import tempfile
        from core.agent import JsonFileCheckpointer
        d = tempfile.mkdtemp(prefix="cp_")
        return JsonFileCheckpointer(path=d), d

    def test_json_append_and_load(self):
        cp, _ = self._json_cp()
        cp.append_events("T-1", [{"type": "start", "ts": 1.0},
                                 {"type": "plan", "ts": 1.1}])
        cp.append_events("T-1", [{"type": "done", "ts": 1.2}])
        evs = cp.load_events("T-1")
        self.assertEqual([e["type"] for e in evs], ["start", "plan", "done"])

    def test_json_reset_and_isolation(self):
        cp, _ = self._json_cp()
        cp.append_events("T-1", [{"type": "start"}])
        cp.append_events("T-2", [{"type": "start"}])
        cp.reset_events("T-1")
        self.assertEqual(cp.load_events("T-1"), [])
        self.assertEqual(len(cp.load_events("T-2")), 1)

    def test_json_missing_thread_returns_empty(self):
        cp, _ = self._json_cp()
        self.assertEqual(cp.load_events("不存在"), [])

    def test_json_survives_truncated_line(self):
        """进程被杀留下的半截行不能让整个回放失败。"""
        cp, d = self._json_cp()
        cp.append_events("T-1", [{"type": "start"}])
        with open(f"{d}/T-1.events.jsonl", "a", encoding="utf-8") as f:
            f.write('{"type": "plan", "ts"')  # 半截
        evs = cp.load_events("T-1")
        self.assertEqual([e["type"] for e in evs], ["start"])

    def test_json_events_file_not_counted_as_thread(self):
        """轨迹文件不能被 list_threads 当成线程。"""
        cp, _ = self._json_cp()
        cp.append_events("T-1", [{"type": "start"}])
        self.assertEqual(cp.list_threads(), [])  # 只有轨迹、没有状态

    def test_inmemory_event_log(self):
        from core.agent import InMemoryCheckpointer
        cp = InMemoryCheckpointer()
        cp.append_events("T-1", [{"type": "a"}, {"type": "b"}])
        self.assertEqual([e["type"] for e in cp.load_events("T-1")],
                         ["a", "b"])
        cp.reset_events("T-1")
        self.assertEqual(cp.load_events("T-1"), [])

    def test_engine_writes_events_to_checkpointer(self):
        from core.agent import InMemoryCheckpointer
        cp = InMemoryCheckpointer()
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "答案"}],
                          checkpointer=cp)
        streamed = _drain(eng.run_stream("查一下", thread_id="T-9"))
        stored = cp.load_events("T-9")
        self.assertEqual(_fingerprint(stored), _fingerprint(streamed))

    def test_rerun_resets_trajectory(self):
        from core.agent import InMemoryCheckpointer
        cp = InMemoryCheckpointer()
        eng = make_engine([{"final": "第一次"}], checkpointer=cp)
        _drain(eng.run_stream("查一下", thread_id="T-9"))
        n1 = len(cp.load_events("T-9"))
        _drain(eng.run_stream("查一下", thread_id="T-9"))
        n2 = len(cp.load_events("T-9"))
        self.assertEqual(n1, n2, "重跑应覆盖旧轨迹而不是累加")

    def test_resume_appends_instead_of_resetting(self):
        """续跑的轨迹必须接在首次之后 —— 回放才看得全。"""
        from core.agent import InMemoryCheckpointer
        cp = InMemoryCheckpointer()
        decisions1 = [{"tool": "kb_search", "args": {"query": "x"}},
                      {"final": "最终答案"}]
        eng = make_engine(decisions1, checkpointer=cp)
        _drain(eng.run_stream("查一下", thread_id="T-9"))
        n1 = len(cp.load_events("T-9"))
        eng2 = make_engine([{"final": "续跑答案"}], checkpointer=cp)
        types = _types(_drain(
            eng2.run_stream("查一下", thread_id="T-9", resume=True)))
        n2 = len(cp.load_events("T-9"))
        self.assertGreaterEqual(n2, n1)
        self.assertIn(EventType.RESUME, types)


# ================================================================== 真库
try:  # pragma: no cover - 有 langgraph 才跑
    import langgraph  # noqa: F401
    _HAS_LANGGRAPH = True
except Exception:  # noqa: BLE001
    _HAS_LANGGRAPH = False


@unittest.skipUnless(_HAS_LANGGRAPH, "需要真 langgraph")
class TestStreamRealLangGraph(unittest.TestCase):
    """真库下同样必须成立（Fake 替身可能掩盖 API 差异）。"""

    def test_real_graph_stream_parity(self):
        decisions = [{"tool": "kb_search", "args": {"query": "精度"}},
                     {"final": "答案是 A"}]
        a = make_engine(list(decisions), real_langgraph=True)
        b = make_engine(list(decisions), real_langgraph=True)
        streamed = _drain(a.run_stream("查精度", thread_id="T-FIX"))
        finished = b.run("查精度", thread_id="T-FIX")
        self.assertEqual(_types(streamed), _types(finished.events))
        self.assertTrue(len(streamed) > 3)

    def test_real_graph_incremental(self):
        log = []
        eng = make_engine([{"tool": "kb_search", "args": {"query": "x"}},
                           {"final": "done"}],
                          real_langgraph=True, registry_log=log)
        gen = eng.run_stream("查一下")
        first = next(gen)
        self.assertEqual(first["type"], EventType.START)
        self.assertEqual(log, [])


if __name__ == "__main__":
    unittest.main()
