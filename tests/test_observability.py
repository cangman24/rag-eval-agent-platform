# -*- coding: utf-8 -*-
"""tests/test_observability.py —— span 树（耗时归因）不变量。

★ 为什么需要这一层：

    轨迹事件（M1）能告诉你"发生了什么、先后顺序是什么"，但回答不了
    "慢在哪、错在哪"。面试官问「你的系统哪里慢」，光看事件流只能猜。
    span 树补的是**耗时归因**：每一次运行长成一棵树，
    根是整次 run，孩子是各个图节点，节点的孩子是它调的工具。

    有了它才能说出这种话：
        「本次 4.3s，其中 tool:kb_search 独占 890ms（20.7%），
          错误节点是 run > node:tools > tool:send_mail（白名单拦截）」

    本文件锁死 6 条不变量：

      O1 span 是**嵌套**的：不显式指定 parent 就挂在当前栈顶
      O2 span 有耗时：t1 - t0，未闭合的算 0 而不是崩
      O3 异常自动标 error，白名单拦截标 blocked，熔断标 limit
      O4 **自身耗时**（self_ms）= 总耗时 - 直接子 span 耗时之和
         （定位瓶颈必须扣掉孩子，否则父节点永远"最慢"，归因就没意义了）
      O5 可序列化 + 重建：落盘 JSONL 后回放出来的树和原来一样
      O6 树能给出错误节点的**完整路径**（run > node:tools > tool:send_mail）

    零依赖、确定性：时钟与 id 生成器都是注入的，不做任何真实等待。
"""
from __future__ import annotations

import unittest

from core.observability import (
    KIND_NODE,
    KIND_RUN,
    KIND_TOOL,
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_LIMIT,
    STATUS_OK,
    Span,
    SpanTracer,
)


class _FakeClock:
    """手动推进的时钟：让耗时完全确定，不依赖真实 sleep。"""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, ms: float) -> None:
        self.t += ms / 1000.0


def _tracer(clock: _FakeClock):
    return SpanTracer(clock=clock)


# ================================================================== O1 嵌套
class TestNesting(unittest.TestCase):
    def test_child_attaches_to_current_span(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:tools", KIND_NODE):
                clk.tick(10)
                with tr.span("tool:kb_search", KIND_TOOL):
                    clk.tick(50)
        spans = tr.spans
        # spans 按开始时间排序（甘特条要的顺序），不是完成顺序
        self.assertEqual([s.name for s in spans],
                         ["run", "node:tools", "tool:kb_search"])
        run, node, tool = spans
        self.assertEqual(tool.parent_id, node.span_id)
        self.assertEqual(node.parent_id, run.span_id)
        self.assertIsNone(run.parent_id)

    def test_explicit_parent_overrides_stack(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        root = tr.start("run", KIND_RUN)
        a = tr.start("node:a", KIND_NODE)
        tr.finish(a)
        b = tr.start("node:b", KIND_NODE, parent_id=root.span_id)
        tr.finish(b)
        tr.finish(root)
        self.assertEqual(a.parent_id, root.span_id)
        self.assertEqual(b.parent_id, root.span_id)

    def test_current_is_top_of_stack(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        self.assertIsNone(tr.current)
        with tr.span("run", KIND_RUN):
            self.assertEqual(tr.current.name, "run")
            with tr.span("node:act", KIND_NODE):
                self.assertEqual(tr.current.name, "node:act")
            self.assertEqual(tr.current.name, "run")
        self.assertIsNone(tr.current)

    def test_root_has_no_parent_when_stack_empty(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        s = tr.start("orphan", KIND_NODE)
        tr.finish(s)
        self.assertIsNone(s.parent_id)


# ================================================================== O2 耗时
class TestDuration(unittest.TestCase):
    def test_duration_ms(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("node:act", KIND_NODE):
            clk.tick(35.5)
        self.assertAlmostEqual(tr.spans[0].duration_ms, 35.5, places=3)

    def test_unclosed_span_is_zero_not_crash(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        tr.start("dangling", KIND_NODE)   # 忘了 finish
        self.assertEqual(tr.spans[0].duration_ms, 0.0)
        self.assertIsNone(tr.spans[0].t1)

    def test_total_ms_spans_first_start_to_last_end(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            clk.tick(100)
            with tr.span("node:a", KIND_NODE):
                clk.tick(200)
            clk.tick(50)
        self.assertAlmostEqual(tr.total_ms(), 350.0, places=3)

    def test_total_ms_zero_when_empty(self):
        self.assertEqual(_tracer(_FakeClock()).total_ms(), 0.0)


# ================================================================== O3 状态
class TestStatus(unittest.TestCase):
    def test_exception_marks_error_and_reraises(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with self.assertRaises(RuntimeError):
            with tr.span("tool:boom", KIND_TOOL):
                clk.tick(5)
                raise RuntimeError("连接超时")
        span = tr.spans[0]
        self.assertEqual(span.status, STATUS_ERROR)
        self.assertIn("连接超时", span.error)
        self.assertIn("RuntimeError", span.error)

    def test_blocked_status(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("tool:send_mail", KIND_TOOL) as sp:
            tr.finish(sp, status=STATUS_BLOCKED, error="不在白名单")
        self.assertEqual(tr.spans[0].status, STATUS_BLOCKED)

    def test_limit_status(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("node:act", KIND_NODE) as sp:
            tr.finish(sp, status=STATUS_LIMIT, error="max_rounds")
        self.assertEqual(tr.spans[0].status, STATUS_LIMIT)

    def test_default_status_ok(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("node:plan", KIND_NODE):
            pass
        self.assertEqual(tr.spans[0].status, STATUS_OK)

    def test_attrs_recorded(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("tool:kb_search", KIND_TOOL, top_k=5) as sp:
            tr.finish(sp, hits=3)
        d = tr.spans[0].to_dict()
        self.assertEqual(d["attrs"]["top_k"], 5)
        self.assertEqual(d["attrs"]["hits"], 3)


# ================================================================== O4 自身耗时
class TestSelfTime(unittest.TestCase):
    def test_self_ms_excludes_children(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            clk.tick(10)
            with tr.span("node:tools", KIND_NODE):
                clk.tick(10)
                with tr.span("tool:kb_search", KIND_TOOL):
                    clk.tick(80)
                clk.tick(10)
            clk.tick(10)
        run, node, tool = tr.spans
        # node:tools 总 100ms，孩子 80ms -> 自身 20ms；run 总 120ms，孩子 100 -> 自身 20
        self.assertAlmostEqual(node.duration_ms, 100.0, places=3)
        self.assertAlmostEqual(tr.self_ms(node), 20.0, places=3)
        self.assertAlmostEqual(tr.self_ms(run), 20.0, places=3)

    def test_bottlenecks_sorted_by_self_time(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:act", KIND_NODE):
                clk.tick(200)
            with tr.span("node:tools", KIND_NODE):
                clk.tick(10)
                with tr.span("tool:kb_search", KIND_TOOL):
                    clk.tick(890)
        top = tr.bottlenecks(2)
        self.assertEqual([b["name"] for b in top],
                         ["tool:kb_search", "node:act"])
        self.assertAlmostEqual(top[0]["self_ms"], 890.0, places=3)

    def test_bottlenecks_never_picks_root(self):
        """根 span 的"自身耗时"几乎为 0，但它总时长最大 ——
        若按总耗时排，瓶颈永远显示成 run，归因就废了。"""
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:act", KIND_NODE):
                clk.tick(50)
        names = [b["name"] for b in tr.bottlenecks(3)]
        self.assertNotIn("run", names)

    def test_pct_of_total(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("tool:a", KIND_TOOL):
                clk.tick(250)
            with tr.span("tool:b", KIND_TOOL):
                clk.tick(750)
        tree = tr.tree()
        kids = tree[0]["children"]
        pct = {k["name"]: k["pct"] for k in kids}
        self.assertAlmostEqual(pct["tool:a"], 25.0, places=1)
        self.assertAlmostEqual(pct["tool:b"], 75.0, places=1)


# ================================================================== O5 序列化
class TestSerialization(unittest.TestCase):
    def _build(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:tools", KIND_NODE):
                clk.tick(5)
                with tr.span("tool:kb_search", KIND_TOOL, top_k=3):
                    clk.tick(40)
        return tr

    def test_roundtrip(self):
        tr = self._build()
        dicts = tr.to_dicts()
        tr2 = SpanTracer(clock=_FakeClock())
        tr2.load_dicts(dicts)
        self.assertEqual([s.name for s in tr2.spans],
                         [s.name for s in tr.spans])
        self.assertEqual(tr2.tree(), tr.tree())

    def test_roundtrip_survives_json(self):
        import json
        tr = self._build()
        raw = [json.loads(json.dumps(d)) for d in tr.to_dicts()]
        tr2 = SpanTracer(clock=_FakeClock())
        tr2.load_dicts(raw)
        self.assertEqual(tr2.total_ms(), tr.total_ms())

    def test_orphan_span_does_not_crash_tree(self):
        """落盘只写了一半（进程被杀）时，孩子找不到父亲 ——
        回放必须能渲染，而不是抛 KeyError。"""
        tr = SpanTracer(clock=_FakeClock())
        tr.load_dicts([
            {"span_id": "s1", "parent_id": "s99", "name": "tool:x",
             "kind": KIND_TOOL, "t0": 1.0, "t1": 2.0, "status": STATUS_OK,
             "error": "", "attrs": {}},
        ])
        tree = tr.tree()
        self.assertEqual(len(tree), 1)
        self.assertEqual(tree[0]["name"], "tool:x")

    def test_missing_keys_defaulted(self):
        tr = SpanTracer(clock=_FakeClock())
        tr.load_dicts([{"span_id": "s1", "name": "n"}])
        s = tr.spans[0]
        self.assertEqual(s.kind, KIND_NODE)
        self.assertEqual(s.status, STATUS_OK)
        self.assertEqual(s.attrs, {})

    def test_span_roundtrip_single(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        s = tr.start("node:x", KIND_NODE, k=1)
        tr.finish(s)
        back = Span.from_dict(s.to_dict())
        self.assertEqual(back.name, s.name)
        self.assertEqual(back.to_dict(), s.to_dict())


# ================================================================== O6 错误定位
class TestErrorPath(unittest.TestCase):
    def _build(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:tools", KIND_NODE):
                with tr.span("tool:send_mail", KIND_TOOL) as sp:
                    tr.finish(sp, status=STATUS_BLOCKED, error="不在白名单")
        return tr

    def test_errors_returns_path(self):
        errs = self._build().errors()
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["path"],
                         "run > node:tools > tool:send_mail")
        self.assertEqual(errs[0]["status"], STATUS_BLOCKED)

    def test_no_errors_returns_empty(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:plan", KIND_NODE):
                pass
        self.assertEqual(tr.errors(), [])

    def test_limit_counted_as_error_node(self):
        """熔断也是"错误节点"：面板要标红，否则看不出为什么提前结束。"""
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:act", KIND_NODE) as sp:
                tr.finish(sp, status=STATUS_LIMIT, error="max_rounds")
        errs = tr.errors()
        self.assertEqual(errs[0]["path"], "run > node:act")

    def test_tree_marks_error_deepest_first(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:tools", KIND_NODE):
                with tr.span("tool:boom", KIND_TOOL) as sp:
                    tr.finish(sp, status=STATUS_ERROR, error="超时")
        tree = tr.tree()
        self.assertTrue(tree[0]["has_error"])
        self.assertTrue(tree[0]["children"][0]["has_error"])


# ================================================================== 其它
class TestSinkAndRender(unittest.TestCase):
    def test_sink_called_on_finish_for_streaming_persist(self):
        """sink 让 span 一结束就落盘（进程被杀也不丢已跑完的部分）。"""
        got = []
        clk = _FakeClock()
        tr = SpanTracer(clock=clk, sink=got.append)
        with tr.span("run", KIND_RUN):
            with tr.span("node:a", KIND_NODE):
                pass
        self.assertEqual([d["name"] for d in got], ["node:a", "run"])

    def test_sink_error_does_not_break_run(self):
        def boom(_d):
            raise OSError("disk full")

        clk = _FakeClock()
        tr = SpanTracer(clock=clk, sink=boom)
        with tr.span("run", KIND_RUN):     # 不该抛
            with tr.span("node:a", KIND_NODE):
                pass
        self.assertEqual(len(tr.spans), 2)

    def test_reset_clears_spans(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            pass
        tr.reset()
        self.assertEqual(tr.spans, [])
        self.assertEqual(tr.total_ms(), 0.0)

    def test_close_all_finishes_unclosed(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            tr.start("node:dangling", KIND_NODE)   # 忘了 finish
        tr.close_all()
        names = {s.name: s.status for s in tr.spans}
        self.assertEqual(names["node:dangling"], STATUS_ERROR)
        self.assertEqual(names["run"], STATUS_OK)

    def test_render_gantt_contains_bars_and_ms(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            with tr.span("node:act", KIND_NODE):
                clk.tick(100)
            with tr.span("tool:kb_search", KIND_TOOL):
                clk.tick(300)
        text = tr.render_gantt(width=40)
        self.assertIn("node:act", text)
        self.assertIn("tool:kb_search", text)
        self.assertIn("300.0ms", text)
        self.assertIn("#", text)

    def test_render_gantt_empty(self):
        self.assertIn("(无 span)", _tracer(_FakeClock()).render_gantt())

    def test_span_ids_unique_and_ordered(self):
        clk = _FakeClock()
        tr = _tracer(clk)
        with tr.span("run", KIND_RUN):
            for i in range(3):
                with tr.span("n%d" % i, KIND_NODE):
                    pass
        ids = [s.span_id for s in tr.spans]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
