# -*- coding: utf-8 -*-
"""tests/test_cli.py —— 命令行 handler 单测。全部离线。

★ 关键设计：`cmd_*` 都接收 service 作为参数，所以在这里注入一个
  内存版的 RagService 就能完整验证 CLI 行为 —— 不需要真模型、不需要
  建库、也不需要 fastapi。这就是"把逻辑与传输解耦"的收益。
"""
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.service import RagService, ServiceConfig  # noqa: E402
from rag1_knowledge_qa.cli import (  # noqa: E402
    build_parser, cmd_ask, cmd_health, cmd_ingest, cmd_retrieve, main,
)
from core.sse import parse_sse  # noqa: E402


DOCS = {"c1": "父块一：检测精度要求 mAP 不低于 0.90。",
        "c2": "父块二：训练集按 7:1:2 划分。"}
METAS = {"c1": {"heading_path": "第二章 > 精度要求"},
         "c2": {"heading_path": "第二章 > 数据集要求"}}


class FakeRetriever:
    def __init__(self, table=None, score_map=None):
        self.table = table or {"精度": ["c1", "c2"], "数据集": ["c2"]}
        self.score_map = score_map or {}

    def retrieve(self, query, cfg=None):
        ids = []
        for key, val in self.table.items():
            if key in query:
                ids.extend(i for i in val if i not in ids)
        return {"doc_ids": ids,
                "scores": [self.score_map.get(i, 0.9) for i in ids]}


class FakeLLM:
    def __init__(self, text="精度要求 0.90 [1]"):
        self.text = text

    def chat(self, messages, **kw):
        class R:
            pass
        r = R()
        r.text = self.text
        return r


def make_service(llm=None, **cfg):
    return RagService(
        retriever=FakeRetriever(),
        doc_lookup=lambda c: DOCS.get(c, ""),
        llm=llm if llm is not None else FakeLLM(),
        meta_lookup=lambda c: METAS.get(c, {}),
        config=ServiceConfig(**cfg),
    )


def run_capture(fn, *a):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = fn(*a)
    return code, buf.getvalue()


def ns(**kw):
    class NS:
        pass
    o = NS()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


# ================================================================== parser
class TestParser(unittest.TestCase):
    def test_parser_builds(self):
        self.assertIsNotNone(build_parser())

    def test_ask_parses(self):
        args = build_parser().parse_args(["ask", "问题内容", "--stream"])
        self.assertEqual(args.command, "ask")
        self.assertEqual(args.query, "问题内容")
        self.assertTrue(args.stream)

    def test_retrieve_defaults(self):
        args = build_parser().parse_args(["retrieve", "q"])
        self.assertEqual(args.top_k, 5)
        self.assertFalse(args.json)

    def test_ingest_defaults(self):
        args = build_parser().parse_args(["ingest"])
        self.assertEqual(args.chunk_size, 512)
        self.assertEqual(args.overlap, 64)
        self.assertEqual(args.docs, "data/raw")

    def test_missing_subcommand_is_error(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])


# ================================================================== handlers
class TestHealthCommand(unittest.TestCase):
    def test_prints_json(self):
        code, out = run_capture(cmd_health, make_service(), ns())
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "0.1.0")


class TestAskCommand(unittest.TestCase):
    def test_plain_output_contains_answer_and_citation(self):
        code, out = run_capture(cmd_ask, make_service(),
                                ns(query="精度", stream=False, json=False))
        self.assertEqual(code, 0)
        self.assertIn("回答:", out)
        self.assertIn("0.90", out)
        self.assertIn("引用:", out)
        self.assertIn("第二章 > 精度要求", out)

    def test_json_output_is_valid(self):
        code, out = run_capture(cmd_ask, make_service(),
                                ns(query="精度", stream=False, json=True))
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertFalse(data["refused"])
        self.assertTrue(data["citations"])

    def test_refusal_flagged_in_output(self):
        svc = make_service(llm=FakeLLM("知识库中未找到相关内容。"))
        code, out = run_capture(cmd_ask, svc,
                                ns(query="精度", stream=False, json=False))
        self.assertIn("拒答", out)

    def test_stream_outputs_valid_sse(self):
        code, out = run_capture(cmd_ask, make_service(),
                                ns(query="精度", stream=True, json=False))
        self.assertEqual(code, 0)
        events = [e for e in parse_sse(out) if "event" in e]
        types = [e["event"] for e in events]
        self.assertEqual(types[0], "start")
        self.assertEqual(types[-1], "done")
        self.assertIn("answer", types)

    def test_error_field_shown(self):
        svc = make_service(llm=None)
        code, out = run_capture(cmd_ask, svc,
                                ns(query="精度", stream=False, json=False))
        self.assertIn("耗时", out)


class TestRetrieveCommand(unittest.TestCase):
    def test_plain_output(self):
        code, out = run_capture(cmd_retrieve, make_service(),
                                ns(query="精度", top_k=5, json=False))
        self.assertEqual(code, 0)
        self.assertIn("命中 2 条", out)
        self.assertIn("score=", out)

    def test_json_output(self):
        code, out = run_capture(cmd_retrieve, make_service(),
                                ns(query="精度", top_k=5, json=True))
        data = json.loads(out)
        self.assertEqual(data["count"], 2)
        self.assertEqual(data["query"], "精度")

    def test_top_k_respected(self):
        code, out = run_capture(cmd_retrieve, make_service(),
                                ns(query="精度", top_k=1, json=True))
        self.assertEqual(json.loads(out)["count"], 1)

    def test_no_hits(self):
        code, out = run_capture(cmd_retrieve, make_service(),
                                ns(query="完全无关话题", top_k=5, json=True))
        self.assertEqual(json.loads(out)["count"], 0)


class TestIngestCommand(unittest.TestCase):
    def test_missing_embedder_gives_actionable_error(self):
        """★ 缺 core/embedder.py 时必须给出可执行的下一步，而不是 stack trace。"""
        try:
            import core.embedder  # noqa: F401
            self.skipTest("本环境已有 core.embedder，跳过缺失依赖分支")
        except ImportError:
            pass
        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            code = cmd_ingest(None, ns(docs="data/raw", model="m",
                                       chunk_size=512, overlap=64,
                                       parent_size=2048,
                                       chunks_out="c.json",
                                       vectors_out="v.json",
                                       manifest_out="m.json"))
        finally:
            sys.stderr = old
        self.assertEqual(code, 2)
        self.assertIn("bootstrap.py", err.getvalue())


class TestMainEntry(unittest.TestCase):
    def test_health_fails_gracefully_without_deps(self):
        """★ 真机依赖缺失时 main() 必须优雅退出（返回 2），而不是抛异常。"""
        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            code = main(["health"])
        finally:
            sys.stderr = old
        # 沙箱里没有 core.embedder -> 装配失败 -> 返回 2
        self.assertIn(code, (0, 2))
        if code == 2:
            self.assertIn("装配服务失败", err.getvalue())

    def test_ingest_returns_two_without_embedder(self):
        try:
            import core.embedder  # noqa: F401
            self.skipTest("本环境已有 core.embedder")
        except ImportError:
            pass
        err = io.StringIO()
        old = sys.stderr
        sys.stderr = err
        try:
            code = main(["ingest", "--docs", "data/raw"])
        finally:
            sys.stderr = old
        self.assertEqual(code, 2)


# ================================================================== 与服务层一致
class TestCliMatchesServiceLayer(unittest.TestCase):
    def test_ask_json_equals_service_answer(self):
        """★ CLI 与 HTTP 接口必须走同一条代码路径（避免两套行为）。"""
        svc = make_service()
        _, out = run_capture(cmd_ask, svc,
                             ns(query="精度", stream=False, json=True))
        cli_result = json.loads(out)
        svc_result = svc.answer("精度")
        # 除 request_id / latency 外应完全一致
        self.assertEqual(cli_result["answer"], svc_result.answer)
        self.assertEqual(cli_result["refused"], svc_result.refused)
        self.assertEqual([c["chunk_id"] for c in cli_result["citations"]],
                         [c.chunk_id for c in svc_result.citations])

    def test_stream_frames_match_service(self):
        svc = make_service()
        _, out = run_capture(cmd_ask, svc,
                             ns(query="精度", stream=True, json=False))
        cli_events = [e["event"] for e in parse_sse(out) if "event" in e]
        svc_events = [e["event"] for e in parse_sse("".join(svc.answer_stream("精度")))
                      if "event" in e]
        self.assertEqual(cli_events, svc_events)


if __name__ == "__main__":
    unittest.main(verbosity=2)
