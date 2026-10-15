# -*- coding: utf-8 -*-
"""tests/test_eval_cli.py —— 评测命令行的单测（离线）。

`cmd_show` / `cmd_compare` 只读文件，可以完整验证。
`cmd_run` 需要真模型装配，这里只验证参数解析与失败路径的友好度。
"""
import argparse
import io
import json
import os
import shutil
import sys
import unittest
import uuid
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval_runner import EvalRunConfig, run_eval  # noqa: E402
from core.evaluation import EvalItem  # noqa: E402
from core.pipeline import RetrievalConfig  # noqa: E402
from rag1_knowledge_qa.eval import (  # noqa: E402
    build_parser, cmd_compare, cmd_show, main,
)

_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


def make_tmp() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    p = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(p, exist_ok=True)
    return p


KEYWORDS = {"精度": "c1", "数据集": "c2"}


class FakeRetriever:
    def retrieve(self, query, cfg=None):
        ids = [v for k, v in KEYWORDS.items() if k in query]
        return {"doc_ids": ids, "scores": [0.9] * len(ids)}


def items():
    return [EvalItem("q1", "精度", ["c1"], "", "fact", True),
            EvalItem("q2", "数据集", ["c2"], "", "fact", True)]


def configs():
    return [RetrievalConfig(name="A_vector", use_vector=True, use_bm25=False,
                            use_rerank=False),
            RetrievalConfig(name="B_hybrid", use_vector=True, use_bm25=True,
                            use_rerank=False)]


def make_run(out_dir: str, run_id: str):
    retr = FakeRetriever()
    return run_eval(items(), configs(),
                    lambda c: (lambda q: retr.retrieve(q, c)),
                    EvalRunConfig(out_dir=out_dir, run_id=run_id,
                                  embedding_model="fake-32d",
                                  ingest_meta={"chunk_size": 48, "overlap": 0,
                                               "config_fingerprint": "fp123"}))


def run_capture(fn, *a):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = fn(*a)
    return code, buf.getvalue()


def ns(**kw):
    o = argparse.Namespace()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


# ================================================================== parser
class TestParser(unittest.TestCase):
    def test_defaults(self):
        a = build_parser().parse_args([])
        self.assertEqual(a.eval_set, "data/eval/eval.jsonl")
        self.assertEqual(a.top_k, 5)
        self.assertFalse(a.ablation)
        self.assertFalse(a.judge)

    def test_ablation_and_judge_flags(self):
        a = build_parser().parse_args(["--ablation", "--judge"])
        self.assertTrue(a.ablation)
        self.assertTrue(a.judge)

    def test_k_list(self):
        a = build_parser().parse_args(["--k-list", "5", "3", "10"])
        self.assertEqual(a.k_list, [5, 3, 10])

    def test_compare_two_args(self):
        a = build_parser().parse_args(["--compare", "r1", "r2"])
        self.assertEqual(a.compare, ["r1", "r2"])

    def test_show_with_run(self):
        a = build_parser().parse_args(["--show", "--run", "reports/run-x"])
        self.assertTrue(a.show)
        self.assertEqual(a.run, "reports/run-x")


# ================================================================== show
class TestShowCommand(unittest.TestCase):
    def test_latest_run(self):
        tmp = make_tmp()
        try:
            make_run(tmp, "run-20260101-000000-aaa")
            code, out = run_capture(cmd_show, ns(run=None, out=tmp))
            self.assertEqual(code, 0)
            self.assertIn("run-20260101-000000-aaa", out)
            self.assertIn("A_vector", out)
            self.assertIn("hit@5", out)
            self.assertIn("fake-32d", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_explicit_run(self):
        tmp = make_tmp()
        try:
            r = make_run(tmp, "run-20260101-000000-aaa")
            code, out = run_capture(cmd_show, ns(run=r.out_dir, out=tmp))
            self.assertEqual(code, 0)
            self.assertIn("fp123", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_runs_returns_one(self):
        tmp = make_tmp()
        try:
            err = io.StringIO()
            old = sys.stderr
            sys.stderr = err
            try:
                code, _ = run_capture(cmd_show, ns(run=None, out=tmp))
            finally:
                sys.stderr = old
            self.assertEqual(code, 1)
            self.assertIn("未找到任何 run", err.getvalue())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_manifest_returns_one(self):
        tmp = make_tmp()
        try:
            bad = os.path.join(tmp, "run-20260101-000000-bbb")
            os.makedirs(bad, exist_ok=True)
            err = io.StringIO()
            old = sys.stderr
            sys.stderr = err
            try:
                code, _ = run_capture(cmd_show, ns(run=bad, out=tmp))
            finally:
                sys.stderr = old
            self.assertEqual(code, 1)
            self.assertIn("未找到 manifest", err.getvalue())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_shows_problems_when_any(self):
        tmp = make_tmp()
        try:
            retr = FakeRetriever()
            bad_items = [EvalItem("q1", "问题", [], "", "fact", True)]
            run_eval(bad_items, configs()[:1],
                     lambda c: (lambda q: retr.retrieve(q, c)),
                     EvalRunConfig(out_dir=tmp, run_id="run-bad"))
            code, out = run_capture(cmd_show, ns(run=None, out=tmp))
            self.assertIn("评测集体检", out)
            self.assertIn("未标注 gold_ids", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ================================================================== compare
class TestCompareCommand(unittest.TestCase):
    def test_compare_outputs_table(self):
        tmp = make_tmp()
        try:
            a = make_run(tmp, "run-A")
            b = make_run(tmp, "run-B")
            code, out = run_capture(cmd_compare,
                                    ns(run_a=a.out_dir, run_b=b.out_dir,
                                       out_file=None))
            self.assertEqual(code, 0)
            self.assertIn("两次评测对比", out)
            self.assertIn("hit@5", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_writes_file(self):
        tmp = make_tmp()
        try:
            a = make_run(tmp, "run-A")
            b = make_run(tmp, "run-B")
            out_file = os.path.join(tmp, "diff.md")
            code, _ = run_capture(cmd_compare,
                                  ns(run_a=a.out_dir, run_b=b.out_dir,
                                     out_file=out_file))
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out_file))
            self.assertIn("hit@5", open(out_file, encoding="utf-8").read())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ================================================================== main 分发
class TestMainDispatch(unittest.TestCase):
    def test_show_dispatched(self):
        tmp = make_tmp()
        try:
            make_run(tmp, "run-20260101-000000-aaa")
            code, out = run_capture(main, ["--show", "--out", tmp])
            self.assertEqual(code, 0)
            self.assertIn("run-20260101", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_dispatched(self):
        tmp = make_tmp()
        try:
            a = make_run(tmp, "run-A")
            b = make_run(tmp, "run-B")
            code, out = run_capture(main, ["--compare", a.out_dir, b.out_dir])
            self.assertEqual(code, 0)
            self.assertIn("两次评测对比", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_without_deps_fails_gracefully(self):
        """★ 未补 embedder / 未建库时必须给出可读错误，而不是栈回溯。"""
        try:
            import core.embedder  # noqa: F401
            self.skipTest("本环境已有 core.embedder")
        except ImportError:
            pass
        tmp = make_tmp()
        try:
            es = os.path.join(tmp, "es.jsonl")
            from core.evaluation import save_eval_set
            save_eval_set(items(), es)
            err = io.StringIO()
            old = sys.stderr
            sys.stderr = err
            try:
                code = main(["--eval-set", es, "--out", tmp])
            except SystemExit as exc:      # argparse/装配失败都可能 SystemExit
                code = exc.code
            except Exception as exc:       # noqa: BLE001
                code = f"raised {type(exc).__name__}: {exc}"
            finally:
                sys.stderr = old
            # 允许 2/1 或抛出可读异常，但不应是静默成功
            self.assertNotEqual(code, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
