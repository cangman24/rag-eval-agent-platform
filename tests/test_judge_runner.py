# -*- coding: utf-8 -*-
"""tests/test_judge_runner.py —— LLM-as-Judge + 评测编排单测。全部离线。"""
import os
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval_runner import (  # noqa: E402
    EvalRunConfig, compare_runs, file_sha256, latest_run_dir,
    load_items_or_raise, new_run_id, run_eval,
)
from core.evaluation import EvalItem  # noqa: E402
from core.judge import (  # noqa: E402
    DIMENSION_LABELS, JUDGE_DIMENSIONS, LLMJudge,
    clamp_score, context_from_hits, make_context_provider, parse_judge_output,
    build_judge_prompt,
)
from core.pipeline import RetrievalConfig, ablation_configs  # noqa: E402
from core.text import tokenize  # noqa: E402
from core.vectorstore import l2_normalize  # noqa: E402

_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


def make_tmp() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    p = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(p, exist_ok=True)
    return p


# ================================================================== judge 解析
class TestClampScore(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(clamp_score(4), 4.0)
        self.assertEqual(clamp_score("3"), 3.0)

    def test_clamped_to_range(self):
        self.assertEqual(clamp_score(9), 5.0)
        self.assertEqual(clamp_score(-1), 0.0)

    def test_invalid_returns_zero(self):
        self.assertEqual(clamp_score("abc"), 0.0)
        self.assertEqual(clamp_score(None), 0.0)

    def test_zero_is_invalid(self):
        self.assertEqual(clamp_score(0), 0.0)


class TestParseJudgeOutput(unittest.TestCase):
    def test_clean_json(self):
        out = parse_judge_output('{"relevance":5,"accuracy":4,'
                                 '"completeness":3,"groundedness":5}')
        self.assertEqual(out["relevance"], 5.0)
        self.assertEqual(out["completeness"], 3.0)

    def test_fenced_json(self):
        out = parse_judge_output('```json\n{"relevance":4,"accuracy":4,'
                                 '"completeness":4,"groundedness":4}\n```')
        self.assertEqual(out["accuracy"], 4.0)

    def test_json_with_surrounding_text(self):
        out = parse_judge_output('评分如下：{"relevance":3,"accuracy":3,'
                                 '"completeness":3,"groundedness":3} 完毕')
        self.assertEqual(out["relevance"], 3.0)

    def test_reason_captured(self):
        out = parse_judge_output('{"relevance":5,"accuracy":5,'
                                 '"completeness":5,"groundedness":5,'
                                 '"reason":"完全正确"}')
        self.assertEqual(out["reason"], "完全正确")

    def test_chinese_keys(self):
        out = parse_judge_output('{"相关性":4,"准确性":4,"完整性":3,"忠实度":5}')
        self.assertEqual(out["relevance"], 4.0)
        self.assertEqual(out["groundedness"], 5.0)

    def test_regex_fallback(self):
        out = parse_judge_output("relevance: 4, accuracy: 5, "
                                 "completeness: 3, groundedness: 4")
        self.assertEqual(out["accuracy"], 5.0)

    def test_partial_scores_allowed(self):
        out = parse_judge_output('{"relevance":4}')
        self.assertEqual(list(out), ["relevance"])

    def test_empty_returns_empty(self):
        self.assertEqual(parse_judge_output(""), {})
        self.assertEqual(parse_judge_output(None), {})

    def test_garbage_returns_empty(self):
        self.assertEqual(parse_judge_output("我看不出好坏"), {})

    def test_out_of_range_clamped(self):
        out = parse_judge_output('{"relevance":99,"accuracy":0}')
        self.assertEqual(out["relevance"], 5.0)
        self.assertNotIn("accuracy", out)


class TestBuildJudgePrompt(unittest.TestCase):
    def test_contains_all_materials(self):
        p = build_judge_prompt("问题？", "回答。", "参考答案。", "检索片段。")
        self.assertIn("问题？", p)
        self.assertIn("回答。", p)
        self.assertIn("参考答案。", p)
        self.assertIn("检索片段。", p)

    def test_all_dimensions_listed(self):
        p = build_judge_prompt("q", "a")
        for dim in JUDGE_DIMENSIONS:
            self.assertIn(DIMENSION_LABELS[dim], p)

    def test_missing_reference_placeholder(self):
        self.assertIn("无参考答案", build_judge_prompt("q", "a"))

    def test_context_truncated(self):
        p = build_judge_prompt("q", "a", context="x" * 10000)
        self.assertLess(len(p), 5000)

    def test_json_is_escaped_in_template(self):
        """模板里的 {{ }} 必须被正确转义成单个花括号。"""
        self.assertIn('{"relevance"', build_judge_prompt("q", "a"))


class TestContextFromHits(unittest.TestCase):
    def test_numbered_and_joined(self):
        ctx = context_from_hits([{"text": "甲"}, {"text": "乙"}])
        self.assertIn("[1] 甲", ctx)
        self.assertIn("[2] 乙", ctx)

    def test_respects_max_chars(self):
        ctx = context_from_hits([{"text": "x" * 500} for _ in range(10)],
                                max_chars=600)
        self.assertLessEqual(len(ctx), 700)

    def test_empty(self):
        self.assertEqual(context_from_hits([]), "")
        self.assertEqual(context_from_hits(None), "")

    def test_skips_blank_text(self):
        self.assertEqual(context_from_hits([{"text": "  "}]), "")


# ================================================================== judge 调用
class FakeLLM:
    def __init__(self, text='{"relevance":5,"accuracy":4,"completeness":3,'
                            '"groundedness":5,"reason":"ok"}',
                 fail=False, raise_exc=None):
        self.text = text
        self.fail = fail
        self.raise_exc = raise_exc
        self.calls = []
        self.seen_kwargs = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        self.seen_kwargs.append(kw)
        if self.raise_exc:
            raise self.raise_exc
        class R:
            pass
        r = R()
        r.text = "" if self.fail else self.text
        return r


class TestLLMJudge(unittest.TestCase):
    def test_successful_judge(self):
        j = LLMJudge(FakeLLM())
        res = j.judge_text("q", "a", "ref", "ctx")
        self.assertTrue(res.ok)
        self.assertEqual(res.scores["accuracy"], 4.0)
        self.assertEqual(res.reason, "ok")
        self.assertGreaterEqual(res.latency_ms, 0.0)

    def test_prompt_includes_context_and_reference(self):
        llm = FakeLLM()
        LLMJudge(llm).judge_text("问题", "回答", "参考", "片段")
        prompt = llm.calls[0][-1]["content"]
        self.assertIn("问题", prompt)
        self.assertIn("参考", prompt)
        self.assertIn("片段", prompt)

    def test_minus_one_temperature(self):
        """★ 评委必须低温，否则同一输入重复跑分数会漂。"""
        llm = FakeLLM()
        LLMJudge(llm, temperature=0.0).judge_text("q", "a")
        self.assertEqual(llm.seen_kwargs[0].get("temperature"), 0.0)

    def test_empty_output_recorded_as_error(self):
        j = LLMJudge(FakeLLM(fail=True))
        res = j.judge_text("q", "a")
        self.assertFalse(res.ok)
        self.assertIn("无法解析", res.error)
        self.assertEqual(j.stats["parse_failures"], 1)

    def test_llm_exception_does_not_raise(self):
        """★ 评委挂了不该中断整轮评测，要记成错误继续。"""
        j = LLMJudge(FakeLLM(raise_exc=RuntimeError("503")))
        res = j.judge_text("q", "a")
        self.assertFalse(res.ok)
        self.assertIn("503", res.error)
        self.assertEqual(j.stats["errors"], 1)

    def test_callable_signature_for_run_answer_eval(self):
        j = LLMJudge(FakeLLM())
        item = EvalItem("q1", "问题", ["c1"], "参考", "fact", True)
        scores = j(item, "回答")
        self.assertEqual(scores["relevance"], 5.0)

    def test_context_provider_used(self):
        llm = FakeLLM()
        j = LLMJudge(llm, context_provider=lambda q: "检索到的片段")
        item = EvalItem("q1", "问题", ["c1"], "", "fact", True)
        j(item, "回答")
        self.assertIn("检索到的片段", llm.calls[0][-1]["content"])

    def test_broken_context_provider_tolerated(self):
        def boom(q):
            raise RuntimeError("检索挂了")
        j = LLMJudge(FakeLLM(), context_provider=boom)
        item = EvalItem("q1", "问题", ["c1"], "", "fact", True)
        self.assertTrue(j(item, "回答"))

    def test_report_shape(self):
        j = LLMJudge(FakeLLM())
        j.judge_text("q", "a")
        rep = j.report()
        self.assertEqual(rep["calls"], 1)
        self.assertIn("judge_prompt_version", rep)
        self.assertEqual(rep["dimensions"], list(JUDGE_DIMENSIONS))


class TestMakeContextProvider(unittest.TestCase):
    class Svc:
        def __init__(self):
            self.config = type("C", (), {"top_k": 2})()
        def retrieve(self, q):
            return {"doc_ids": ["c1", "c2"], "scores": [0.9, 0.8]}
        def doc_lookup(self, cid):
            return {"c1": "片段一", "c2": "片段二"}.get(cid, "")

    def test_provider_returns_context(self):
        provider = make_context_provider(self.Svc())
        ctx = provider("任意")
        self.assertIn("[1] 片段一", ctx)
        self.assertIn("[2] 片段二", ctx)

    def test_provider_survives_retrieval_failure(self):
        class Bad(self.Svc):
            def retrieve(self, q):
                raise RuntimeError("挂了")
        self.assertEqual(make_context_provider(Bad())("q"), "")


# ================================================================== runner
DOCS = {
    "c1": "检测精度要求 mAP 不低于 0.90。",
    "c2": "训练集按 7:1:2 划分。",
    "c3": "模型导出 ONNX 量化，时延不超过 200 毫秒。",
}
KEYWORDS = {"精度": "c1", "mAP": "c1", "数据集": "c2", "划分": "c2",
            "ONNX": "c3", "时延": "c3"}


class FakeRetriever:
    def __init__(self):
        self.calls = 0
    def retrieve(self, query, cfg=None):
        self.calls += 1
        ids = []
        for k, v in KEYWORDS.items():
            if k in query and v not in ids:
                ids.append(v)
        return {"doc_ids": ids, "scores": [0.9] * len(ids)}


def make_items():
    return [
        EvalItem("q1", "精度要求", ["c1"], "不低于 0.90", "fact", True),
        EvalItem("q2", "数据集划分", ["c2"], "7:1:2", "fact", True),
        EvalItem("q3", "ONNX 时延", ["c3"], "200 毫秒", "term", True),
        EvalItem("q9", "完全无关的问题", [], "", "out_of_scope", False),
    ]


def make_configs():
    return [
        RetrievalConfig(name="A_vector", use_vector=True, use_bm25=False,
                        use_rerank=False, top_k=5),
        RetrievalConfig(name="B_hybrid", use_vector=True, use_bm25=True,
                        use_rerank=False, top_k=5),
    ]


class TestEvalRunId(unittest.TestCase):
    def test_unique(self):
        self.assertNotEqual(new_run_id(), new_run_id())

    def test_prefix(self):
        self.assertTrue(new_run_id("myrun").startswith("myrun-"))


class TestFileSha(unittest.TestCase):
    def test_stable_and_short(self):
        tmp = make_tmp()
        try:
            p = os.path.join(tmp, "a.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("内容")
            a = file_sha256(p)
            self.assertEqual(a, file_sha256(p))
            self.assertEqual(len(a), 16)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_differs_on_content(self):
        tmp = make_tmp()
        try:
            p1 = os.path.join(tmp, "a.txt")
            p2 = os.path.join(tmp, "b.txt")
            for p, c in ((p1, "甲"), (p2, "乙")):
                with open(p, "w", encoding="utf-8") as f:
                    f.write(c)
            self.assertNotEqual(file_sha256(p1), file_sha256(p2))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestRunEval(unittest.TestCase):
    def _run(self, **kw):
        tmp = make_tmp()
        retr = FakeRetriever()
        cfg = EvalRunConfig(out_dir=tmp, top_k=5, **kw)
        res = run_eval(make_items(), make_configs(),
                       retrieve_fn_factory=lambda c: (lambda q: retr.retrieve(q, c)),
                       config=cfg)
        return tmp, res, retr

    def test_creates_run_dir_and_files(self):
        tmp, res, _ = self._run()
        try:
            self.assertTrue(os.path.isdir(res.out_dir))
            for name in ("SUMMARY.md", "eval_manifest.json",
                         "eval_set.jsonl"):
                self.assertTrue(os.path.exists(os.path.join(res.out_dir, name)),
                                f"缺少 {name}")
            # 每档一份逐档报告
            md = [f for f in os.listdir(res.out_dir) if f.startswith("retrieval_")]
            self.assertEqual(len(md), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_manifest_contents(self):
        tmp, res, _ = self._run(embedding_model="bge-m3",
                               ingest_meta={"chunk_size": 512, "overlap": 64,
                                            "config_fingerprint": "abc123"})
        try:
            m = res.manifest
            self.assertEqual(m["ingest"]["embedding_model"], "bge-m3")
            self.assertEqual(m["ingest"]["config_fingerprint"], "abc123")
            self.assertEqual(m["eval_set"]["n"], 4)
            self.assertEqual(m["eval_set"]["answerable"], 3)
            self.assertEqual(len(m["results"]), 2)
            self.assertEqual(m["retrieval"]["top_k"], 5)
            self.assertIn("by_type", m["eval_set"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ingest_meta_embedding_model_survives_default_config(self):
        """★ 回归：EvalRunConfig 的 embedding_model 默认是 'unknown'，
        不能把 ingest_meta 里真实的模型名冲掉（报告里曾出现 unknown）。"""
        tmp = make_tmp()
        try:
            retr = FakeRetriever()
            res = run_eval(
                make_items(), make_configs()[:1],
                lambda c: (lambda q: retr.retrieve(q, c)),
                EvalRunConfig(out_dir=tmp,
                              ingest_meta={"embedding_model": "bge-m3-real",
                                           "chunk_size": 512}))
            self.assertEqual(res.manifest["ingest"]["embedding_model"],
                             "bge-m3-real")
            self.assertEqual(res.manifest["ingest"]["chunk_size"], 512)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_explicit_embedding_model_wins(self):
        tmp = make_tmp()
        try:
            retr = FakeRetriever()
            res = run_eval(
                make_items(), make_configs()[:1],
                lambda c: (lambda q: retr.retrieve(q, c)),
                EvalRunConfig(out_dir=tmp, embedding_model="explicit-model",
                              ingest_meta={"embedding_model": "from-meta"}))
            self.assertEqual(res.manifest["ingest"]["embedding_model"],
                             "explicit-model")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_eval_set_relative_path_recorded(self):
        """报告里应存相对路径，避免把本机目录结构写进交付物。"""
        tmp = make_tmp()
        try:
            from core.evaluation import save_eval_set
            es = os.path.join(tmp, "es.jsonl")
            save_eval_set(make_items(), es)
            retr = FakeRetriever()
            res = run_eval(make_items(), make_configs()[:1],
                           lambda c: (lambda q: retr.retrieve(q, c)),
                           EvalRunConfig(out_dir=os.path.join(tmp, "out"),
                                         eval_set_path=es))
            rel = res.manifest["eval_set"].get("path_rel")
            self.assertIsNotNone(rel)
            self.assertFalse(os.path.isabs(rel), rel)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_retrieval_metrics_computed(self):
        tmp, res, _ = self._run()
        try:
            self.assertEqual(len(res.retrieval_reports), 2)
            for r in res.retrieval_reports:
                self.assertEqual(r.n, 3)     # 不可答样本被排除
                self.assertIn("hit@5", r.overall)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_eval_set_sha_recorded_when_path_given(self):
        tmp = make_tmp()
        try:
            from core.evaluation import save_eval_set
            es = os.path.join(tmp, "es.jsonl")
            save_eval_set(make_items(), es)
            retr = FakeRetriever()
            res = run_eval(make_items(), make_configs(),
                           lambda c: (lambda q: retr.retrieve(q, c)),
                           EvalRunConfig(out_dir=os.path.join(tmp, "out"),
                                         eval_set_path=es))
            self.assertEqual(res.manifest["eval_set"]["sha256"], file_sha256(es))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_answer_eval_with_judge(self):
        tmp = make_tmp()
        try:
            llm = FakeLLM()
            judge = LLMJudge(llm)
            retr = FakeRetriever()
            res = run_eval(
                make_items(), make_configs(),
                lambda c: (lambda q: retr.retrieve(q, c)),
                EvalRunConfig(out_dir=tmp),
                answer_fn=lambda q: ("知识库中未找到相关内容。" if "无关" in q
                                     else "这是回答 [1]"),
                judge_fn=judge,
                judge_report=judge.report,
            )
            am = res.answer_metrics
            self.assertIsNotNone(am)
            self.assertEqual(am["refusal_accuracy"], 1.0)
            self.assertEqual(am["accuracy_mean"], 4.0)
            self.assertEqual(am["hallucination_rate"], 0.0)
            self.assertEqual(res.manifest["judge"]["calls"], 3)  # 3 条可答
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_answer_eval_without_judge(self):
        tmp = make_tmp()
        try:
            retr = FakeRetriever()
            res = run_eval(make_items(), make_configs(),
                           lambda c: (lambda q: retr.retrieve(q, c)),
                           EvalRunConfig(out_dir=tmp),
                           answer_fn=lambda q: "回答")
            self.assertIsNotNone(res.answer_metrics)
            self.assertIn("refusal_accuracy", res.answer_metrics)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_answer_fn_skips_generation_eval(self):
        tmp, res, _ = self._run()
        try:
            self.assertIsNone(res.answer_metrics)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_multi_k_tables(self):
        tmp, res, _ = self._run(k_list=(5, 3))
        try:
            summary = open(res.summary_path, encoding="utf-8").read()
            self.assertIn("多 k 值对比", summary)
            self.assertIn("k=3", summary)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_summary_contains_key_sections(self):
        tmp, res, _ = self._run()
        try:
            s = open(res.summary_path, encoding="utf-8").read()
            for key in ("检索消融", "分类型表现", "失败样本", "评测集体检", "复现方式"):
                self.assertIn(key, s)
            self.assertIn("1-based", s)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_progress_callback_called(self):
        tmp = make_tmp()
        try:
            msgs = []
            retr = FakeRetriever()
            run_eval(make_items(), make_configs(),
                     lambda c: (lambda q: retr.retrieve(q, c)),
                     EvalRunConfig(out_dir=tmp), progress=msgs.append)
            self.assertTrue(any("评测集" in m for m in msgs))
            self.assertTrue(any("A_vector" in m for m in msgs))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_judge_input_cached_kwarg_exists(self):
        """LLMJudge 调用时要带 use_cache（便于评测省钱）。"""
        llm = FakeLLM()
        LLMJudge(llm, use_cache=False).judge_text("q", "a")
        self.assertIn("use_cache", llm.seen_kwargs[0])

    def test_problems_reported_in_summary(self):
        tmp = make_tmp()
        try:
            bad = [EvalItem("q1", "问题", [], "", "fact", True)]  # 可答但无 gold
            retr = FakeRetriever()
            res = run_eval(bad, make_configs()[:1],
                           lambda c: (lambda q: retr.retrieve(q, c)),
                           EvalRunConfig(out_dir=tmp))
            s = open(res.summary_path, encoding="utf-8").read()
            self.assertIn("未标注 gold_ids", s)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLatestRunAndCompare(unittest.TestCase):
    def test_latest_run_dir(self):
        tmp = make_tmp()
        try:
            retr = FakeRetriever()
            for _ in range(2):
                run_eval(make_items(), make_configs()[:1],
                         lambda c: (lambda q: retr.retrieve(q, c)),
                         EvalRunConfig(out_dir=tmp))
            latest = latest_run_dir(tmp)
            self.assertIsNotNone(latest)
            self.assertTrue(os.path.basename(latest).startswith("run-"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_latest_on_empty_dir(self):
        tmp = make_tmp()
        try:
            self.assertIsNone(latest_run_dir(tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_runs(self):
        tmp = make_tmp()
        try:
            retr = FakeRetriever()
            cfg = EvalRunConfig(out_dir=tmp, run_id="run-A")
            a = run_eval(make_items(), make_configs()[:1],
                         lambda c: (lambda q: retr.retrieve(q, c)), cfg)
            cfg2 = EvalRunConfig(out_dir=tmp, run_id="run-B")
            b = run_eval(make_items(), make_configs()[:1],
                         lambda c: (lambda q: retr.retrieve(q, c)), cfg2)
            text = compare_runs(a.out_dir, b.out_dir)
            self.assertIn("A（基线）", text)
            self.assertIn("hit@5", text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compare_missing_manifest_raises(self):
        tmp = make_tmp()
        try:
            with self.assertRaises(FileNotFoundError):
                compare_runs(tmp, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLoadItemsOrRaise(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_items_or_raise(os.path.join(_TMP_ROOT, "nope.jsonl"))


class TestEndToEndWithRealIngest(unittest.TestCase):
    """真链路：文档 -> 建库 -> 两档消融 -> 报告目录。"""

    def test_full_eval_run(self):
        from core.ingest import IngestConfig, ingest_dir
        from core.pipeline import HybridRetriever
        from core.evaluation import save_eval_set

        tmp = make_tmp()
        docs_dir = os.path.join(tmp, "docs")
        os.makedirs(docs_dir, exist_ok=True)
        with open(os.path.join(docs_dir, "spec.md"), "w", encoding="utf-8") as f:
            f.write("# 精度要求\n检测精度要求 mAP@0.5 不低于 0.90。\n"
                    "# 数据集\n训练集按 7:1:2 划分。\n"
                    "# 部署\n模型导出 ONNX 量化，端侧时延不超过 200 毫秒。\n")

        def embed(texts):
            out = []
            for t in texts:
                v = [0.0] * 32
                for tok in tokenize(t):
                    v[sum(map(ord, tok)) % 32] += 1.0
                out.append(l2_normalize(v))
            return out

        try:
            res = ingest_dir(docs_dir, embed, config=IngestConfig(
                chunk_size=48, overlap=0, parent_size=150, min_chunk_tokens=1))
            retriever = HybridRetriever(
                bm25=res.bm25, vector_store=res.vector_store,
                embed_fn=lambda q: embed([q])[0],
                doc_lookup=lambda d: res.child_text(d))

            # 用 bm25 定位 gold（演示口径）
            def gold(kw):
                hits = res.bm25.search(kw, top_k=1, min_score=0.0)
                return [hits[0][0]] if hits else []

            items = [
                EvalItem("q1", "mAP 精度", gold("mAP 精度"), "", "fact", True),
                EvalItem("q2", "数据集划分", gold("7:1:2"), "", "fact", True),
                EvalItem("q3", "ONNX 时延", gold("ONNX"), "", "term", True),
            ]
            es = os.path.join(tmp, "es.jsonl")
            save_eval_set(items, es)

            out_dir = os.path.join(tmp, "reports")
            result = run_eval(
                items, ablation_configs(top_k=5),
                lambda c: (lambda q: retriever.retrieve(q, c)),
                EvalRunConfig(out_dir=out_dir, eval_set_path=es,
                              embedding_model="fake-hash-32d",
                              ingest_meta=res.stats),
            )
            self.assertTrue(os.path.exists(result.summary_path))
            m = result.manifest
            self.assertEqual(m["eval_set"]["n"], 3)
            self.assertEqual(m["ingest"]["embedding_model"], "fake-hash-32d")
            # 四档消融都跑出指标
            self.assertEqual(len(m["results"]), 4)
            for r in m["results"]:
                self.assertIn("hit@5", r["overall"])
            # 汇总里应有消融表
            s = open(result.summary_path, encoding="utf-8").read()
            self.assertIn("A_vector_only", s)
            self.assertIn("D_hybrid_rerank_rules", s)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
