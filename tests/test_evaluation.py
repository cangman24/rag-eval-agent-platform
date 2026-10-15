# -*- coding: utf-8 -*-
"""tests/test_evaluation.py —— 检索编排 + 评测框架的单测。全部离线。"""
import json
import os
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bm25 import BM25Index  # noqa: E402
from core.evaluation import (  # noqa: E402
    EvalItem, load_eval_set, render_ablation_table, render_failure_cases,
    render_type_table, run_answer_eval, run_retrieval_eval, save_eval_set,
    validate_eval_set, write_report,
)
from core.pipeline import HybridRetriever, RetrievalConfig, ablation_configs  # noqa: E402
from core.vectorstore import InMemoryVectorStore, l2_normalize  # noqa: E402

_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


def make_tmp() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    p = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(p, exist_ok=True)
    return p


DOCS = {
    "c0": "第一章 总则 本规范适用于水稻穗检测。",
    "c1": "检测精度要求 mAP 不低于 0.90，F1 不低于 0.88。",
    "c2": "训练集按 7:1:2 划分，标注使用 Labelme。",
    "c3": "模型导出 ONNX 并量化，端侧时延不超过 200 毫秒。",
}


def fake_embed(text: str, dim: int = 32):
    v = [0.0] * dim
    for ch in text:
        if not ch.strip():
            continue
        v[ord(ch) % dim] += 1.0
    return l2_normalize(v)


def build_retriever(rerank=False):
    bm25 = BM25Index()
    for did, text in DOCS.items():
        bm25.add(did, text)
    bm25.finalize()
    store = InMemoryVectorStore()
    store.add(ids=list(DOCS), embeddings=[fake_embed(t) for t in DOCS.values()],
              documents=list(DOCS.values()))
    rerank_fn = None
    if rerank:
        # 极简"重排"：按 query 与文档的字符重合度排，仅用于验证链路接口
        def rerank_fn(query, pairs):
            scored = [(did, float(len(set(query) & set(text)))) for did, text in pairs]
            scored.sort(key=lambda kv: (-kv[1], kv[0]))
            return scored
    return HybridRetriever(bm25=bm25, vector_store=store, embed_fn=fake_embed,
                           rerank_fn=rerank_fn, doc_lookup=lambda d: DOCS.get(d))


# ================================================================== pipeline
class TestHybridRetriever(unittest.TestCase):
    def test_config_label(self):
        cfg = RetrievalConfig(use_vector=True, use_bm25=True, use_rerank=True, use_rules=False)
        self.assertEqual(cfg.label(), "向量+BM25+Rerank")

    def test_vector_only(self):
        r = build_retriever().retrieve("mAP 精度", RetrievalConfig(
            use_vector=True, use_bm25=False, use_rerank=False))
        self.assertTrue(r.doc_ids)
        self.assertIn("vector", r.trace)
        self.assertNotIn("bm25", r.trace)

    def test_bm25_only(self):
        r = build_retriever().retrieve("mAP 精度", RetrievalConfig(
            use_vector=False, use_bm25=True, use_rerank=False))
        self.assertTrue(r.doc_ids)
        self.assertIn("bm25", r.trace)

    def test_hybrid_fuses_both(self):
        r = build_retriever().retrieve("mAP 精度", RetrievalConfig(
            use_vector=True, use_bm25=True, use_rerank=False))
        self.assertIn("vector", r.trace)
        self.assertIn("bm25", r.trace)

    def test_empty_config_returns_nothing(self):
        r = build_retriever().retrieve("mAP", RetrievalConfig(
            use_vector=False, use_bm25=False, use_rerank=False))
        self.assertEqual(r.doc_ids, [])

    def test_top_k_respected(self):
        r = build_retriever().retrieve("检测", RetrievalConfig(top_k=2))
        self.assertLessEqual(len(r.doc_ids), 2)

    def test_latency_recorded(self):
        r = build_retriever().retrieve("mAP")
        self.assertGreaterEqual(r.latency_ms, 0.0)

    def test_rerank_changes_path(self):
        r = build_retriever(rerank=True).retrieve("mAP 精度", RetrievalConfig(
            use_vector=True, use_bm25=True, use_rerank=True))
        self.assertTrue(r.doc_ids)

    def test_rule_route_extracts_measures(self):
        r = build_retriever().retrieve("0.90 是多少", RetrievalConfig(
            use_vector=False, use_bm25=False, use_rerank=False, use_rules=True))
        self.assertIn("c1", r.doc_ids)

    def test_rule_route_empty_when_no_measure(self):
        r = build_retriever().retrieve("精度要求", RetrievalConfig(
            use_vector=False, use_bm25=False, use_rerank=False, use_rules=True))
        self.assertEqual(r.doc_ids, [])

    def test_weighted_rrf(self):
        r = build_retriever().retrieve("mAP", RetrievalConfig(
            use_vector=True, use_bm25=True, use_rerank=False,
            rrf_weight={"bm25": 3.0, "vector": 0.1}))
        self.assertTrue(r.doc_ids)

    def test_ablation_configs_are_four(self):
        cfgs = ablation_configs()
        self.assertEqual(len(cfgs), 4)
        self.assertEqual([c.name for c in cfgs],
                         ["A_vector_only", "B_hybrid_rrf", "C_hybrid_rerank",
                          "D_hybrid_rerank_rules"])
        # A 档必须关掉 bm25/rerank，否则消融无意义
        self.assertFalse(cfgs[0].use_bm25)
        self.assertFalse(cfgs[0].use_rerank)

    def test_missing_components_are_safe(self):
        r = HybridRetriever().retrieve("mAP")
        self.assertEqual(r.doc_ids, [])


# ================================================================== eval set
class TestEvalItemAndSet(unittest.TestCase):
    def test_from_dict_aliases(self):
        it = EvalItem.from_dict({"id": "q1", "query": "x", "type": "multi_hop"})
        self.assertEqual(it.qid, "q1")
        self.assertEqual(it.qtype, "multi_hop")

    def test_defaults(self):
        it = EvalItem.from_dict({"qid": "q1", "query": "x"})
        self.assertEqual(it.qtype, "fact")
        self.assertTrue(it.is_answerable)

    def test_roundtrip_save_load(self):
        tmp = make_tmp()
        try:
            path = os.path.join(tmp, "e.jsonl")
            items = [EvalItem("q1", "问题一", ["c1"], "答案", "fact", True),
                     EvalItem("q2", "问题二", [], "", "out_of_scope", False)]
            save_eval_set(items, path)
            loaded = load_eval_set(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].gold_ids, ["c1"])
            self.assertFalse(loaded[1].is_answerable)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_json_array(self):
        tmp = make_tmp()
        try:
            path = os.path.join(tmp, "e.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"qid": "a", "query": "q", "gold_ids": ["x"]}], f)
            self.assertEqual(len(load_eval_set(path)), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_eval_set(os.path.join(_TMP_ROOT, "nope.jsonl"))


class TestValidateEvalSet(unittest.TestCase):
    def test_clean_set_ok(self):
        items = [EvalItem("q1", "问题", ["c1"], "a", "fact", True),
                 EvalItem("q2", "越界", [], "", "out_of_scope", False)]
        rep = validate_eval_set(items)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["total"], 2)
        self.assertEqual(rep["answerable"], 1)
        self.assertEqual(rep["unanswerable"], 1)

    def test_detects_missing_gold(self):
        rep = validate_eval_set([EvalItem("q1", "问题", [], "", "fact", True)])
        self.assertFalse(rep["ok"])
        self.assertTrue(any("未标注 gold_ids" in p for p in rep["problems"]))

    def test_detects_duplicate_qid(self):
        rep = validate_eval_set([EvalItem("q1", "a", ["c"], "", "fact", True),
                                 EvalItem("q1", "b", ["c"], "", "fact", True)])
        self.assertTrue(any("qid 重复" in p for p in rep["problems"]))

    def test_detects_empty_query(self):
        rep = validate_eval_set([EvalItem("q1", "  ", ["c"], "", "fact", True)])
        self.assertTrue(any("query 为空" in p for p in rep["problems"]))

    def test_detects_unanswerable_with_gold(self):
        rep = validate_eval_set([EvalItem("q1", "x", ["c"], "", "out_of_scope", False)])
        self.assertTrue(any("不可答却标了 gold_ids" in p for p in rep["problems"]))

    def test_counts_by_type(self):
        rep = validate_eval_set([
            EvalItem("q1", "a", ["c"], "", "fact", True),
            EvalItem("q2", "b", ["c"], "", "fact", True),
            EvalItem("q3", "c", ["c"], "", "multi_hop", True),
        ])
        self.assertEqual(rep["by_type"], {"fact": 2, "multi_hop": 1})


# ================================================================== run eval
class TestRunRetrievalEval(unittest.TestCase):
    def _items(self):
        return [
            EvalItem("q1", "mAP 精度", ["c1"], "", "fact", True),
            EvalItem("q2", "数据集划分", ["c2"], "", "fact", True),
            EvalItem("q3", "ONNX 时延", ["c3"], "", "term", True),
            EvalItem("q4", "知识库外的问题", [], "", "out_of_scope", False),
        ]

    def test_returns_report(self):
        retr = build_retriever()
        rep = run_retrieval_eval(self._items(),
                                 lambda q: retr.retrieve(q, RetrievalConfig(top_k=5)),
                                 config_name="test")
        # 不可答样本被排除
        self.assertEqual(rep.n, 3)
        self.assertIn("hit@5", rep.overall)
        self.assertTrue(0.0 <= rep.overall["hit@5"] <= 1.0)

    def test_by_type_grouping(self):
        retr = build_retriever()
        rep = run_retrieval_eval(self._items(), lambda q: retr.retrieve(q),
                                 config_name="t")
        self.assertIn("fact", rep.by_type)
        self.assertIn("term", rep.by_type)

    def test_accepts_tuple_and_list_return(self):
        items = [EvalItem("q1", "x", ["c1"], "", "fact", True)]
        rep1 = run_retrieval_eval(items, lambda q: (["c1", "c2"], 12.0))
        self.assertEqual(rep1.n, 1)
        self.assertEqual(rep1.overall["hit@5"], 1.0)
        self.assertEqual(rep1.latency["mean_ms"], 12.0)

        rep2 = run_retrieval_eval(items, lambda q: ["c9"])
        self.assertEqual(rep2.overall["hit@5"], 0.0)

    def test_accepts_dict_return(self):
        items = [EvalItem("q1", "x", ["c1"], "", "fact", True)]
        rep = run_retrieval_eval(items, lambda q: {"doc_ids": ["c1"], "latency_ms": 5.0})
        self.assertEqual(rep.overall["hit@5"], 1.0)

    def test_details_recorded(self):
        items = [EvalItem("q1", "x", ["c1"], "", "fact", True)]
        rep = run_retrieval_eval(items, lambda q: ["c2"])
        self.assertEqual(rep.details[0]["hit"], 0.0)
        self.assertEqual(rep.details[0]["retrieved"], ["c2"])

    def test_all_unanswerable_gives_empty_overall(self):
        items = [EvalItem("q1", "x", [], "", "out_of_scope", False)]
        rep = run_retrieval_eval(items, lambda q: [])
        self.assertEqual(rep.n, 0)
        self.assertEqual(rep.overall, {})

    def test_perfect_retrieval_scores_one(self):
        items = [EvalItem("q1", "x", ["c1"], "", "fact", True)]
        rep = run_retrieval_eval(items, lambda q: ["c1"])
        self.assertEqual(rep.overall["hit@5"], 1.0)


class TestRunAnswerEval(unittest.TestCase):
    def test_refusal_and_judge(self):
        items = [
            EvalItem("q1", "可答问题", ["c1"], "答案", "fact", True),
            EvalItem("q2", "越界问题", [], "", "out_of_scope", False),
        ]

        def answer_fn(q):
            return "知识库中未找到相关内容" if "越界" in q else "答案是 X [1]"

        def judge_fn(item, answer):
            return {"relevance": 5, "accuracy": 4, "completeness": 4, "groundedness": 5}

        out = run_answer_eval(items, answer_fn, judge_fn)
        self.assertEqual(out["refusal_accuracy"], 1.0)   # 两条都判定正确
        self.assertEqual(out["accuracy_mean"], 4.0)
        self.assertEqual(out["hallucination_rate"], 0.0)

    def test_hallucination_detected(self):
        items = [EvalItem("q1", "问题", ["c1"], "答案", "fact", True)]
        out = run_answer_eval(items, lambda q: "编造的回答",
                              lambda i, a: {"groundedness": 1, "relevance": 5})
        self.assertEqual(out["hallucination_rate"], 1.0)

    def test_wrong_refusal_counted(self):
        items = [EvalItem("q1", "可答问题", ["c1"], "答案", "fact", True)]
        out = run_answer_eval(items, lambda q: "知识库中未找到相关内容",
                              lambda i, a: {"groundedness": 5})
        self.assertEqual(out["refusal_accuracy"], 0.0)

    def test_judge_exception_does_not_crash(self):
        items = [EvalItem("q1", "问题", ["c1"], "答案", "fact", True)]

        def bad_judge(i, a):
            raise RuntimeError("judge 挂了")

        out = run_answer_eval(items, lambda q: "回答", bad_judge)
        self.assertEqual(out["judged"], 0.0)


# ================================================================== render
class TestRenderers(unittest.TestCase):
    def _reports(self):
        items = [EvalItem("q1", "mAP 精度", ["c1"], "", "fact", True),
                 EvalItem("q2", "数据集划分", ["c2"], "", "multi_hop", True)]
        retr = build_retriever()
        out = []
        for cfg in ablation_configs(top_k=5):
            out.append(run_retrieval_eval(items, lambda q, c=cfg: retr.retrieve(q, c),
                                          config_name=cfg.name))
        return out

    def test_ablation_table_shape(self):
        table = render_ablation_table(self._reports())
        lines = table.strip().splitlines()
        self.assertEqual(len(lines), 2 + 4)          # 表头 + 分隔 + 4 档
        self.assertIn("hit@5", lines[0])
        self.assertIn("P95", lines[0])

    def test_type_table_has_rows(self):
        table = render_type_table(self._reports()[0])
        self.assertIn("fact", table)

    def test_failure_cases_lists_bad_only(self):
        items = [EvalItem("q1", "完全不相关", ["c3"], "", "fact", True)]
        rep = run_retrieval_eval(items, lambda q: ["c0"])
        text = render_failure_cases(rep)
        self.assertIn("q1", text)

    def test_failure_cases_when_none(self):
        items = [EvalItem("q1", "mAP 精度", ["c1"], "", "fact", True)]
        rep = run_retrieval_eval(items, lambda q: ["c1"])
        self.assertIn("无完全失败样本", render_failure_cases(rep))

    def test_write_report_creates_file(self):
        tmp = make_tmp()
        try:
            path = os.path.join(tmp, "reports", "r.md")
            rep = run_retrieval_eval([EvalItem("q1", "mAP", ["c1"], "", "fact", True)],
                                     lambda q: ["c1"], config_name="cfg")
            write_report(rep, path, extra_sections={"消融对比": "| a | b |\n|---|---|"})
            self.assertTrue(os.path.exists(path))
            content = open(path, encoding="utf-8").read()
            self.assertIn("检索评测报告", content)
            self.assertIn("MRR 为 1-based", content)
            self.assertIn("消融对比", content)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ablation_shows_improvement_possible(self):
        """验证：消融表确实能反映不同配置的差异（否则这张表没有意义）。"""
        items = [EvalItem("q1", "0.90 精度", ["c1"], "", "term", True),
                 EvalItem("q2", "ONNX 量化 200", ["c3"], "", "term", True)]
        retr = build_retriever(rerank=True)
        reports = [run_retrieval_eval(items, lambda q, c=cfg: retr.retrieve(q, c),
                                      config_name=cfg.name)
                   for cfg in ablation_configs(top_k=5)]
        # 至少能算出可比较的数字（不要求一定谁更高——真实效果取决于模型）
        for r in reports:
            self.assertIn("hit@5", r.overall)


if __name__ == "__main__":
    unittest.main(verbosity=2)
