# -*- coding: utf-8 -*-
"""tests/test_core.py —— 纯标准库单测，零第三方依赖。

运行：python -m unittest discover -s tests -v
或：  python tests/test_core.py

★ 为什么要先测这些：简历上所有检索指标都由 metrics.py 算出。
  如果 MRR 误用 0-based 排名，你后面 7 天的数字全是错的，
  而且面试时手推一遍就露馅。
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import (  # noqa: E402
    aggregate, all_retrieval_metrics, hallucination_rate, hit_at_k,
    latency_stats, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k,
    refusal_correctness,
)
from core.retrieval import (  # noqa: E402
    build_heading_path, count_tokens_approx, dedup_keep_order, fuse_and_cut,
    heading_level, recursive_chunk, rrf_fuse, weighted_fuse,
)


class TestHitAtK(unittest.TestCase):
    def test_hit_within_k(self):
        self.assertEqual(hit_at_k(["a", "b", "c"], ["c"], 3), 1.0)

    def test_miss_because_k_too_small(self):
        # gold 在第 3 位，但只看前 2 条 -> 未命中
        self.assertEqual(hit_at_k(["a", "b", "c"], ["c"], 2), 0.0)

    def test_multi_gold_hits_any(self):
        self.assertEqual(hit_at_k(["x", "b"], ["b", "c"], 5), 1.0)

    def test_empty_gold_is_zero(self):
        self.assertEqual(hit_at_k(["a", "b"], [], 5), 0.0)

    def test_no_retrieval(self):
        self.assertEqual(hit_at_k([], ["a"], 5), 0.0)


class TestMRR(unittest.TestCase):
    def test_rank1_gives_one(self):
        self.assertEqual(mrr_at_k(["a", "b"], ["a"], 5), 1.0)

    def test_rank3_gives_one_third(self):
        # 1-based 排名的关键验证点
        self.assertAlmostEqual(mrr_at_k(["a", "b", "c"], ["c"], 5), 1 / 3)

    def test_rank2(self):
        self.assertAlmostEqual(mrr_at_k(["a", "b"], ["b"], 5), 0.5)

    def test_miss_is_zero(self):
        self.assertEqual(mrr_at_k(["a", "b", "c"], ["z"], 5), 0.0)

    def test_respects_k_cutoff(self):
        # gold 在第 6 位，k=5 -> 未命中
        self.assertEqual(mrr_at_k(list("abcdef"), ["f"], 5), 0.0)

    def test_earliest_hit_wins(self):
        # gold 出现在第 2 和第 4 位 -> 取第 2 位
        self.assertAlmostEqual(mrr_at_k(["a", "g", "b", "g2"], ["g", "g2"], 5), 0.5)


class TestRecallPrecision(unittest.TestCase):
    def test_recall_partial(self):
        # gold 2 个，命中 1 个
        self.assertAlmostEqual(recall_at_k(["a", "b"], ["a", "z"], 5), 0.5)

    def test_recall_full(self):
        self.assertEqual(recall_at_k(["a", "b"], ["a", "b"], 5), 1.0)

    def test_precision_denominator_is_k(self):
        # 前 3 条命中 1 条（另一条不在前 3），分母固定为 5
        self.assertAlmostEqual(precision_at_k(["a", "x", "y", "b"], ["a", "b"], 5), 0.4)

    def test_recall_empty_gold(self):
        self.assertEqual(recall_at_k(["a"], [], 5), 0.0)


class TestNDCG(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], ["a", "b"], 5), 1.0)

    def test_worse_ranking_lower_score(self):
        good = ndcg_at_k(["a", "b", "c"], ["a"], 5)
        bad = ndcg_at_k(["c", "b", "a"], ["a"], 5)
        self.assertGreater(good, bad)

    def test_miss_is_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], ["z"], 5), 0.0)

    def test_bounded(self):
        v = ndcg_at_k(["a", "b", "c"], ["b", "c"], 3)
        self.assertTrue(0.0 <= v <= 1.0)

    def test_known_value(self):
        # gold=[a] at rank2 -> dcg=1/log2(3); idcg=1/log2(2)=1
        self.assertAlmostEqual(ndcg_at_k(["x", "a"], ["a"], 5), 1 / math.log2(3))


class TestAggregateAndLatency(unittest.TestCase):
    def test_aggregate_mean(self):
        rows = [{"hit@5": 1.0, "mrr@5": 0.5}, {"hit@5": 0.0, "mrr@5": 0.1}]
        agg = aggregate(rows)
        self.assertAlmostEqual(agg["hit@5"], 0.5)
        self.assertAlmostEqual(agg["mrr@5"], 0.3)

    def test_aggregate_empty(self):
        self.assertEqual(aggregate([]), {})

    def test_latency_p95_ordering(self):
        stats = latency_stats([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        self.assertLessEqual(stats["p50_ms"], stats["p95_ms"])
        self.assertEqual(stats["max_ms"], 1000.0)
        self.assertEqual(stats["mean_ms"], 550.0)

    def test_latency_empty(self):
        self.assertEqual(latency_stats([]), {})


class TestRefusalAndHallucination(unittest.TestCase):
    def test_correct_refusal(self):
        self.assertEqual(refusal_correctness(False, "知识库中未找到相关内容"), 1.0)

    def test_hallucinated_answer_on_unanswerable(self):
        self.assertEqual(refusal_correctness(False, "该型号转速为 3000rpm"), 0.0)

    def test_wrong_refusal_on_answerable(self):
        self.assertEqual(refusal_correctness(True, "知识库中未找到相关内容"), 0.0)

    def test_normal_answer(self):
        self.assertEqual(refusal_correctness(True, "mAP 为 0.946"), 1.0)

    def test_hallucination_rate(self):
        # 4 条中 1 条低于阈值 3.0
        self.assertAlmostEqual(hallucination_rate([5, 4, 3, 2]), 0.25)

    def test_hallucination_rate_boundary_not_counted(self):
        # 恰好等于阈值不算幻觉（< 才对）
        self.assertEqual(hallucination_rate([3.0, 4.0]), 0.0)

    def test_hallucination_rate_empty(self):
        self.assertEqual(hallucination_rate([]), 0.0)


class TestRRF(unittest.TestCase):
    def test_single_list_preserves_order(self):
        fused = rrf_fuse({"vector": ["a", "b", "c"]})
        self.assertEqual([d for d, _ in fused], ["a", "b", "c"])

    def test_agreement_beats_single_list_top(self):
        # b 在两路都排第 2；a 只在一路排第 1 -> b 应当胜出
        fused = rrf_fuse({"vector": ["a", "b"], "bm25": ["c", "b"]}, k=60)
        self.assertEqual(fused[0][0], "b")

    def test_doc_in_both_lists_gets_summed_score(self):
        fused = dict(rrf_fuse({"vector": ["x"], "bm25": ["x"]}, k=60))
        self.assertAlmostEqual(fused["x"], 2.0 / 61)

    def test_scores_are_descending(self):
        fused = rrf_fuse({"v": ["a", "b", "c"], "b": ["c", "b", "a"]}, k=60)
        scores = [s for _, s in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_input(self):
        self.assertEqual(rrf_fuse({}), [])

    def test_missing_doc_only_in_one_list_still_scored(self):
        fused = rrf_fuse({"v": ["a"], "b": ["z"]}, k=60)
        self.assertEqual(len(fused), 2)


class TestWeightedFuseContrast(unittest.TestCase):
    def test_normalization_makes_scale_irrelevant(self):
        # vector 分数在 [0,1]，bm25 在 [0,100]。归一化后应可比。
        fused = weighted_fuse(
            {"vector": [("a", 0.9), ("b", 0.8)], "bm25": [("b", 90.0), ("c", 10.0)]}
        )
        ids = [d for d, _ in fused]
        self.assertIn("a", ids)
        self.assertIn("b", ids)

    def test_empty(self):
        self.assertEqual(weighted_fuse({}), [])


class TestDedupAndCut(unittest.TestCase):
    def test_dedup_keeps_first_order(self):
        self.assertEqual(dedup_keep_order(["b", "a", "b", "c", "a"]), ["b", "a", "c"])

    def test_fuse_and_cut_respects_top_k(self):
        cut = fuse_and_cut({"v": ["a", "b", "c", "d"], "b": ["e", "f"]}, top_k=3)
        self.assertEqual(len(cut), 3)

    def test_fuse_and_cut_empty(self):
        self.assertEqual(fuse_and_cut({}, top_k=5), [])


class TestTokenCount(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(count_tokens_approx(""), 0)

    def test_chinese_counts_per_char(self):
        self.assertEqual(count_tokens_approx("水稻穗检测"), 5)

    def test_ascii_word_counts_once(self):
        self.assertEqual(count_tokens_approx("hello"), 1)

    def test_mixed(self):
        # "mAP" 1 词 + "为" 1 字 + "0.946" 视为 1 词 + 标点
        n = count_tokens_approx("mAP 为 0.946。")
        self.assertGreaterEqual(n, 3)

    def test_monotonic(self):
        self.assertLess(count_tokens_approx("短句"), count_tokens_approx("这是一个明显更长的句子内容"))


class TestHeadingLevel(unittest.TestCase):
    def test_markdown_levels(self):
        self.assertEqual(heading_level("# 概述"), 1)
        self.assertEqual(heading_level("### 细则"), 3)

    def test_chinese_chapter_is_level1(self):
        self.assertEqual(heading_level("第一章 总则"), 1)
        self.assertEqual(heading_level("第二篇 检测"), 1)

    def test_chinese_section_is_level2(self):
        self.assertEqual(heading_level("第三节 精度要求"), 2)

    def test_article_line_is_content_not_heading(self):
        """★ 领域关键点：第X条后面是条文正文，不能被当标题剥离。"""
        self.assertIsNone(heading_level("第一条 本规范适用于水稻穗检测。"))
        self.assertIsNone(heading_level("第十五条 检测精度要求 mAP 不低于 0.9。"))

    def test_numeric_levels(self):
        self.assertEqual(heading_level("3 检测方法"), 1)
        self.assertEqual(heading_level("3.2 精度要求"), 2)
        self.assertEqual(heading_level("3.2.1 细则"), 3)

    def test_numeric_sentence_is_not_heading(self):
        self.assertIsNone(heading_level("3 个要点如下。"))
        self.assertIsNone(heading_level("2024 年检测量显著上升。"))

    def test_plain_sentence_is_not_heading(self):
        self.assertIsNone(heading_level("本章规定了水稻穗的检测精度要求。"))
        self.assertIsNone(heading_level(""))

    def test_ascii_sentence_not_heading(self):
        self.assertIsNone(heading_level("mAP should be above 0.9"))


class TestBuildHeadingPath(unittest.TestCase):
    def test_join(self):
        stack = [(1, "第三章"), (2, "3.2 检测标准")]
        self.assertEqual(build_heading_path(stack), "第三章 > 3.2 检测标准")

    def test_empty(self):
        self.assertEqual(build_heading_path([]), "")


class TestRecursiveChunk(unittest.TestCase):
    def test_basic_split_respects_chunk_size(self):
        text = "".join(f"这是第{i}个句子，用于测试切分逻辑是否遵守预算。" for i in range(60))
        chunks = recursive_chunk(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        # 允许 overlap 带来的少量超出
        for c in chunks[:-1]:
            self.assertLessEqual(c["token_len"], 100 + 20)

    def test_heading_path_accumulates(self):
        text = "第一章 总则\n第一条 本规范适用于水稻穗检测。\n第二章 技术要求\n第一条 检测精度要求 mAP 不低于 0.9。"
        chunks = recursive_chunk(text, chunk_size=200, overlap=0)
        paths = [c["heading_path"] for c in chunks]
        # 章是标题，会进入 heading_path
        self.assertTrue(any("第一章" in p for p in paths), paths)
        self.assertTrue(any("第二章" in p for p in paths), paths)
        # 第X条是正文，不应作为独立层级覆盖章
        self.assertFalse(any(p.startswith("第一条") for p in paths), paths)
        # 正文内容必须完整保留（这是最初的 bug 所在）
        joined = "".join(c["text"] for c in chunks)
        self.assertIn("本规范适用于水稻穗检测", joined)
        self.assertIn("mAP 不低于 0.9", joined)

    def test_section_nests_under_chapter(self):
        text = "第一章 总则\n第一节 适用范围\n内容甲。\n第二章 技术要求\n内容乙。"
        chunks = recursive_chunk(text, chunk_size=200, overlap=0)
        paths = [c["heading_path"] for c in chunks]
        self.assertTrue(any("第一章" in p and "第一节" in p for p in paths), paths)
        # 第二章是同级，应把第一章/第一节都出栈
        self.assertTrue(any(p.startswith("第二章") and "第一章" not in p for p in paths), paths)

    def test_markdown_heading(self):
        text = "# 概述\n内容甲。\n## 细则\n内容乙。"
        chunks = recursive_chunk(text, chunk_size=100, overlap=0)
        self.assertTrue(any("概述 > 细则" in c["heading_path"] for c in chunks))
    def test_no_content_lost(self):
        text = "第一句内容。第二句内容。第三句内容。"
        chunks = recursive_chunk(text, chunk_size=8, overlap=0)
        joined = "".join(c["text"] for c in chunks)
        for frag in ["第一句内容", "第二句内容", "第三句内容"]:
            self.assertIn(frag, joined)

    def test_overlap_creates_repeated_context(self):
        text = "".join(f"句子{i}的内容比较长一些以便触发切分。" for i in range(20))
        with_ov = recursive_chunk(text, chunk_size=60, overlap=30)
        without = recursive_chunk(text, chunk_size=60, overlap=0)
        self.assertGreaterEqual(len(with_ov), len(without))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            recursive_chunk("x", chunk_size=0)
        with self.assertRaises(ValueError):
            recursive_chunk("x", chunk_size=100, overlap=100)

    def test_empty_text(self):
        self.assertEqual(recursive_chunk("", chunk_size=100, overlap=10), [])

    def test_token_len_recorded(self):
        chunks = recursive_chunk("水稻穗检测精度要求。", chunk_size=100, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token_len"], count_tokens_approx(chunks[0]["text"]))


class TestAllRetrievalMetricsIntegration(unittest.TestCase):
    """模拟一档配置在一条 query 上的完整指标输出。"""

    def test_keys_and_ranges(self):
        m = all_retrieval_metrics(["a", "b", "c", "d", "e"], ["c", "d"], k=5)
        for key in ("hit@5", "mrr@5", "recall@5", "precision@5", "ndcg@5"):
            self.assertIn(key, m)
            self.assertTrue(0.0 <= m[key] <= 1.0, f"{key} out of range: {m[key]}")
        self.assertEqual(m["hit@5"], 1.0)
        self.assertAlmostEqual(m["mrr@5"], 1 / 3)
        self.assertAlmostEqual(m["recall@5"], 1.0)
        self.assertAlmostEqual(m["precision@5"], 0.4)

    def test_ablation_style_comparison(self):
        """模拟简历里那一行：纯向量 73% -> 混合 92% 应当能复现出差异。"""
        gold = ["g1", "g2", "g3", "g4"]
        vector_only = [
            ["g1", "x", "x", "x", "x"],   # hit
            ["x", "x", "x", "x", "x"],    # miss
            ["x", "g2", "x", "x", "x"],   # hit
            ["x", "x", "x", "x", "x"],    # miss
        ]
        hybrid = [
            ["g1", "x", "x", "x", "x"],
            ["x", "g2", "x", "x", "x"],
            ["g3", "x", "x", "x", "x"],
            ["x", "x", "x", "x", "g4"],
        ]
        a = aggregate([all_retrieval_metrics(r, gold, 5) for r in vector_only])
        b = aggregate([all_retrieval_metrics(r, gold, 5) for r in hybrid])
        self.assertAlmostEqual(a["hit@5"], 0.5)
        self.assertAlmostEqual(b["hit@5"], 1.0)
        self.assertGreater(b["mrr@5"], a["mrr@5"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
