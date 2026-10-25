# -*- coding: utf-8 -*-
"""tests/test_pipeline.py —— text / bm25 / vectorstore / llm 的单测。

全部离线可跑（无网络、无 GPU、无第三方依赖）。
"""
import os
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ★ 注意：不使用系统 temp 目录。DSH 沙箱下 %TEMP% 不可写，
#   而这些测试需要真实落盘。统一放在测试目录下的 .tmp/（已被 .gitignore 排除）。
_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


def make_tmp() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    path = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(path, exist_ok=True)
    return path


def clean_tmp(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)

from core.bm25 import BM25Index  # noqa: E402
from core.llm import (  # noqa: E402
    DiskCache, LLMClient, LLMError, NonRetryableError,
    REFUSAL_MARKER, RetryableError, TokenBucket, build_context,
    build_answer_messages, should_refuse,
)
from core.text import (  # noqa: E402
    extract_measures, jaccard, normalize_whitespace, tokenize,
)
from core.vectorstore import (  # noqa: E402
    InMemoryVectorStore, dot, l2_normalize,
)


# ===================================================================== text
class TestNormalize(unittest.TestCase):
    def test_fullwidth_space_and_zero_width(self):
        self.assertEqual(normalize_whitespace("水稻\u3000穗\u200b检测"), "水稻 穗检测")

    def test_collapses_blank_lines(self):
        self.assertEqual(normalize_whitespace("a\n\n\n\nb"), "a\n\nb")

    def test_empty(self):
        self.assertEqual(normalize_whitespace(""), "")


class TestTokenize(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(tokenize(""), [])

    def test_ascii_lowercased(self):
        self.assertIn("map", tokenize("mAP"))

    def test_cjk_produces_unigram_and_bigram(self):
        toks = tokenize("水稻穗")
        # 单字保证召回
        for ch in "水稻穗":
            self.assertIn(ch, toks)
        # 双字提供区分度
        self.assertIn("水稻", toks)
        self.assertIn("稻穗", toks)

    def test_single_char_cjk_has_no_bigram(self):
        self.assertEqual(sorted(tokenize("稻")), ["稻"])

    def test_measure_terms_kept(self):
        toks = tokenize("mAP@0.5 不低于 0.90，F1 不低于 0.88")
        self.assertIn("0.90", toks)
        self.assertIn("0.88", toks)
        self.assertIn("0.5", toks)

    def test_standard_number_kept(self):
        toks = tokenize("依据 GB/T 30600-2022 执行")
        joined = " ".join(toks)
        self.assertIn("30600", joined)
        self.assertIn("2022", joined)

    def test_mixed_script(self):
        toks = tokenize("使用 YOLOv8 检测水稻穗")
        self.assertIn("yolov8", toks)
        self.assertIn("水稻", toks)

    def test_keep_measure_false(self):
        toks = tokenize("精度 0.90", keep_measure=False)
        self.assertNotIn("0.90", toks)


class TestExtractMeasuresAndJaccard(unittest.TestCase):
    def test_extract_measures(self):
        # ★ 刻意排除单个数字：中文语料里 "3 个要点" 这类极常见，
        #   收进词表只会稀释 IDF。只收小数、百分比、多位整数。
        self.assertEqual(extract_measures("mAP 0.946 与 F1 0.91，提升 3 个百分点"),
                         ["0.946", "0.91"])

    def test_extract_measures_keeps_standard_numbers(self):
        self.assertEqual(extract_measures("依据 GB/T 30600-2022，覆盖率 75%"),
                         ["30600", "2022", "75%"])

    def test_extract_measures_ignores_ratio_digits(self):
        # "7:1:2" 里的个位数不应进词表
        self.assertEqual(extract_measures("训练集按 7:1:2 划分"), [])

    def test_extract_measures_empty(self):
        self.assertEqual(extract_measures(""), [])

    def test_jaccard_identical(self):
        self.assertEqual(jaccard(["a", "b"], ["b", "a"]), 1.0)

    def test_jaccard_disjoint(self):
        self.assertEqual(jaccard(["a"], ["b"]), 0.0)

    def test_jaccard_partial(self):
        self.assertAlmostEqual(jaccard(["a", "b", "c"], ["a", "b", "d"]), 0.5)

    def test_jaccard_both_empty(self):
        self.assertEqual(jaccard([], []), 1.0)


# ===================================================================== bm25
class TestBM25(unittest.TestCase):
    def _index(self):
        idx = BM25Index()
        idx.add("d1", "水稻穗检测精度要求 mAP 不低于 0.90")
        idx.add("d2", "小麦头检测数据集按 7:1:2 划分")
        idx.add("d3", "模型应支持导出 ONNX 并量化以用于端侧推理")
        idx.finalize()
        return idx

    def test_len(self):
        self.assertEqual(len(self._index()), 3)

    def test_relevant_doc_ranks_first(self):
        idx = self._index()
        results = idx.search("水稻穗 精度", top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0][0], "d1")

    def test_irrelevant_term_returns_nothing_with_threshold(self):
        """中文单字噪声：不加阈值时无关查询也会命中低分文档（BM25 的真实弱点）。

        实测：无关查询靠单字重合能拿到 ~1.0 分；相关查询在 2.x 量级。
        所以 min_score 要按语料量级来定，不能想当然取 0.5。
        """
        idx = self._index()
        noisy = idx.search("完全不相关的词条")
        self.assertTrue(noisy, "默认阈值下应命中低分噪声，用于说明为何需要分数下限")
        noise_score = noisy[0][1]

        relevant = idx.search("水稻穗 精度")
        relevant_score = relevant[0][1]
        self.assertGreater(relevant_score, noise_score, "相关查询得分必须显著高于噪声")

        # 取一个介于两者之间的阈值即可干净过滤
        cut = (noise_score + relevant_score) / 2
        self.assertEqual(idx.search("完全不相关的词条", min_score=cut), [])
        self.assertTrue(idx.search("水稻穗 精度", min_score=cut))

    def test_relevant_doc_survives_threshold(self):
        idx = self._index()
        results = idx.search("水稻穗 精度", min_score=0.5)
        self.assertTrue(results)
        self.assertEqual(results[0][0], "d1")

    def test_min_score_can_exclude_all(self):
        idx = self._index()
        self.assertEqual(idx.search("水稻穗", min_score=999.0), [])

    def test_second_doc_retrievable(self):
        idx = self._index()
        self.assertEqual(idx.search("数据集 划分")[0][0], "d2")

    def test_english_term(self):
        idx = self._index()
        self.assertEqual(idx.search("ONNX 量化")[0][0], "d3")

    def test_scores_descending(self):
        idx = self._index()
        scores = [s for _, s in idx.search("检测 精度 数据集", top_k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_idf_penalizes_common_terms(self):
        idx = BM25Index()
        # "检测" 出现在所有文档 -> IDF 低；"水稻" 只在一篇 -> IDF 高
        idx.add("a", "检测 水稻")
        idx.add("b", "检测 小麦")
        idx.add("c", "检测 玉米")
        idx.finalize()
        self.assertGreater(idx.idf("水稻"), idx.idf("检测"))

    def test_empty_query(self):
        self.assertEqual(self._index().search(""), [])

    def test_empty_index(self):
        idx = BM25Index()
        idx.finalize()
        self.assertEqual(idx.search("任意查询"), [])

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            BM25Index(k1=-1)
        with self.assertRaises(ValueError):
            BM25Index(b=1.5)

    def test_add_many_finalizes(self):
        idx = BM25Index()
        idx.add_many([("a", "水稻检测"), ("b", "小麦检测")])
        self.assertGreater(idx.avgdl, 0.0)
        self.assertTrue(idx.search("水稻"))

    def test_deterministic_tie_break(self):
        idx = BM25Index()
        idx.add("z", "相同内容")
        idx.add("a", "相同内容")
        idx.finalize()
        first = idx.search("相同内容")
        self.assertEqual([d for d, _ in first], sorted([d for d, _ in first]))

    def test_top_k_limits(self):
        idx = self._index()
        self.assertLessEqual(len(idx.search("检测", top_k=2)), 2)


# ================================================================= vectorstore
class TestVectorMath(unittest.TestCase):
    def test_normalize_unit_length(self):
        v = l2_normalize([3.0, 4.0])
        self.assertAlmostEqual(dot(v, v), 1.0)

    def test_normalize_zero_vector(self):
        self.assertEqual(l2_normalize([0.0, 0.0]), [0.0, 0.0])

    def test_dot_basic(self):
        self.assertAlmostEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)


class TestInMemoryVectorStore(unittest.TestCase):
    def _store(self):
        s = InMemoryVectorStore()
        s.add(
            ids=["d1", "d2", "d3"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            documents=["水稻检测", "小麦检测", "玉米检测"],
            metadatas=[{"heading_path": "第一章"}, {"heading_path": "第二章"},
                       {"heading_path": "第三章"}],
        )
        return s

    def test_count(self):
        self.assertEqual(self._store().count(), 3)

    def test_query_returns_nearest_first(self):
        res = self._store().query([1.0, 0.0, 0.0], top_k=3)
        self.assertEqual(res[0][0], "d1")
        # d3 与 d1 方向接近，应排在 d2 之前
        self.assertEqual([d for d, _ in res], ["d1", "d3", "d2"])

    def test_similarity_range(self):
        for _, score in self._store().query([1.0, 0.0, 0.0], top_k=3):
            self.assertLessEqual(score, 1.0 + 1e-9)
            self.assertGreaterEqual(score, -1.0 - 1e-9)

    def test_get(self):
        rec = self._store().get("d2")
        self.assertEqual(rec["document"], "小麦检测")
        self.assertEqual(rec["metadata"]["heading_path"], "第二章")

    def test_get_missing(self):
        self.assertIsNone(self._store().get("nope"))

    def test_delete(self):
        s = self._store()
        s.delete(["d2"])
        self.assertEqual(s.count(), 2)
        self.assertIsNone(s.get("d2"))

    def test_delete_missing_is_noop(self):
        s = self._store()
        s.delete(["nope"])
        self.assertEqual(s.count(), 3)

    def test_length_mismatch_raises(self):
        s = InMemoryVectorStore()
        with self.assertRaises(ValueError):
            s.add(ids=["a"], embeddings=[[1.0]], documents=["x", "y"])

    def test_metadata_defaults_to_empty(self):
        s = InMemoryVectorStore()
        s.add(ids=["a"], embeddings=[[1.0, 0.0]], documents=["x"])
        self.assertEqual(s.get("a")["metadata"], {})

    def test_persist_and_load_roundtrip(self):
        tmp = make_tmp()
        try:
            path = os.path.join(tmp, "vs.json")
            s = self._store()
            s.persist(path)
            loaded = InMemoryVectorStore.load(path)
            self.assertEqual(loaded.count(), 3)
            self.assertEqual(loaded.get("d1")["document"], "水稻检测")
            # 检索结果应一致
            self.assertEqual(
                [d for d, _ in loaded.query([1.0, 0.0, 0.0], top_k=3)],
                [d for d, _ in s.query([1.0, 0.0, 0.0], top_k=3)],
            )
        finally:
            clean_tmp(tmp)

    def test_top_k_larger_than_store(self):
        self.assertEqual(len(self._store().query([1.0, 0.0, 0.0], top_k=99)), 3)


# ===================================================================== llm
def ok_response(text="回答内容", pt=100, ct=20):
    return {
        "model": "deepseek-chat",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct},
    }


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestTokenBucket(unittest.TestCase):
    def test_immediate_acquire_within_capacity(self):
        b = TokenBucket(rate=100.0, capacity=5)
        for _ in range(5):
            self.assertTrue(b.acquire())

    def test_rate_limit_eventually_allows(self):
        b = TokenBucket(rate=1000.0, capacity=1)
        self.assertTrue(b.acquire())
        self.assertTrue(b.acquire(timeout=1.0))

    def test_timeout_returns_false(self):
        b = TokenBucket(rate=0.001, capacity=1)
        self.assertTrue(b.acquire())
        self.assertFalse(b.acquire(timeout=0.05))

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            TokenBucket(rate=0)


class TestDiskCache(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()

    def tearDown(self):
        clean_tmp(self.tmp)

    def test_roundtrip(self):
        c = DiskCache(self.tmp)
        key = DiskCache.make_key("a", 1, {"x": 2})
        self.assertIsNone(c.get(key))
        c.set(key, {"text": "hi"})
        self.assertEqual(c.get(key)["text"], "hi")

    def test_key_is_deterministic_and_order_insensitive(self):
        self.assertEqual(DiskCache.make_key({"a": 1, "b": 2}), DiskCache.make_key({"b": 2, "a": 1}))

    def test_disabled_cache(self):
        c = DiskCache(self.tmp, enabled=False)
        key = DiskCache.make_key("x")
        c.set(key, {"text": "y"})
        self.assertIsNone(c.get(key))

    def test_stats(self):
        c = DiskCache(self.tmp)
        key = DiskCache.make_key("x")
        c.get(key)
        c.set(key, {"text": "y"})
        c.get(key)
        self.assertEqual(c.stats(), {"hits": 1, "misses": 1})

    def test_corrupt_file_is_miss_not_crash(self):
        c = DiskCache(self.tmp)
        key = DiskCache.make_key("x")
        with open(os.path.join(self.tmp, f"{key}.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertIsNone(c.get(key))


class TestErrorClassification(unittest.TestCase):
    def test_429_is_retryable(self):
        self.assertIs(LLMClient.classify_error(FakeHTTPError(429)), RetryableError)

    def test_5xx_is_retryable(self):
        for code in (500, 502, 503, 504):
            self.assertIs(LLMClient.classify_error(FakeHTTPError(code)), RetryableError)

    def test_400_and_401_are_not_retryable(self):
        for code in (400, 401, 403, 404):
            self.assertIs(LLMClient.classify_error(FakeHTTPError(code)), NonRetryableError)

    def test_timeout_by_name(self):
        class APITimeoutError(Exception):
            pass
        self.assertIs(LLMClient.classify_error(APITimeoutError()), RetryableError)

    def test_connection_by_message(self):
        self.assertIs(LLMClient.classify_error(Exception("Connection reset")), RetryableError)

    def test_unknown_is_not_retryable(self):
        self.assertIs(LLMClient.classify_error(ValueError("bad arg")), NonRetryableError)


class TestLLMRetry(unittest.TestCase):
    def _client(self, transport, **kw):
        slept = []
        client = LLMClient(api_key="k", transport=transport, sleep=slept.append,
                           jitter=False, rate=None, **kw)
        return client, slept

    def test_success_first_try(self):
        client, slept = self._client(lambda **kw: ok_response())
        res = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(res.text, "回答内容")
        self.assertEqual(res.attempts, 1)
        self.assertEqual(res.total_tokens, 120)
        self.assertEqual(slept, [])

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def transport(**kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise FakeHTTPError(429)
            return ok_response()

        client, slept = self._client(transport)
        res = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(res.attempts, 3)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(len(slept), 2)
        self.assertEqual(client.usage["retries"], 2)

    def test_backoff_is_exponential(self):
        def transport(**kw):
            raise FakeHTTPError(503)

        client, slept = self._client(transport, base_delay=1.0, max_retries=3)
        with self.assertRaises(RetryableError):
            client.chat([{"role": "user", "content": "hi"}])
        # 1, 2, 4
        self.assertEqual(slept, [1.0, 2.0, 4.0])

    def test_non_retryable_fails_immediately(self):
        calls = {"n": 0}

        def transport(**kw):
            calls["n"] += 1
            raise FakeHTTPError(401)

        client, slept = self._client(transport)
        with self.assertRaises(NonRetryableError):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(calls["n"], 1)
        self.assertEqual(slept, [])

    def test_exhausts_retries(self):
        client, _ = self._client(lambda **kw: (_ for _ in ()).throw(FakeHTTPError(429)),
                                 max_retries=2)
        with self.assertRaises(RetryableError):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(client.usage["errors"], 1)

    def test_jitter_varies_delay(self):
        slept = []
        client = LLMClient(api_key="k", transport=lambda **kw: (_ for _ in ()).throw(FakeHTTPError(429)),
                           sleep=slept.append, jitter=True, rate=None,
                           base_delay=1.0, max_retries=3)
        with self.assertRaises(RetryableError):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(slept), 3)
        # 抖动后不应完全等于无抖动的 1/2/4
        self.assertFalse(slept == [1.0, 2.0, 4.0])


class TestLLMCache(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmp()

    def tearDown(self):
        clean_tmp(self.tmp)

    def test_second_identical_call_is_cached(self):
        calls = {"n": 0}

        def transport(**kw):
            calls["n"] += 1
            return ok_response()

        cache = DiskCache(self.tmp)
        client = LLMClient(api_key="k", transport=transport, cache=cache, rate=None)
        msgs = [{"role": "user", "content": "同样的输入"}]
        r1 = client.chat(msgs)
        r2 = client.chat(msgs)
        self.assertFalse(r1.cached)
        self.assertTrue(r2.cached)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(client.usage["cached"], 1)

    def test_different_input_not_cached(self):
        calls = {"n": 0}

        def transport(**kw):
            calls["n"] += 1
            return ok_response()

        client = LLMClient(api_key="k", transport=transport,
                           cache=DiskCache(self.tmp), rate=None)
        client.chat([{"role": "user", "content": "A"}])
        client.chat([{"role": "user", "content": "B"}])
        self.assertEqual(calls["n"], 2)

    def test_use_cache_false_bypasses(self):
        calls = {"n": 0}

        def transport(**kw):
            calls["n"] += 1
            return ok_response()

        client = LLMClient(api_key="k", transport=transport,
                           cache=DiskCache(self.tmp), rate=None)
        msgs = [{"role": "user", "content": "same"}]
        client.chat(msgs)
        client.chat(msgs, use_cache=False)
        self.assertEqual(calls["n"], 2)


class TestLLMJsonMode(unittest.TestCase):
    def _client(self, payload_text):
        return LLMClient(api_key="k", transport=lambda **kw: ok_response(payload_text),
                         rate=None)

    def test_parses_plain_json(self):
        client = self._client('{"score": 5}')
        self.assertEqual(client.chat_json([{"role": "user", "content": "x"}])["score"], 5)

    def test_parses_fenced_json(self):
        client = self._client('```json\n{"score": 4}\n```')
        self.assertEqual(client.chat_json([{"role": "user", "content": "x"}])["score"], 4)

    def test_invalid_json_raises(self):
        client = self._client("这不是 JSON")
        with self.assertRaises(LLMError):
            client.chat_json([{"role": "user", "content": "x"}])

    def test_sets_response_format(self):
        seen = {}

        def transport(**kw):
            seen.update(kw)
            return ok_response('{"ok": true}')

        client = LLMClient(api_key="k", transport=transport, rate=None)
        client.chat_json([{"role": "user", "content": "x"}])
        self.assertEqual(seen["response_format"], {"type": "json_object"})


class TestBuildContextAndRefusal(unittest.TestCase):
    def test_context_numbering(self):
        hits = [
            {"doc_id": "a", "text": "内容甲", "heading_path": "第一章"},
            {"doc_id": "b", "text": "内容乙", "heading_path": ""},
        ]
        ctx = build_context(hits)
        self.assertIn("[1] 第一章", ctx)
        self.assertIn("[2]", ctx)
        self.assertIn("内容甲", ctx)

    def test_answer_prompt_contains_rules_and_refusal(self):
        msgs = build_answer_messages("问题？", [{"doc_id": "a", "text": "片段", "score": 0.9}])
        body = msgs[-1]["content"]
        self.assertIn("引用", body)
        self.assertIn(REFUSAL_MARKER, body)
        self.assertIn("片段", body)

    def test_answer_prompt_requires_partial_answer_before_refusal(self):
        """★ 回归：拒答条件必须写成分级，否则会大面积过度拒答。

        初版 prompt 第 2 条是：

            如果片段中没有足够信息，必须回答"知识库中未找到相关内容"

        这句话太容易触发 —— 模型见到"部分匹配"或"信息不完整"就套用固定话术，
        而不是尽力作答。实测（119 条评测集，清缓存全新调用）：

            拒答正确率 0.60，可答覆盖率仅 51.5%
            相关性 3.21 / 准确性 3.03 / 完整性 2.92

        改成「只有完全找不到相关内容才拒答；只要含部分相关信息，就必须先答
        能答的部分并指出缺什么」之后：

            拒答正确率 0.86，相关性 4.17 / 准确性 3.69 / 完整性 3.61
            幻觉率反而从 2.02% 降到 1.01%

        ★ 即「过度拒答」与「幻觉」并非此消彼长：把"部分信息也要作答"
          这条写清楚，两个指标同时变好。这条测试锁住这个结论。
        """
        body = build_answer_messages(
            "问题？", [{"doc_id": "a", "text": "片段", "score": 0.9}])[-1]["content"]
        self.assertIn("完全找不到", body,
                      "拒答条件必须限定为「完全找不到」，否则会过度拒答")
        self.assertIn("部分", body,
                      "必须明确要求：含部分相关信息时先答能答的部分")
        self.assertIn("不要因为信息不完整就直接整段拒答", body)
        self.assertNotIn("如果片段中没有足够信息，必须回答", body,
                         "旧的二元拒答写法会让模型大面积过度拒答")

    def test_should_refuse_empty(self):
        self.assertTrue(should_refuse([], 0.3))

    def test_should_refuse_below_threshold(self):
        self.assertTrue(should_refuse([{"score": 0.1}], 0.3))

    def test_should_not_refuse_above_threshold(self):
        self.assertFalse(should_refuse([{"score": 0.5}], 0.3))

    def test_threshold_boundary_not_refused(self):
        self.assertFalse(should_refuse([{"score": 0.3}], 0.3))


class TestUsageAccounting(unittest.TestCase):
    def test_accumulates_tokens(self):
        client = LLMClient(api_key="k", transport=lambda **kw: ok_response(pt=50, ct=10),
                           rate=None)
        client.chat([{"role": "user", "content": "a"}])
        client.chat([{"role": "user", "content": "b"}])
        rep = client.cost_report()
        self.assertEqual(rep["calls"], 2)
        self.assertEqual(rep["prompt_tokens"], 100)
        self.assertEqual(rep["completion_tokens"], 20)


class TestMemoryStoreButFakeEmbeddingPipeline(unittest.TestCase):
    """端到端（无网络/无GPU）：切分 -> 假向量 -> 向量检索 + BM25 -> RRF 融合。"""

    def test_hybrid_retrieval_prefers_agreement(self):
        from core.retrieval import rrf_fuse, recursive_chunk

        text = ("第一章 精度要求\n"
                "水稻穗检测 mAP 不低于 0.90。\n"
                "第二章 数据集\n"
                "训练集按 7:1:2 划分，标注用 Labelme。\n"
                "第三章 部署\n"
                "模型导出 ONNX 并量化。\n")
        chunks = recursive_chunk(text, chunk_size=40, overlap=0)
        self.assertEqual(len(chunks), 3)

        # BM25 真检索
        bm25 = BM25Index()
        for i, c in enumerate(chunks):
            bm25.add(f"c{i}", c["text"])
        bm25.finalize()
        bm25_hits = [d for d, _ in bm25.search("mAP 精度", top_k=10)]
        self.assertEqual(bm25_hits[0], "c0")

        # 假向量：用字符重叠度当相似度，模拟"语义路"
        import hashlib

        def fake_vec(s: str, dim: int = 32):
            v = [0.0] * dim
            for tok in tokenize(s):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % dim] += 1.0
            return l2_normalize(v)

        store = InMemoryVectorStore()
        store.add(ids=[f"c{i}" for i in range(len(chunks))],
                  embeddings=[fake_vec(c["text"]) for c in chunks],
                  documents=[c["text"] for c in chunks])
        vec_hits = [d for d, _ in store.query(fake_vec("mAP 精度"), top_k=10)]
        self.assertTrue(vec_hits)

        fused = rrf_fuse({"vector": vec_hits, "bm25": bm25_hits}, k=60)
        self.assertTrue(fused)
        # 检索融合后应仍能把最相关片段排在前列
        self.assertIn(fused[0][0], {"c0", "c2"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
