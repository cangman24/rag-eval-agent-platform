# -*- coding: utf-8 -*-
"""tests/test_memory.py —— 三层记忆 + 上下文压缩单测。全部离线。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import (  # noqa: E402
    ContextCompressor, EpisodicMemory, LayeredMemory,
    Message, Turn, UserProfile, WorkingMemory, truncate_tool_result,
)
from core.vectorstore import InMemoryVectorStore, l2_normalize  # noqa: E402


def fake_embed(text: str, dim: int = 32):
    v = [0.0] * dim
    for ch in text:
        if ch.strip():
            v[ord(ch) % dim] += 1.0
    return l2_normalize(v)


# ================================================================== message
class TestMessage(unittest.TestCase):
    def test_roundtrip(self):
        m = Message("user", "内容", ts=1.0, meta={"a": 1})
        m2 = Message.from_dict(m.to_dict())
        self.assertEqual(m2.role, "user")
        self.assertEqual(m2.content, "内容")
        self.assertEqual(m2.meta, {"a": 1})

    def test_from_dict_defaults(self):
        m = Message.from_dict({})
        self.assertEqual(m.role, "user")
        self.assertEqual(m.content, "")

    def test_tokens_positive(self):
        self.assertGreater(Message("user", "水稻穗检测精度").tokens(), 0)

    def test_turn_roundtrip(self):
        t = Turn("问", "答")
        t2 = Turn.from_dict(t.to_dict())
        self.assertEqual(t2.user, "问")
        self.assertEqual(t2.assistant, "答")


# ================================================================== working
class TestWorkingMemory(unittest.TestCase):
    def test_add_and_recent(self):
        wm = WorkingMemory(max_turns=3)
        for i in range(3):
            wm.add(f"问{i}", f"答{i}")
        self.assertEqual(len(wm), 3)
        self.assertEqual(wm.recent()[-1].user, "问2")

    def test_lru_eviction(self):
        wm = WorkingMemory(max_turns=2)
        wm.add("问0")
        wm.add("问1")
        wm.add("问2")
        self.assertEqual(len(wm), 2)
        users = [t.user for t in wm.recent()]
        self.assertNotIn("问0", users, "最旧的轮次应被淘汰")
        self.assertIn("问2", users)

    def test_recent_n(self):
        wm = WorkingMemory(max_turns=10)
        for i in range(5):
            wm.add(f"问{i}")
        self.assertEqual(len(wm.recent(2)), 2)
        self.assertEqual(wm.recent(2)[-1].user, "问4")

    def test_messages_shape(self):
        wm = WorkingMemory()
        wm.add("问", "答")
        msgs = wm.messages()
        self.assertEqual([m.role for m in msgs], ["user", "assistant"])

    def test_messages_skip_empty_assistant(self):
        wm = WorkingMemory()
        wm.add("问")
        self.assertEqual([m.role for m in wm.messages()], ["user"])

    def test_tokens(self):
        wm = WorkingMemory()
        wm.add("这是一个问题", "这是回答")
        self.assertGreater(wm.tokens(), 0)

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("问")
        wm.clear()
        self.assertEqual(len(wm), 0)

    def test_invalid_max_turns(self):
        with self.assertRaises(ValueError):
            WorkingMemory(max_turns=0)

    def test_duplicate_content_still_counts(self):
        wm = WorkingMemory(max_turns=5)
        wm.add("同一个问题")
        wm.add("同一个问题")
        self.assertEqual(len(wm), 2)


# ================================================================== episodic
class TestEpisodicMemory(unittest.TestCase):
    def test_add_and_len(self):
        em = EpisodicMemory()
        em.add("历史摘要一")
        self.assertEqual(len(em), 1)

    def test_recall_without_vector_store_uses_token_overlap(self):
        em = EpisodicMemory()
        em.add("检测精度要求 mAP 不低于 0.90")
        em.add("数据集按 7:1:2 划分")
        hits = em.recall("mAP 精度", top_k=2)
        self.assertTrue(hits)
        self.assertIn("mAP", hits[0]["summary"])

    def test_recall_with_vector_store(self):
        store = InMemoryVectorStore()
        em = EpisodicMemory(vector_store=store, embed_fn=fake_embed)
        em.add("水稻穗检测精度")
        em.add("小麦头数据集划分")
        self.assertEqual(store.count(), 2)
        hits = em.recall("水稻穗", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("score", hits[0])

    def test_recall_returns_meta(self):
        em = EpisodicMemory()
        em.add("摘要", meta={"session": "s1"})
        hits = em.recall("摘要", top_k=1)
        self.assertEqual(hits[0]["meta"]["session"], "s1")

    def test_recall_no_match_returns_empty(self):
        em = EpisodicMemory()
        em.add("检测精度要求")
        self.assertEqual(em.recall("完全无关的航天器轨道", top_k=3), [])

    def test_id_is_deterministic_for_same_text(self):
        em = EpisodicMemory()
        self.assertEqual(em.add("同样内容"), em.add("同样内容"))


# ================================================================== profile
class TestUserProfile(unittest.TestCase):
    def test_update_and_get(self):
        p = UserProfile()
        p.update("常用教室", "A301")
        self.assertEqual(p.get("常用教室"), "A301")
        self.assertIsNone(p.get("不存在"))

    def test_update_many(self):
        p = UserProfile()
        p.update_many({"a": 1, "b": 2})
        self.assertEqual(len(p), 2)

    def test_to_text_sorted(self):
        p = UserProfile()
        p.update_many({"b": 2, "a": 1})
        self.assertEqual(p.to_text(), "a=1；b=2")

    def test_to_text_empty(self):
        self.assertEqual(UserProfile().to_text(), "")

    def test_overwrite(self):
        p = UserProfile()
        p.update("k", 1)
        p.update("k", 2)
        self.assertEqual(p.get("k"), 2)
        self.assertEqual(len(p), 1)


# ================================================================== layered
class TestLayeredMemory(unittest.TestCase):
    def _mem(self):
        store = InMemoryVectorStore()
        return LayeredMemory(
            working=WorkingMemory(max_turns=5),
            episodic=EpisodicMemory(vector_store=store, embed_fn=fake_embed),
            profile=UserProfile(),
        )

    def test_build_context_shape(self):
        mem = self._mem()
        mem.working.add("之前问过精度", "要求 0.90")
        mem.episodic.add("用户关心检测精度")
        mem.profile.update("角色", "算法工程师")
        ctx = mem.build_context("精度要求", recent_n=3, recall_k=2)
        self.assertEqual(ctx["profile"], "角色=算法工程师")
        self.assertTrue(ctx["episodic"])
        self.assertTrue(ctx["recent"])
        self.assertIn("role", ctx["recent"][0])

    def test_build_context_empty_query_skips_recall(self):
        mem = self._mem()
        mem.episodic.add("某条摘要")
        ctx = mem.build_context("", recent_n=3)
        self.assertEqual(ctx["episodic"], [])

    def test_total_tokens(self):
        mem = self._mem()
        mem.working.add("问", "答")
        self.assertGreater(mem.total_tokens(), 0)


# ================================================================== truncate
class TestTruncateToolResult(unittest.TestCase):
    def test_short_unchanged(self):
        self.assertEqual(truncate_tool_result("短内容", 100), "短内容")

    def test_long_truncated_with_marker(self):
        out = truncate_tool_result("x" * 500, 100)
        self.assertLess(len(out), 500)
        self.assertIn("已截断", out)

    def test_none_becomes_empty(self):
        self.assertEqual(truncate_tool_result(None), "")

    def test_boundary_exact(self):
        s = "y" * 100
        self.assertEqual(truncate_tool_result(s, 100), s)


# ================================================================== compress
def msgs(n: int, chars: int = 40, role: str = "user") -> list:
    return [Message(role, f"消息{i}-" + "内" * chars) for i in range(n)]


class TestContextCompressor(unittest.TestCase):
    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            ContextCompressor(token_budget=1000, tier1=0.9, tier2=0.6)
        with self.assertRaises(ValueError):
            ContextCompressor(token_budget=0)
        with self.assertRaises(ValueError):
            ContextCompressor(keep_recent=(5, 3))

    def test_below_threshold_no_compression(self):
        c = ContextCompressor(token_budget=100000)
        r = c.compress(msgs(3))
        self.assertEqual(r.level, "none")
        self.assertEqual(r.tokens_after, r.tokens_before)
        self.assertEqual(r.saved_ratio, 0.0)

    def test_tier1_triggers_early_summary(self):
        # 预算设得很小，确保 ratio 落在 tier1 区间
        c = ContextCompressor(token_budget=300, tier1=0.6, tier2=0.85, tier3=0.95,
                              keep_recent=(5, 3, 2))
        r = c.compress(msgs(12, chars=20))
        self.assertIn(r.level, ("early_summary", "structured_archive", "full_compress"))
        self.assertLess(r.tokens_after, r.tokens_before)

    def test_tier3_triggers_full_compress(self):
        c = ContextCompressor(token_budget=100, tier1=0.6, tier2=0.85, tier3=0.95,
                              keep_recent=(12, 8, 2))
        r = c.compress(msgs(20, chars=30))
        self.assertEqual(r.level, "full_compress")
        self.assertLess(r.tokens_after, r.tokens_before)

    def test_summary_message_present_after_compression(self):
        c = ContextCompressor(token_budget=200, keep_recent=(2, 2, 2))
        r = c.compress(msgs(10, chars=20))
        self.assertTrue(any(m.role == "summary" for m in r.messages))
        self.assertEqual(r.messages[0].role, "summary")

    def test_recent_messages_preserved(self):
        c = ContextCompressor(token_budget=200, keep_recent=(2, 2, 2))
        original = msgs(10, chars=20)
        r = c.compress(original)
        # 最后 2 条应原样保留
        self.assertEqual(r.messages[-1].content, original[-1].content)
        self.assertEqual(r.messages[-2].content, original[-2].content)

    def test_saved_ratio_positive(self):
        c = ContextCompressor(token_budget=200, keep_recent=(2, 2, 2))
        r = c.compress(msgs(10, chars=30))
        self.assertGreater(r.saved_ratio, 0.0)
        self.assertLessEqual(r.saved_ratio, 1.0)

    def test_custom_summarize_fn_used(self):
        c = ContextCompressor(token_budget=200, keep_recent=(2, 2, 2),
                              summarize_fn=lambda t: "自定义摘要")
        r = c.compress(msgs(10, chars=20))
        self.assertEqual(r.summary, "自定义摘要")
        self.assertIn("自定义摘要", r.messages[0].content)

    def test_summarize_fn_exception_falls_back(self):
        def boom(t):
            raise RuntimeError("摘要模型挂了")
        c = ContextCompressor(token_budget=200, keep_recent=(2, 2, 2), summarize_fn=boom)
        r = c.compress(msgs(10, chars=20))
        self.assertTrue(r.summary, "摘要失败必须降级而不是留空")
        self.assertTrue(any(m.role == "summary" for m in r.messages))

    def test_tool_messages_truncated(self):
        c = ContextCompressor(token_budget=100000, max_tool_chars=50)
        m = [Message("tool", "z" * 500)]
        r = c.compress(m)
        self.assertLess(len(r.messages[0].content), 500)
        self.assertIn("已截断", r.messages[0].content)

    def test_tool_truncation_happens_even_below_threshold(self):
        """★ 关键：工具结果截断是无条件的第一步压缩，不依赖阈值。"""
        c = ContextCompressor(token_budget=10**6, max_tool_chars=50)
        r = c.compress([Message("tool", "z" * 500)])
        self.assertEqual(r.level, "none")
        self.assertLess(len(r.messages[0].content), 500)

    def test_fewer_messages_than_keep_is_safe(self):
        c = ContextCompressor(token_budget=1, keep_recent=(10, 10, 10))
        r = c.compress(msgs(3, chars=20))
        self.assertEqual(len(r.messages), 3)

    def test_extra_tokens_counted(self):
        c = ContextCompressor(token_budget=200, keep_recent=(2, 2, 2))
        r = c.compress(msgs(4, chars=10), extra_tokens=500)
        self.assertGreater(r.ratio, 1.0)

    def test_should_compress_flag(self):
        c = ContextCompressor(token_budget=100)
        self.assertFalse(c.should_compress(msgs(1, chars=5)))
        self.assertTrue(c.should_compress(msgs(10, chars=40)))

    def test_ratio_monotonic_with_budget(self):
        big = ContextCompressor(token_budget=100000).compress(msgs(5, chars=20))
        small = ContextCompressor(token_budget=200).compress(msgs(5, chars=20))
        self.assertLess(big.ratio, small.ratio)

    def test_no_message_loss_of_recent_content(self):
        """压缩后最近的内容必须完整可取到（否则答案会缺上下文）。"""
        c = ContextCompressor(token_budget=250, keep_recent=(3, 2, 2))
        original = msgs(9, chars=15)
        r = c.compress(original)
        joined = "".join(m.content for m in r.messages)
        for m in original[-2:]:
            self.assertIn(m.content, joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
