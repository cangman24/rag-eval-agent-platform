# -*- coding: utf-8 -*-
"""tests/test_store.py —— 对话持久化与 SSE 折叠（零依赖）。

覆盖两件容易出错的事：
  1. 一次问答必须原子地落成 user + assistant + n 条 citation，
     刷新页面重建的"出处卡片"完全依赖 citations 表；
  2. 拒答场景绝不能留下引用（否则又回到"未找到 + 3 条来源"的自相矛盾）。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sse import TurnFolder, fold_turn, format_sse
from core.store import ChatStore


def _citation(i, chunk_id, doc_id=None):
    return format_sse(
        {"index": i, "chunk_id": chunk_id, "doc_id": doc_id or chunk_id.split("#")[0],
         "score": 1.0 - i / 10, "snippet": f"片段{i}"},
        event="citation",
    )


class TestChatStore(unittest.TestCase):
    def setUp(self):
        self.store = ChatStore(":memory:")

    def tearDown(self):
        self.store.close()

    # ---------------------------------------------------------- 会话
    def test_create_and_get_empty_conversation(self):
        cid = self.store.create_conversation("测试会话")
        conv = self.store.get_conversation(cid)
        self.assertIsNotNone(conv)
        self.assertEqual(conv["title"], "测试会话")
        self.assertEqual(conv["messages"], [])
        self.assertEqual(conv["message_count"], 0)

    def test_ensure_conversation_is_idempotent(self):
        cid = self.store.ensure_conversation(None, title="第一个问题")
        again = self.store.ensure_conversation(cid)
        self.assertEqual(cid, again)
        self.assertEqual(len(self.store.list_conversations()), 1)

    def test_ensure_conversation_recreates_missing_id(self):
        # 前端 localStorage 里存了 id，但库被清过 —— 不能 404，要能续上
        cid = self.store.ensure_conversation("c_deadbeef0000")
        self.assertEqual(cid, "c_deadbeef0000")
        self.assertIsNotNone(self.store.get_conversation(cid))

    def test_title_falls_back_to_first_user_question(self):
        cid = self.store.create_conversation("")
        self.store.append_message(cid, "user", "倒伏面积评估的精度要求是什么")
        self.store.append_message(cid, "assistant", "不低于 3 厘米")
        self.assertEqual(self.store.get_conversation(cid)["title"],
                         "倒伏面积评估的精度要求是什么")

    def test_message_counts_and_preview(self):
        cid = self.store.create_conversation("")
        self.store.append_message(cid, "user", "问一")
        self.store.append_message(cid, "assistant", "答一")
        item = self.store.list_conversations()[0]
        self.assertEqual(item["message_count"], 2)
        self.assertEqual(item["preview"], "答一")
        self.assertTrue(item["has_messages"])

    def test_list_orders_by_recent_update(self):
        a = self.store.create_conversation("A")
        b = self.store.create_conversation("B")
        self.store.append_message(a, "user", "让 A 变成最近")
        self.assertEqual([x["id"] for x in self.store.list_conversations()], [a, b])

    def test_delete_cascades_messages_and_citations(self):
        cid = self.store.create_conversation("删我")
        self.store.append_message(cid, "user", "q")
        self.store.append_message(cid, "assistant", "a",
                                  citations=[{"index": 1, "chunk_id": "d.md#p0#c0",
                                              "snippet": "x"}])
        self.assertTrue(self.store.delete_conversation(cid))
        self.assertIsNone(self.store.get_conversation(cid))
        self.assertEqual(self.store.stats()["citations"], 0)
        self.assertEqual(self.store.stats()["messages"], 0)

    # ---------------------------------------------------------- 消息与引用
    def test_roundtrip_preserves_answer_and_citations(self):
        cid = self.store.create_conversation("")
        self.store.append_message(cid, "user", "精度要求")
        self.store.append_message(
            cid, "assistant", "不低于 3 厘米 [1]",
            latency_ms=520.0, request_id="r1",
            citations=[
                {"index": 1, "chunk_id": "app_倒伏面积评估.md#p0#c0",
                 "doc_id": "app_倒伏面积评估.md", "score": 0.91, "snippet": "第五条"},
                {"index": 2, "chunk_id": "process_精度评价.md#p0#c0",
                 "doc_id": "process_精度评价.md", "score": 0.88, "snippet": "第二条"},
            ],
            meta={"engine": "rag"},
        )
        msgs = self.store.get_conversation(cid)["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["content"], "不低于 3 厘米 [1]")
        self.assertEqual(msgs[1]["latency_ms"], 520.0)
        self.assertEqual(len(msgs[1]["citations"]), 2)
        self.assertEqual(msgs[1]["citations"][0]["chunk_id"], "app_倒伏面积评估.md#p0#c0")
        self.assertEqual(msgs[1]["citations"][1]["doc_id"], "process_精度评价.md")
        self.assertEqual(msgs[1]["meta"]["engine"], "rag")

    def test_refused_flag_survives_roundtrip(self):
        cid = self.store.create_conversation("")
        self.store.append_message(cid, "assistant", "知识库中未找到相关内容。",
                                  refused=True)
        self.assertTrue(self.store.get_conversation(cid)["messages"][0]["refused"])

    def test_citation_index_is_primary_key(self):
        # 同一 message 重复写同一个 idx 不应炸（INSERT OR REPLACE）
        cid = self.store.create_conversation("")
        mid = self.store.append_message(cid, "user", "q")
        self.store.append_message(cid, "assistant", "a", citations=[
            {"index": 1, "chunk_id": "a.md#p0#c0"},
            {"index": 1, "chunk_id": "a.md#p0#c0"},
        ])
        self.assertEqual(len(self.store.get_conversation(cid)["messages"][1]["citations"]), 1)

    def test_stats_aggregates_docs(self):
        cid = self.store.create_conversation("")
        self.store.append_message(cid, "assistant", "a", citations=[
            {"index": 1, "chunk_id": "x.md#p0#c0", "doc_id": "x.md"},
            {"index": 2, "chunk_id": "x.md#p1#c0", "doc_id": "x.md"},
            {"index": 3, "chunk_id": "y.md#p0#c0", "doc_id": "y.md"},
        ])
        st = self.store.stats()
        self.assertEqual(st["citations"], 3)
        self.assertEqual(st["distinct_docs_cited"], 2)
        self.assertEqual(st["top_docs"][0], {"doc_id": "x.md", "n": 2})

    def test_persistence_across_reopen(self):
        """真正的持久化：关掉再打开同一个文件，数据还在（刷新页面的等价物）。"""
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "app.db")
        try:
            s1 = ChatStore(path)
            cid = s1.create_conversation("")
            s1.append_message(cid, "user", "问")
            s1.append_message(cid, "assistant", "答", citations=[
                {"index": 1, "chunk_id": "d.md#p0#c0"}])
            s1.close()

            s2 = ChatStore(path)
            conv = s2.get_conversation(cid)
            self.assertEqual(conv["messages"][0]["content"], "问")
            self.assertEqual(conv["messages"][1]["citations"][0]["chunk_id"],
                             "d.md#p0#c0")
            s2.close()
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)


class TestTurnFolder(unittest.TestCase):
    """SSE 帧 -> 可落库消息的折叠。"""

    def test_normal_turn(self):
        frames = [
            format_sse({"query": "q"}, event="start"),
            format_sse({"count": 5, "top_score": 1.2}, event="retrieval"),
            _citation(1, "a.md#p0#c0"),
            _citation(2, "b.md#p0#c1"),
            format_sse({"text": "答案 [1][2]", "citations": ["a.md#p0#c0", "b.md#p0#c1"]},
                       event="answer"),
            format_sse({"latency_ms": 512.5}, event="done"),
        ]
        turn = fold_turn(frames)
        self.assertEqual(turn["answer"], "答案 [1][2]")
        self.assertEqual(len(turn["citations"]), 2)
        self.assertFalse(turn["refused"])
        self.assertEqual(turn["latency_ms"], 512.5)
        self.assertIsNone(turn["error"])

    def test_refusal_drops_citations(self):
        """拒答场景即使前面漏发过 citation，折叠后也必须为空。"""
        frames = [
            _citation(1, "a.md#p0#c0"),
            format_sse({"text": "知识库中未找到相关内容。"}, event="refusal"),
            format_sse({"latency_ms": 90}, event="done"),
        ]
        turn = fold_turn(frames)
        self.assertTrue(turn["refused"])
        self.assertEqual(turn["citations"], [])
        self.assertIn("未找到", turn["answer"])

    def test_error_is_recorded_not_raised(self):
        frames = [
            format_sse({"error": "llm_failed: 429"}, event="error"),
            format_sse({"latency_ms": 30}, event="done"),
        ]
        turn = fold_turn(frames)
        self.assertEqual(turn["error"], "llm_failed: 429")
        self.assertEqual(turn["answer"], "")

    def test_malformed_frames_are_ignored(self):
        turn = fold_turn(["event: answer\ndata: {坏json\n\n", "随便一段垃圾\n\n"])
        self.assertEqual(turn["answer"], "")

    def test_incremental_feed_matches_batch(self):
        """流式端点用 TurnFolder 逐帧喂；结果必须与一次性 fold_turn 完全一致。"""
        frames = [
            _citation(1, "a.md#p0#c0"),
            format_sse({"text": "答案"}, event="answer"),
            format_sse({"latency_ms": 100.0}, event="done"),
        ]
        f = TurnFolder()
        for fr in frames:
            f.feed(fr)
        self.assertEqual(f.result(), fold_turn(frames))

    def test_empty_stream(self):
        turn = fold_turn([])
        self.assertEqual(turn["answer"], "")
        self.assertEqual(turn["citations"], [])
        self.assertIsNone(turn["latency_ms"])


if __name__ == "__main__":
    unittest.main()
