# -*- coding: utf-8 -*-
"""服务层 token 级流式测试（离线）。

这里锁的是 M3 最容易搞坏的两件事：
  ① **拒答不发引用**（test_stream_refusal_path 那条契约在流式下仍然成立）；
  ② **前缀缓冲不能丢字**（攒着不发的那几个字符，放行时必须整段补发）。
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import LLMDelta  # noqa: E402
from core.service import RagService, ServiceConfig  # noqa: E402
from core.sse import parse_sse  # noqa: E402


# ================================================================== 夹具
# ★ 刻意**不** import tests/test_service.py：两个拆分仓的 test_service.py
#   内容并不完全相同（项目一仓那版没有 FakeLLM），跨文件依赖会让同一个
#   测试在一个仓绿、在另一个仓 ImportError。替身就地定义，比复用更稳。
DOCS = {
    "c1": "父块一：检测精度要求 mAP 不低于 0.90。",
    "c2": "父块二：训练集按 7:1:2 划分。",
}
METAS = {
    "c1": {"heading_path": "第二章 > 精度要求"},
    "c2": {"heading_path": "第二章 > 数据集要求"},
}


def doc_lookup(cid):
    return DOCS.get(cid, "")


def meta_lookup(cid):
    return METAS.get(cid, {})


class FakeRetriever:
    def __init__(self, table=None, scores=None):
        self.table = table if table is not None else {"精度": ["c1", "c2"]}
        self.scores = scores or {}
        self.calls = 0

    def retrieve(self, query, cfg=None):
        self.calls += 1
        ids = []
        for key, val in self.table.items():
            if key in query or query in key:
                for i in val:
                    if i not in ids:
                        ids.append(i)
        return {"doc_ids": ids,
                "scores": [self.scores.get(i, 0.9) for i in ids]}


class WholeFakeLLM:
    """整段式假 LLM（没有 can_stream），用来验证回退路径。"""

    def __init__(self, text="整段答案 [1]"):
        self.text = text
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1

        class R:
            pass

        r = R()
        r.text = self.text
        return r


def make_rag(llm=None, table=None, scores=None, **cfg_kw):
    return RagService(
        retriever=FakeRetriever(table, scores),
        doc_lookup=doc_lookup, llm=llm, meta_lookup=meta_lookup,
        config=ServiceConfig(**cfg_kw),
    )


# ================================================================== 夹具
class StreamingFakeLLM:
    """自报能流式的假 LLM。

    刻意让 chat() 直接抛异常：流式路径一旦退回整段式，测试立刻失败，
    而不是"看起来也能跑"。
    """

    def __init__(self, parts, fail_after=None, first_ms=12.5):
        self.parts = list(parts)
        self.fail_after = fail_after
        self.first_ms = first_ms
        self.stream_calls = 0
        self.chat_calls = 0

    def can_stream(self) -> bool:
        return True

    def chat(self, messages, **kw):  # pragma: no cover - 不该被调用
        self.chat_calls += 1
        raise AssertionError("流式路径不该调用整段式 chat()")

    def chat_stream(self, messages, **kw):
        self.stream_calls += 1
        for i, p in enumerate(self.parts):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("DeepSeek 503")
            yield LLMDelta(text=p, first_token_ms=self.first_ms + i)
        yield LLMDelta(done=True, full_text="".join(self.parts),
                       prompt_tokens=11, completion_tokens=len(self.parts),
                       first_token_ms=self.first_ms, latency_ms=100.0)


def events(frames):
    """把 SSE 帧变成 [(event, data)]，跳过注释帧。"""
    out = []
    for f in frames:
        for ev in parse_sse(f):
            if "comment" in ev:
                continue
            raw = ev.get("data") or "{}"
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
            out.append((ev.get("event"), data))
    return out


def kinds(evs):
    return [k for k, _ in evs]


def tokens(evs):
    return [d.get("text", "") for k, d in evs if k == "token"]


def make_streaming_rag(parts, **kw):
    llm = StreamingFakeLLM(parts, **kw)
    svc = make_rag(llm=llm)
    return svc, llm


# ================================================================== 正常作答
class TokenStreamTest(unittest.TestCase):
    def test_streams_token_events(self):
        svc, llm = make_streaming_rag(["精度要求", "是 mAP≥0.90", " [1]"])
        evs = events(svc.answer_stream("精度要求是什么"))
        self.assertIn("token", kinds(evs))
        self.assertIn("answer", kinds(evs))
        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(llm.chat_calls, 0)

    def test_answer_event_carries_full_text(self):
        """token 是增量，answer 是全文 —— 两者都要对。"""
        parts = ["精度要求", "是 mAP≥0.90", " [1]"]
        svc, _ = make_streaming_rag(parts)
        evs = events(svc.answer_stream("精度要求是什么"))
        ans = [d for k, d in evs if k == "answer"][0]
        self.assertEqual(ans["text"], "".join(parts))
        self.assertEqual(ans["citations"], ["c1", "c2"])

    def test_no_character_lost_across_buffer(self):
        """★ 前缀缓冲期攒下的字符，放行时必须整段补发，不能只发当前 token。"""
        parts = ["知识", "库中", "有答案", "，见[1]"]
        svc, _ = make_streaming_rag(parts)
        evs = events(svc.answer_stream("精度要求是什么"))
        joined = "".join(tokens(evs))
        self.assertEqual(joined, "".join(parts))

    def test_buffer_delays_but_does_not_drop(self):
        """前几个 token 会被缓冲（因为仍可能是拒答话术的开头）。"""
        parts = ["知识", "库中", "有答案"]
        svc, _ = make_streaming_rag(parts)
        evs = events(svc.answer_stream("精度要求是什么"))
        # 三个 token 里前两个不发，第三个一次性把缓冲补出去
        self.assertEqual(len(tokens(evs)), 1)
        self.assertEqual(tokens(evs)[0], "知识库中有答案")

    def test_citation_precedes_first_token(self):
        """★ 引用必须在第一个 token 之前到达，否则前端会先看到答案再看到出处。"""
        svc, _ = make_streaming_rag(["这是答案 [1]"])
        evs = events(svc.answer_stream("精度要求是什么"))
        seq = kinds(evs)
        first_token = seq.index("token")
        cite_idx = [i for i, k in enumerate(seq) if k == "citation"]
        self.assertTrue(cite_idx, "没有发出 citation 事件")
        self.assertTrue(all(i < first_token for i in cite_idx),
                        f"citation 位置 {cite_idx} 应全部在首个 token {first_token} 之前")

    def test_citation_emitted_only_once(self):
        svc, _ = make_streaming_rag(["答案一", "答案二", "答案三"])
        evs = events(svc.answer_stream("精度要求是什么"))
        cits = [d for k, d in evs if k == "citation"]
        self.assertEqual(len(cits), 2)          # c1 + c2，各一次
        self.assertEqual(len({c["chunk_id"] for c in cits}), 2)

    def test_done_carries_first_token_latency(self):
        svc, _ = make_streaming_rag(["答案"])
        evs = events(svc.answer_stream("精度要求是什么"))
        done = [d for k, d in evs if k == "done"][0]
        self.assertIsNotNone(done.get("first_token_ms"))
        self.assertGreater(done["first_token_ms"], 0)
        self.assertIsNotNone(done.get("latency_ms"))

    def test_start_and_retrieval_still_first(self):
        """事件前缀顺序不能变：start -> retrieval -> ..."""
        svc, _ = make_streaming_rag(["答案"])
        seq = kinds(events(svc.answer_stream("精度要求是什么")))
        self.assertEqual(seq[0], "start")
        self.assertEqual(seq[1], "retrieval")

    def test_retrieval_happens_once(self):
        """★ 顺手修掉的旧 bug：answer_stream 曾检索两次（自己一次 + answer 内部一次）。"""
        llm = StreamingFakeLLM(["答案"])
        svc = make_rag(llm=llm)
        list(svc.answer_stream("精度要求是什么"))
        self.assertEqual(svc.retriever.calls, 1)


# ================================================================== 拒答
class StreamRefusalTest(unittest.TestCase):
    def test_refusal_emits_no_citation(self):
        """★ 核心契约：拒答时一条引用都不能发（流式下同样成立）。"""
        svc, _ = make_streaming_rag(["知识", "库中未找到", "相关内容。"])
        evs = events(svc.answer_stream("库里没有的问题"))
        self.assertIn("refusal", kinds(evs))
        self.assertNotIn("citation", kinds(evs))
        self.assertNotIn("token", kinds(evs))
        self.assertNotIn("answer", kinds(evs))

    def test_refusal_text_is_complete(self):
        svc, _ = make_streaming_rag(["知识", "库中未找到", "相关内容。"])
        evs = events(svc.answer_stream("库里没有的问题"))
        ref = [d for k, d in evs if k == "refusal"][0]
        self.assertEqual(ref["text"], "知识库中未找到相关内容。")

    def test_refusal_does_not_release_citation_early(self):
        """前半段像拒答、后半段改口 —— 按"已命中拒答"处理，不补发引用。

        宁可漏一次引用，也不能出现"未找到 + 3 条来源"这种自相矛盾。
        """
        svc, _ = make_streaming_rag(["知识库中未找到", "，但片段 [1] 提到 0.9"])
        evs = events(svc.answer_stream("问题"))
        self.assertIn("refusal", kinds(evs))
        self.assertNotIn("citation", kinds(evs))

    def test_low_score_refusal_skips_llm(self):
        """检索层就能定的低置信拒答，不该去调 LLM（省钱也省时）。"""
        llm = StreamingFakeLLM(["不该被调用"])
        svc = make_rag(llm=llm, table={"精度": ["c1"]},
                       scores={"c1": 0.01})
        evs = events(svc.answer_stream("精度要求"))
        self.assertIn("refusal", kinds(evs))
        self.assertNotIn("citation", kinds(evs))
        self.assertEqual(llm.stream_calls, 0)


# ================================================================== 降级
class StreamFallbackTest(unittest.TestCase):
    def test_non_streaming_llm_falls_back_to_whole_answer(self):
        """注入的整段式 fake LLM 没有 can_stream -> 走老路径，不发 token 事件。"""
        svc = make_rag(llm=WholeFakeLLM("整段答案 [1]"))
        evs = events(svc.answer_stream("精度要求是什么"))
        self.assertNotIn("token", kinds(evs))
        self.assertIn("answer", kinds(evs))
        self.assertIn("citation", kinds(evs))

    def test_llm_failure_before_release_gives_retrieval_fallback(self):
        """一个字还没出去就挂了 -> 可以安全兜底成检索片段直出。"""
        svc, _ = make_streaming_rag(["永远不会到"], fail_after=0)
        evs = events(svc.answer_stream("精度要求是什么"))
        self.assertIn("error", kinds(evs))
        texts = tokens(evs)
        self.assertTrue(texts)
        self.assertIn("检索结果", texts[0])
        # 兜底也必须有引用，否则用户看不到依据
        self.assertIn("citation", kinds(evs))

    def test_llm_failure_after_release_does_not_repeat_text(self):
        """已经吐过字再挂：只补 error，不能把答案重发一遍。"""
        svc, _ = make_streaming_rag(["知识库中有答案 [1]", "第二段"],
                                    fail_after=1)
        evs = events(svc.answer_stream("精度要求是什么"))
        self.assertIn("error", kinds(evs))
        joined = "".join(tokens(evs))
        # 只应包含真正的增量，不含兜底的"检索结果"
        self.assertNotIn("检索结果", joined)

    def test_no_llm_service_uses_old_path(self):
        svc = make_rag(llm=None)
        evs = events(svc.answer_stream("精度要求是什么"))
        self.assertNotIn("token", kinds(evs))
        self.assertIn("answer", kinds(evs))


# ================================================================== 兼容
class BackwardCompatTest(unittest.TestCase):
    def test_max_answer_chars_applied_to_stream(self):
        """流式也要受 max_answer_chars 约束，否则长答案会突破上限。"""
        svc = make_rag(llm=StreamingFakeLLM(["一二三四五", "六七八九十"]),
                       max_answer_chars=4)
        evs = events(svc.answer_stream("精度要求是什么"))
        ans = [d for k, d in evs if k == "answer"][0]
        self.assertEqual(ans["text"], "一二三四")


if __name__ == "__main__":
    unittest.main()
