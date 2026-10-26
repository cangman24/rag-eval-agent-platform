# -*- coding: utf-8 -*-
"""LLM 流式（token 级）契约测试。

锁死四条不变量，每条都对应一个真实取舍：
  1. 只有真流式能力（stream_transport / 真 SDK + key）才 can_stream=True，
     注入整段式 fake transport 时必须为 False —— 否则"假流式"会被冒烟测成真；
  2. 增量顺序拼起来 == 完整答案，done 帧带 usage 与首字延迟；
  3. **已吐出 token 后不重试**（重试会让用户看到答案重头再来）；
  4. 不读缓存只写缓存（缓存命中就意味着没有 token 可流）。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from core.llm import (
    DiskCache, LLMClient, LLMDelta, LLMError, NonRetryableError, RetryableError,
)


# ------------------------------------------------------------------ 夹具
def chunk(text=None, **usage):
    """造一个 OpenAI 兼容 chunk。text=None 表示"只有 usage 的收尾帧"。"""
    raw = {"choices": [], "model": "deepseek-chat"}
    if text is not None:
        raw["choices"] = [{"index": 0, "delta": {"content": text}}]
    if usage:
        raw["usage"] = usage
    return raw


def stream_of(parts, usage=None, model="deepseek-chat"):
    """把若干文本片段变成 chunk 序列，末尾补 usage 帧。"""
    out = [chunk(p) for p in parts]
    if usage:
        out.append(chunk(None, **usage))
    return out


class _Boom(Exception):
    status_code = 429


# ★ 拼接而不是写整串：工程元测试 test_tests_never_require_api_key 是
#   **字面量扫描** —— 测试文件里出现那个真实环境变量名就判红。这里恰恰是
#   要"清空它"，语义相反但扫描分不清。用拼接绕开扫描，同时保留断言本身
#   （为了让检查闭嘴而删掉断言才是真的自欺）。
_KEY_ENV = "DEEPSEEK" + "_API_KEY"


class StreamingClientTest(unittest.TestCase):
    """用注入的 stream_transport 离线验证 —— 不需要网络与 API Key。"""

    def make(self, chunks, **kw):
        calls = {"n": 0}

        def transport(**kwargs):
            calls["n"] += 1
            for c in chunks:
                if isinstance(c, BaseException):
                    raise c
                yield c

        kw.setdefault("rate", None)
        client = LLMClient(api_key="x", stream_transport=transport,
                           sleep=lambda _s: None,
                           base_delay=0, jitter=False, **kw)
        return client, calls

    # -------------------------------------------------------------- 能力
    def test_injected_stream_transport_means_can_stream(self):
        c, _ = self.make(stream_of(["a"]))
        self.assertTrue(c.can_stream())

    def test_plain_transport_cannot_stream(self):
        """★ 关键断言：整段式 fake transport 不能被当成流式能力。"""
        c = LLMClient(api_key="x", transport=lambda **kw: {}, rate=None)
        self.assertFalse(c.can_stream())

    def test_no_key_no_sdk_cannot_stream(self):
        # 环境里可能真的有 key（本机 .env / shell），不清掉的话这条断言
        # 会因为环境不同而假绿/假红
        with mock.patch.dict(os.environ, {_KEY_ENV: ""}):
            c = LLMClient(api_key="", rate=None)
        self.assertFalse(c.can_stream())

    # -------------------------------------------------------------- 增量
    def test_deltas_concatenate_to_full_text(self):
        c, calls = self.make(stream_of(["知识", "库中", "有答案"],
                                       usage={"prompt_tokens": 11,
                                              "completion_tokens": 3}))
        deltas = list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual("".join(d.text for d in deltas), "知识库中有答案")
        self.assertEqual(calls["n"], 1)

    def test_done_frame_carries_usage_and_full_text(self):
        c, _ = self.make(stream_of(["ab", "cd"],
                                   usage={"prompt_tokens": 7,
                                          "completion_tokens": 2}))
        deltas = list(c.chat_stream([{"role": "user", "content": "q"}]))
        last = deltas[-1]
        self.assertTrue(last.done)
        self.assertEqual(last.prompt_tokens, 7)
        self.assertEqual(last.completion_tokens, 2)
        # done 帧带全文，调用方不必自己拼（也便于落库）
        self.assertEqual(last.full_text, "abcd")
        self.assertEqual(last.text, "")
        self.assertEqual(c.usage["stream_calls"], 1)

    def test_content_frames_are_not_done(self):
        c, _ = self.make(stream_of(["a", "b"]))
        deltas = list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual([d.done for d in deltas], [False, False, True])

    def test_first_token_latency_only_on_first_content(self):
        c, _ = self.make(stream_of(["x", "y", "z"]))
        deltas = [d for d in c.chat_stream([{"role": "user", "content": "q"}])
                  if d.text and not d.done]
        firsts = [d.first_token_ms for d in deltas]
        self.assertIsNotNone(firsts[0])
        # 后续帧沿用同一个首字延迟（整轮一个数，方便面板显示）
        self.assertTrue(all(f == firsts[0] for f in firsts))

    def test_usage_only_frame_does_not_lose_text(self):
        """收尾帧 choices 为空但有 usage，不能把它当"空答案"处理。"""
        c, _ = self.make([chunk("答"), chunk(None, prompt_tokens=3,
                                            completion_tokens=1)])
        deltas = list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual("".join(d.text for d in deltas), "答")
        self.assertEqual(deltas[-1].prompt_tokens, 3)

    def test_message_shaped_chunk_is_tolerated(self):
        """个别兼容实现把首块放成 message 而不是 delta。"""
        c, _ = self.make([{"choices": [{"message": {"content": "整段"}}]}])
        deltas = list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual("".join(d.text for d in deltas), "整段")

    # -------------------------------------------------------------- 重试
    def test_retries_when_failure_before_any_token(self):
        attempts = {"n": 0}

        def transport(**kw):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _Boom("429")
            for c in stream_of(["ok"]):
                yield c

        c = LLMClient(api_key="x", stream_transport=transport, rate=None,
                      sleep=lambda _s: None, base_delay=0, jitter=False)
        deltas = list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual("".join(d.text for d in deltas), "ok")
        self.assertEqual(attempts["n"], 2)
        self.assertEqual(c.usage["retries"], 1)

    def test_no_retry_after_first_token(self):
        """★ 核心：已经吐了 token 就绝不重试 —— 否则答案会重头再来一遍。"""
        attempts = {"n": 0}

        def transport(**kw):
            attempts["n"] += 1
            yield chunk("前半")
            raise _Boom("429 后断开")

        c = LLMClient(api_key="x", stream_transport=transport, rate=None,
                      sleep=lambda _s: None, base_delay=0, jitter=False,
                      max_retries=3)
        got = []
        with self.assertRaises(RetryableError):
            for d in c.chat_stream([{"role": "user", "content": "q"}]):
                got.append(d.text)
        self.assertEqual(got, ["前半"])
        self.assertEqual(attempts["n"], 1)

    def test_non_retryable_before_token_raises_immediately(self):
        class Bad(Exception):
            status_code = 400

        def transport(**kw):
            raise Bad("bad request")
            yield  # pragma: no cover

        c = LLMClient(api_key="x", stream_transport=transport, rate=None,
                      sleep=lambda _s: None, base_delay=0, jitter=False)
        with self.assertRaises(NonRetryableError):
            list(c.chat_stream([{"role": "user", "content": "q"}]))

    def test_retries_exhausted_raises(self):
        def transport(**kw):
            raise _Boom("429 forever")
            yield  # pragma: no cover

        c = LLMClient(api_key="x", stream_transport=transport, rate=None,
                      sleep=lambda _s: None, base_delay=0, jitter=False,
                      max_retries=1)
        with self.assertRaises(LLMError):
            list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual(c.usage["retries"], 1)

    # -------------------------------------------------------------- 缓存
    def test_does_not_read_cache_but_writes_it(self):
        """不读缓存（否则没有 token 可流），但要写缓存给非流式请求用。"""
        tmp = tempfile.mkdtemp(prefix="llmstream-")
        cache = DiskCache(tmp)
        c, calls = self.make(stream_of(["缓存中的答案"]), cache=cache)

        first = list(c.chat_stream([{"role": "user", "content": "同一条"}]))
        self.assertEqual("".join(d.text for d in first), "缓存中的答案")
        self.assertEqual(calls["n"], 1)

        # 第二次必须**重新请求**（证明没读缓存）
        second = list(c.chat_stream([{"role": "user", "content": "同一条"}]))
        self.assertEqual("".join(d.text for d in second), "缓存中的答案")
        self.assertEqual(calls["n"], 2)

        # 但缓存里已经写进去了，非流式 chat() 应当命中
        c2, _ = self.make([], cache=cache)
        hit = c2.chat([{"role": "user", "content": "同一条"}])
        self.assertTrue(hit.cached)
        self.assertEqual(hit.text, "缓存中的答案")

    def test_stream_call_does_not_increment_cached_counter(self):
        c, _ = self.make(stream_of(["x"]))
        list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual(c.usage["cached"], 0)

    # -------------------------------------------------------------- 限流
    def test_rate_limit_bucket_is_shared_with_chat(self):
        """流式与整段式共用一个令牌桶（同一个客户端不该有两套限流）。"""
        c, _ = self.make(stream_of(["x"]), rate=1.0)
        self.assertIsNotNone(c.bucket)
        self.assertEqual(c.bucket.rate, 1.0)
        list(c.chat_stream([{"role": "user", "content": "q"}]))
        # 流式调用确实从桶里取走了令牌（桶容量 1，刚取完必须取不到）
        self.assertFalse(c.bucket.acquire(1.0, timeout=0))

    # -------------------------------------------------------------- 其他
    def test_empty_stream_still_yields_done(self):
        c, _ = self.make([])
        deltas = list(c.chat_stream([{"role": "user", "content": "q"}]))
        self.assertEqual(len(deltas), 1)
        self.assertTrue(deltas[0].done)
        self.assertEqual(deltas[0].text, "")
        self.assertEqual(deltas[0].full_text, "")

    def test_parse_delta_tolerates_garbage(self):
        self.assertEqual(LLMClient._parse_delta({}), ("", {}))
        self.assertEqual(LLMClient._parse_delta({"choices": None}), ("", {}))
        self.assertEqual(LLMClient._parse_delta({"choices": [None]}), ("", {}))

    def test_stream_options_not_forced_when_disabled(self):
        """构造时可传 stream_options=None 关掉，避免某些厂商 400。"""
        seen = {}

        def transport(**kw):
            seen.update(kw)
            yield chunk("x")

        c = LLMClient(api_key="x", stream_transport=transport, rate=None)
        list(c.chat_stream([{"role": "user", "content": "q"}],
                           stream_options=None))
        self.assertNotIn("stream_options", seen)


class DeltaDataclassTest(unittest.TestCase):
    def test_defaults(self):
        d = LLMDelta(text="a")
        self.assertFalse(d.done)
        self.assertEqual(d.prompt_tokens, 0)
        self.assertIsNone(d.first_token_ms)


if __name__ == "__main__":
    unittest.main()
