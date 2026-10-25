# -*- coding: utf-8 -*-
"""tests/test_service.py —— API 服务层单测。全部离线。

★ 这一层是"框架无关"的，所以即使沙箱里没有 fastapi，业务逻辑也能全量验证。
  FastAPI 的接线在 rag1_knowledge_qa/api_fastapi.py，只是薄薄一层。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import AgentEngine, InMemoryCheckpointer  # noqa: E402
from core.mcp import ToolRegistry  # noqa: E402
from core.service import (  # noqa: E402
    AgentService, AnswerResult, Citation, RagService, ServiceConfig, build_citations,
    dispatch, new_request_id,
)
from core.sse import parse_sse  # noqa: E402


# ================================================================== 假依赖
class FakeRetriever:
    """可控的检索器。

    ★ 用**关键词包含**匹配而非精确匹配：真实的检索器对 "精度要求"
      也会召回含"精度"的文档。用精确匹配会让测试意外走进拒答分支——
      我最初就踩了这个坑（query='精度要求' 而表里只有 '精度'，
      结果 hits 为空、测试断言失败，排查了好一会儿）。
    """

    def __init__(self, table=None, score_map=None, raise_on=None):
        self.table = table or {}
        self.score_map = score_map or {}
        self.raise_on = raise_on
        self.calls = 0

    def _match(self, query: str):
        ids: list = []
        for key, val in self.table.items():
            if key in query or query in key:
                for i in val:
                    if i not in ids:
                        ids.append(i)
        return ids

    def retrieve(self, query, cfg=None):
        self.calls += 1
        if self.raise_on and self.raise_on in query:
            raise RuntimeError("向量库连接失败")
        ids = self._match(query)
        return {"doc_ids": ids,
                "scores": [self.score_map.get(i, 0.9) for i in ids]}


class FakeLLM:
    def __init__(self, text="这是生成的答案 [1]", fail=False):
        self.text = text
        self.fail = fail
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("DeepSeek 503")
        class R:
            pass
        r = R()
        r.text = self.text
        return r


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


def make_rag(llm=None, table=None, scores=None, **cfg_kw):
    return RagService(
        retriever=FakeRetriever(table or {"精度": ["c1", "c2"]}, scores),
        doc_lookup=doc_lookup, llm=llm, meta_lookup=meta_lookup,
        config=ServiceConfig(**cfg_kw),
    )


# ================================================================== 基础
class TestHelpers(unittest.TestCase):
    def test_request_id_unique(self):
        self.assertNotEqual(new_request_id(), new_request_id())

    def test_request_id_length(self):
        self.assertEqual(len(new_request_id()), 12)

    def test_citation_to_dict(self):
        c = Citation(1, "c1", "路径", "片段")
        d = c.to_dict()
        self.assertEqual(d["index"], 1)
        self.assertEqual(d["chunk_id"], "c1")

    def test_build_citations_index_from_one(self):
        """★ 引用序号必须从 1 开始 —— 与 Prompt 里的 [1][2] 对应。"""
        cits = build_citations([{"chunk_id": "c1", "text": "甲", "score": 1.0},
                                {"chunk_id": "c2", "text": "乙", "score": 0.5}],
                               meta_lookup)
        self.assertEqual([c.index for c in cits], [1, 2])
        self.assertEqual(cits[0].heading_path, "第二章 > 精度要求")

    def test_build_citations_snippet_truncated(self):
        cits = build_citations([{"chunk_id": "c1", "text": "x" * 500}],
                               snippet_chars=50)
        self.assertLessEqual(len(cits[0].snippet), 50)

    def test_build_citations_without_meta(self):
        cits = build_citations([{"chunk_id": "c1", "text": "甲"}])
        self.assertEqual(cits[0].heading_path, "")

    def test_answer_result_to_dict(self):
        r = AnswerResult("q", "a", [Citation(1, "c1")], request_id="rid")
        d = r.to_dict()
        self.assertEqual(d["request_id"], "rid")
        self.assertEqual(len(d["citations"]), 1)

    def test_config_fingerprint(self):
        fp = ServiceConfig(top_k=7).fingerprint()
        self.assertEqual(fp["top_k"], 7)


# ================================================================== RAG
class TestRagServiceAnswer(unittest.TestCase):
    def test_happy_path_with_llm(self):
        llm = FakeLLM("精度要求 0.90 [1]")
        svc = make_rag(llm=llm)
        res = svc.answer("精度要求")
        self.assertFalse(res.refused)
        self.assertIn("0.90", res.answer)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(len(res.citations), 2)
        self.assertIsNone(res.error)

    def test_empty_query_rejected(self):
        svc = make_rag(llm=FakeLLM())
        res = svc.answer("   ")
        self.assertEqual(res.error, "empty_query")
        self.assertIn("不能为空", res.answer)

    def test_low_confidence_refuses(self):
        """★ 低置信拒答：宁可不答，也不能编。"""
        svc = make_rag(llm=FakeLLM("不该被调用"),
                       table={"冷门问题": ["c1"]},
                       scores={"c1": 0.05}, refusal_threshold=0.3)
        res = svc.answer("冷门问题")
        self.assertTrue(res.refused)
        self.assertIn("未找到", res.answer)
        self.assertEqual(res.citations, [])

    def test_no_hits_refuses(self):
        svc = make_rag(llm=FakeLLM(), table={"未知话题": []})
        res = svc.answer("未知话题")
        self.assertTrue(res.refused)

    def test_threshold_boundary_not_refused(self):
        svc = make_rag(llm=FakeLLM(), table={"边界": ["c1"]},
                       scores={"c1": 0.3}, refusal_threshold=0.3)
        self.assertFalse(svc.answer("边界").refused)

    def test_llm_refusal_marker_respected(self):
        """LLM 自己说没找到 -> 标记 refused 且清空引用（不能给假引用）。"""
        svc = make_rag(llm=FakeLLM("知识库中未找到相关内容。"))
        res = svc.answer("精度")
        self.assertTrue(res.refused)
        self.assertEqual(res.citations, [])

    def test_no_llm_degrades_to_snippets(self):
        """★ 没配 LLM 时不能崩，要降级成检索片段（本地冒烟必需）。"""
        svc = make_rag(llm=None)
        res = svc.answer("精度")
        self.assertIn("未配置 LLM", res.answer)
        self.assertIn("mAP", res.answer)

    def test_retrieval_exception_becomes_error(self):
        svc = make_rag(llm=FakeLLM(), )
        svc.retriever = FakeRetriever(raise_on="触发")
        res = svc.answer("触发异常")
        self.assertIn("retrieval_failed", res.error)
        self.assertIn("暂时不可用", res.answer)

    def test_llm_exception_degrades(self):
        """★ LLM 挂了要给出检索结果，而不是 500。"""
        svc = make_rag(llm=FakeLLM(fail=True))
        res = svc.answer("精度")
        self.assertIn("llm_failed", res.error)
        self.assertIn("生成服务异常", res.answer)
        self.assertIn("mAP", res.answer)

    def test_answer_truncated_to_max_chars(self):
        long_text = "答" * 5000
        svc = make_rag(llm=FakeLLM(long_text), max_answer_chars=100)
        self.assertLessEqual(len(svc.answer("精度").answer), 100)

    def test_request_id_passthrough(self):
        svc = make_rag(llm=FakeLLM())
        self.assertEqual(svc.answer("精度", request_id="fixed-id").request_id,
                         "fixed-id")

    def test_latency_recorded(self):
        svc = make_rag(llm=FakeLLM())
        self.assertGreaterEqual(svc.answer("精度").latency_ms, 0.0)

    def test_hits_return_parent_text(self):
        """★ 命中子块，但 hits 里的 text 应是父块正文（父子块的关键）。"""
        svc = make_rag(llm=FakeLLM())
        res = svc.answer("精度")
        self.assertTrue(res.hits)
        self.assertIn("父块一", res.hits[0]["text"])

    def test_top_k_respected(self):
        svc = make_rag(llm=FakeLLM(), table={"精度": ["c1", "c2", "c3"]}, top_k=1)
        self.assertEqual(len(svc.answer("精度").hits), 1)

    def test_callable_retriever_supported(self):
        svc = RagService(retriever=lambda q: {"doc_ids": ["c1"], "scores": [0.9]},
                         doc_lookup=doc_lookup, llm=FakeLLM())
        self.assertFalse(svc.answer("任意").refused)


class TestRagServiceStream(unittest.TestCase):
    def _events(self, frames):
        return [e for e in parse_sse(frames) if "event" in e]

    def test_stream_event_sequence(self):
        svc = make_rag(llm=FakeLLM("答案 [1]"))
        events = self._events("".join(svc.answer_stream("精度")))
        types = [e["event"] for e in events]
        self.assertEqual(types[0], "start")
        self.assertIn("retrieval", types)
        self.assertEqual(types.count("citation"), 2)
        self.assertIn("answer", types)
        self.assertEqual(types[-1], "done")

    def test_stream_refusal_path(self):
        svc = make_rag(llm=FakeLLM(), table={"冷门": ["c1"]},
                       scores={"c1": 0.01})
        events = self._events("".join(svc.answer_stream("冷门")))
        types = [e["event"] for e in events]
        self.assertIn("refusal", types)
        self.assertNotIn("citation", types)

    def test_stream_ids_increment(self):
        svc = make_rag(llm=FakeLLM())
        events = self._events("".join(svc.answer_stream("精度")))
        self.assertEqual([e["id"] for e in events],
                         [str(i + 1) for i in range(len(events))])

    def test_stream_retrieval_failure_emits_error(self):
        svc = make_rag(llm=FakeLLM())
        svc.retriever = FakeRetriever(raise_on="炸")
        events = self._events("".join(svc.answer_stream("炸")))
        types = [e["event"] for e in events]
        self.assertIn("error", types)
        self.assertEqual(types[-1], "done")

    def test_stream_llm_failure_still_emits_answer(self):
        svc = make_rag(llm=FakeLLM(fail=True))
        events = self._events("".join(svc.answer_stream("精度")))
        types = [e["event"] for e in events]
        self.assertIn("answer", types)
        self.assertIn("error", types)


class TestRagServiceHealth(unittest.TestCase):
    def test_health_shape(self):
        svc = make_rag(llm=FakeLLM())
        h = svc.health()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(h["llm_configured"])
        self.assertIn("config", h)

    def test_health_without_llm(self):
        self.assertFalse(make_rag(llm=None).health()["llm_configured"])

    def test_health_reports_index_sizes(self):
        from core.bm25 import BM25Index
        from core.pipeline import HybridRetriever
        from core.vectorstore import InMemoryVectorStore

        store = InMemoryVectorStore()
        store.add(ids=["c1"], embeddings=[[1.0, 0.0]], documents=["甲"])
        bm25 = BM25Index()
        bm25.add("c1", "甲")
        bm25.finalize()
        retr = HybridRetriever(bm25=bm25, vector_store=store,
                               embed_fn=lambda t: [1.0, 0.0])
        svc = RagService(retriever=retr, doc_lookup=doc_lookup, llm=None)
        h = svc.health()
        self.assertEqual(h["index_size"], 1)
        self.assertEqual(h["bm25_size"], 1)


# ================================================================== Agent
def make_agent_service(**kw):
    reg = ToolRegistry()

    @reg.tool(description="检索知识库")
    def kb_search(query: str):
        """检索

        query: 检索词
        """
        return {"hits": [{"id": "c1", "text": "父块一"}]}

    @reg.tool(description="发邮件", sensitive=True)
    def send_mail(to: str):
        """发邮件

        to: 收件人
        """
        return {"sent": True}

    decisions = kw.pop("decisions", [
        {"tool": "kb_search", "args": {"query": "x"}},
        {"final": "答案 [1]", "citations": ["c1"]},
    ])
    cursor = {"i": 0}

    def act(state, allowed):
        d = decisions[min(cursor["i"], len(decisions) - 1)]
        cursor["i"] += 1
        return d

    engine = AgentEngine(
        registry=reg, intent_fn=kw.pop("intent_fn", lambda q: "kb"),
        plan_fn=lambda q, t: ["检索", "汇总"], act_fn=act,
        route_table={"kb": ["kb_search"], "general": []},
        checkpointer=kw.pop("checkpointer", InMemoryCheckpointer()),
        max_rounds=kw.pop("max_rounds", 6),
    )
    return AgentService(engine, kw.pop("config", None)), engine, reg


class TestAgentService(unittest.TestCase):
    def test_run_shape(self):
        svc, _, _ = make_agent_service()
        out = svc.run("查制度", thread_id="T1")
        self.assertEqual(out["thread_id"], "T1")
        self.assertEqual(out["answer"], "答案 [1]")
        self.assertTrue(out["finished"])
        self.assertEqual(out["tool_calls"], 1)
        self.assertIsNone(out["error"])

    def test_run_generates_thread_id(self):
        svc, _, _ = make_agent_service()
        self.assertTrue(svc.run("查制度")["thread_id"])

    def test_whitelist_blocks_sensitive_tool(self):
        svc, _, _ = make_agent_service(decisions=[
            {"tool": "send_mail", "args": {"to": "a@b.com"}},
            {"final": "无权限"},
        ])
        out = svc.run("查制度")
        self.assertTrue(out["finished"])
        self.assertEqual(out["tool_calls"], 0)

    def test_trace_after_run(self):
        svc, _, _ = make_agent_service()
        svc.run("查制度", thread_id="T2")
        tr = svc.trace("T2")
        self.assertTrue(tr["found"])
        self.assertEqual(tr["plan"], ["检索", "汇总"])
        self.assertTrue(tr["events"])
        self.assertTrue(tr["scratchpad"])

    def test_trace_unknown_thread(self):
        svc, _, _ = make_agent_service()
        tr = svc.trace("nope")
        self.assertFalse(tr["found"])
        self.assertEqual(tr["events"], [])

    def test_resume_continues(self):
        cp = InMemoryCheckpointer()
        # 第一段：1 轮就熔断
        svc1, _, _ = make_agent_service(checkpointer=cp, max_rounds=1,
                                        decisions=[{"tool": "kb_search",
                                                    "args": {"query": "x"}}])
        out1 = svc1.run("查制度", thread_id="T3")
        self.assertFalse(out1["finished"])

        # 第二段：同一 checkpointer 续跑
        svc2, _, _ = make_agent_service(checkpointer=cp, max_rounds=5,
                                        decisions=[{"final": "续跑答案"}])
        out2 = svc2.resume("T3")
        self.assertTrue(out2["finished"])
        self.assertEqual(out2["answer"], "续跑答案")

    def test_run_stream_events(self):
        svc, _, _ = make_agent_service()
        frames = "".join(svc.run_stream("查制度", thread_id="T4"))
        events = [e for e in parse_sse(frames) if "event" in e]
        types = [e["event"] for e in events]
        for expected in ("start", "intent", "plan", "tool_call",
                         "tool_result", "answer", "done"):
            self.assertIn(expected, types)

    def test_tools_listing(self):
        svc, _, _ = make_agent_service()
        info = svc.tools()
        names = sorted(t["name"] for t in info["tools"])
        self.assertEqual(names, ["kb_search", "send_mail"])
        self.assertIn("kb", info["routes"])

    def test_health_shape(self):
        svc, _, _ = make_agent_service()
        h = svc.health()
        self.assertEqual(h["tool_count"], 2)
        self.assertEqual(h["checkpointer"], "InMemoryCheckpointer")
        self.assertEqual(h["max_rounds"], 6)

    def test_health_without_registry(self):
        class Bare:
            pass
        self.assertEqual(AgentService(Bare()).health()["tool_count"], 0)

    def test_tools_without_registry(self):
        class Bare:
            pass
        self.assertEqual(AgentService(Bare()).tools(), {"tools": [], "routes": {}})


# ================================================================== dispatch
class TestDispatch(unittest.TestCase):
    def test_rag_answer(self):
        svc = make_rag(llm=FakeLLM("答案"))
        out = dispatch(svc, "answer", {"query": "精度"})
        self.assertIn("answer", out)
        self.assertIn("citations", out)

    def test_rag_health(self):
        self.assertEqual(dispatch(make_rag(), "health")["status"], "ok")

    def test_rag_unknown_action(self):
        self.assertIn("unknown_action", dispatch(make_rag(), "fly")["error"])

    def test_agent_run_and_trace(self):
        svc, _, _ = make_agent_service()
        out = dispatch(svc, "run", {"query": "查制度", "thread_id": "D1"})
        self.assertTrue(out["finished"])
        self.assertTrue(dispatch(svc, "trace", {"thread_id": "D1"})["found"])

    def test_agent_tools_and_health(self):
        svc, _, _ = make_agent_service()
        self.assertTrue(dispatch(svc, "tools")["tools"])
        self.assertEqual(dispatch(svc, "health")["status"], "ok")

    def test_agent_resume(self):
        cp = InMemoryCheckpointer()
        svc1, _, _ = make_agent_service(checkpointer=cp, max_rounds=1,
                                        decisions=[{"tool": "kb_search",
                                                    "args": {"query": "x"}}])
        dispatch(svc1, "run", {"query": "查制度", "thread_id": "D2"})
        svc2, _, _ = make_agent_service(checkpointer=cp, max_rounds=5,
                                        decisions=[{"final": "续跑"}])
        self.assertTrue(dispatch(svc2, "resume", {"thread_id": "D2"})["finished"])

    def test_unknown_service(self):
        self.assertIn("unknown_service", dispatch(object(), "run")["error"])

    def test_payload_defaults(self):
        svc = make_rag(llm=FakeLLM())
        out = dispatch(svc, "answer", None)
        self.assertEqual(out["error"], "empty_query")


# ================================================================== 集成
class TestServiceIntegration(unittest.TestCase):
    def test_rag_service_over_real_ingest(self):
        """★ 真链路：解析文档 -> 建库 -> 服务层问答 -> SSE。"""
        import shutil
        import uuid
        from core.ingest import IngestConfig, ingest_dir
        from core.pipeline import HybridRetriever
        from core.vectorstore import l2_normalize
        from core.text import tokenize

        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".tmp", uuid.uuid4().hex[:12])
        os.makedirs(tmp, exist_ok=True)

        def embed_batch(texts):
            out = []
            for t in texts:
                v = [0.0] * 32
                for tok in tokenize(t):
                    v[sum(map(ord, tok)) % 32] += 1.0
                out.append(l2_normalize(v))
            return out

        try:
            with open(os.path.join(tmp, "spec.md"), "w", encoding="utf-8") as f:
                f.write("# 精度要求\n检测精度要求 mAP@0.5 不低于 0.90。\n"
                        "# 数据集\n训练集按 7:1:2 划分。\n")
            res = ingest_dir(tmp, embed_batch, config=IngestConfig(
                chunk_size=48, overlap=0, parent_size=150, min_chunk_tokens=1))
            retriever = HybridRetriever(
                bm25=res.bm25, vector_store=res.vector_store,
                embed_fn=lambda q: embed_batch([q])[0],
                doc_lookup=lambda d: res.doc_text(d))
            svc = RagService(
                retriever=retriever, doc_lookup=lambda d: res.doc_text(d),
                llm=None, meta_lookup=res.metadata_for,
                config=ServiceConfig(refusal_threshold=0.0, top_k=2))

            out = svc.answer("mAP 精度要求")
            self.assertFalse(out.refused)
            self.assertTrue(out.citations)
            self.assertIn("0.90", out.answer)

            frames = "".join(svc.answer_stream("mAP 精度要求"))
            events = [e for e in parse_sse(frames) if "event" in e]
            self.assertIn("citation", [e["event"] for e in events])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
