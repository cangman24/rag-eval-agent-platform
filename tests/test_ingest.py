# -*- coding: utf-8 -*-
"""tests/test_ingest.py —— 建库流水线单测。全部离线（注入假嵌入）。"""
import json
import os
import shutil
import sys
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingest import (  # noqa: E402
    Chunk, IngestConfig, build_parents, chunk_documents,
    ingest_dir, load_chunks, save_chunks, save_manifest,
)
from core.loaders import LoadedDoc  # noqa: E402
from core.vectorstore import InMemoryVectorStore, l2_normalize  # noqa: E402

_TMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp")


def make_tmp() -> str:
    os.makedirs(_TMP_ROOT, exist_ok=True)
    p = os.path.join(_TMP_ROOT, uuid.uuid4().hex[:12])
    os.makedirs(p, exist_ok=True)
    return p


def write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def fake_embed_batch(texts):
    """批量假嵌入：字符哈希，L2 归一化。仅用于验证链路。"""
    out = []
    for t in texts:
        v = [0.0] * 32
        for ch in t:
            if ch.strip():
                v[ord(ch) % 32] += 1.0
        out.append(l2_normalize(v))
    return out


SPEC = """# 技术要求

## 精度要求
检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。
在密集遮挡场景下，允许 mAP 下降不超过 3 个百分点。

## 数据集要求
训练集、验证集、测试集建议按 7:1:2 划分。
标注使用 Labelme 并转换为 YOLO 格式。

# 部署要求
模型应支持导出 ONNX 并量化，以满足端侧推理的时延要求。
推理时延在边缘设备上不宜超过 200 毫秒。
"""


# ================================================================== config
class TestIngestConfig(unittest.TestCase):
    def test_fingerprint_stable(self):
        a = IngestConfig(chunk_size=512, overlap=64, embedding_model="bge-m3")
        b = IngestConfig(chunk_size=512, overlap=64, embedding_model="bge-m3")
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_fingerprint_changes_with_params(self):
        a = IngestConfig(chunk_size=512)
        b = IngestConfig(chunk_size=256)
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_fingerprint_changes_with_model(self):
        self.assertNotEqual(IngestConfig(embedding_model="a").fingerprint(),
                            IngestConfig(embedding_model="b").fingerprint())

    def test_fingerprint_ignores_include_order(self):
        a = IngestConfig(include=[".md", ".txt"])
        b = IngestConfig(include=[".txt", ".md"])
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_fingerprint_length(self):
        self.assertEqual(len(IngestConfig().fingerprint()), 16)


# ================================================================== parents
class TestBuildParents(unittest.TestCase):
    def test_returns_id_text_path_triples(self):
        parents = build_parents("doc1", SPEC, parent_size=200)
        self.assertTrue(parents)
        self.assertEqual(len(parents[0]), 3)
        self.assertTrue(parents[0][0].startswith("doc1#p"))

    def test_ids_unique(self):
        parents = build_parents("doc1", SPEC, parent_size=100)
        ids = [p[0] for p in parents]
        self.assertEqual(len(ids), len(set(ids)))

    def test_content_not_lost(self):
        parents = build_parents("doc1", SPEC, parent_size=100)
        joined = "".join(p[1] for p in parents)
        self.assertIn("mAP@0.5", joined)
        self.assertIn("200 毫秒", joined)

    def test_heading_path_present(self):
        parents = build_parents("doc1", SPEC, parent_size=120)
        paths = [p[2] for p in parents]
        self.assertTrue(any(p for p in paths), paths)


# ================================================================== chunking
class TestChunkDocuments(unittest.TestCase):
    def _docs(self):
        return [LoadedDoc("spec.md", SPEC, "/x/spec.md",
                          {"format": "markdown", "filename": "spec.md"})]

    def test_parent_child_relation_is_exact(self):
        """★ 子块必须落在自己父块的文本里 —— 归属不能靠启发式猜。"""
        cfg = IngestConfig(chunk_size=64, overlap=0, parent_size=200,
                           min_chunk_tokens=1)
        chunks, parents, _ = chunk_documents(self._docs(), cfg)
        self.assertTrue(chunks)
        for c in chunks:
            self.assertIn(c.parent_id, parents)
            # 子块文本应当是父块文本的子串（嵌套切分的直接推论）
            self.assertIn(c.text[:30], parents[c.parent_id],
                          f"{c.chunk_id} 的文本不在其父块内")

    def test_headings_accumulate_within_parent(self):
        cfg = IngestConfig(chunk_size=40, overlap=0, parent_size=150,
                           min_chunk_tokens=1)
        chunks, _, _ = chunk_documents(self._docs(), cfg)
        paths = [c.heading_path for c in chunks if c.heading_path]
        self.assertTrue(any("技术要求" in p for p in paths), paths)

    def test_short_chunks_dropped(self):
        cfg = IngestConfig(chunk_size=64, overlap=0, parent_size=200,
                           min_chunk_tokens=1000)
        chunks, _, stats = chunk_documents(self._docs(), cfg)
        self.assertEqual(chunks, [])
        self.assertGreater(stats["dropped_short_chunks"], 0)

    def test_chunk_ids_unique(self):
        cfg = IngestConfig(chunk_size=40, overlap=0, parent_size=150,
                           min_chunk_tokens=1)
        chunks, _, _ = chunk_documents(self._docs(), cfg)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_metadata_carries_source(self):
        cfg = IngestConfig(chunk_size=64, overlap=0, parent_size=200,
                           min_chunk_tokens=1)
        chunks, _, _ = chunk_documents(self._docs(), cfg)
        self.assertEqual(chunks[0].meta["format"], "markdown")
        self.assertEqual(chunks[0].meta["filename"], "spec.md")

    def test_empty_docs_safe(self):
        chunks, parents, _ = chunk_documents([], IngestConfig())
        self.assertEqual(chunks, [])
        self.assertEqual(parents, {})

    def test_chunk_to_dict(self):
        c = Chunk("id", "t", "d", "p")
        self.assertEqual(c.to_dict()["chunk_id"], "id")


# ================================================================== ingest_dir
class TestIngestDir(unittest.TestCase):
    def _tree(self):
        tmp = make_tmp()
        write(os.path.join(tmp, "spec.md"), SPEC)
        write(os.path.join(tmp, "note.txt"), "补充说明：验收需提交精度报告与混淆矩阵。")
        return tmp

    def _run(self, tmp, **kw):
        cfg = kw.pop("config", IngestConfig(chunk_size=64, overlap=0,
                                            parent_size=200, min_chunk_tokens=1,
                                            embedding_model="fake"))
        return ingest_dir(tmp, fake_embed_batch, config=cfg, **kw)

    def test_builds_indexes(self):
        tmp = self._tree()
        try:
            res = self._run(tmp)
            self.assertGreater(len(res.chunks), 0)
            self.assertGreater(res.vector_store.count(), 0)
            self.assertGreater(len(res.bm25), 0)
            self.assertEqual(res.stats["documents"], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_vector_and_bm25_have_same_ids(self):
        tmp = self._tree()
        try:
            res = self._run(tmp)
            self.assertEqual(sorted(res.bm25.doc_ids),
                             sorted(c.chunk_id for c in res.chunks))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_retrieval_works_end_to_end(self):
        """端到端：建库 -> BM25 检索能命中正确章节。"""
        tmp = self._tree()
        try:
            res = self._run(tmp)
            hits = res.bm25.search("mAP 精度", top_k=3, min_score=0.0)
            self.assertTrue(hits)
            top_text = res.child_text(hits[0][0])
            self.assertIn("mAP", top_text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_parent_text_returned_for_context(self):
        """★ 父子块的核心用途：命中子块后回吐父块正文。"""
        tmp = self._tree()
        try:
            res = self._run(tmp)
            cid = res.chunks[0].chunk_id
            parent_text = res.doc_text(cid)
            child_text = res.child_text(cid)
            self.assertGreaterEqual(len(parent_text), len(child_text))
            self.assertIn(child_text[:20], parent_text)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_metadata_for(self):
        tmp = self._tree()
        try:
            res = self._run(tmp)
            md = res.metadata_for(res.chunks[0].chunk_id)
            self.assertIn("doc_id", md)
            self.assertIn("parent_id", md)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unknown_chunk_returns_empty(self):
        tmp = self._tree()
        try:
            res = self._run(tmp)
            self.assertEqual(res.doc_text("nope"), "")
            self.assertEqual(res.child_text("nope"), "")
            self.assertEqual(res.metadata_for("nope"), {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bad_file_collected_not_fatal(self):
        tmp = self._tree()
        try:
            write(os.path.join(tmp, "bad.json"), "{ 坏 json")
            res = self._run(tmp)
            self.assertGreater(len(res.chunks), 0)
            self.assertEqual(len(res.errors), 1)
            self.assertIn("bad.json", res.errors[0]["path"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_injected_vector_store_used(self):
        tmp = self._tree()
        try:
            store = InMemoryVectorStore()
            res = self._run(tmp, vector_store=store)
            self.assertIs(res.vector_store, store)
            self.assertGreater(store.count(), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_embed_count_mismatch_raises(self):
        """★ 嵌入返回数量不对必须立刻报错，否则索引与文本会错位。"""
        tmp = self._tree()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                ingest_dir(tmp, lambda texts: [[1.0, 0.0]], config=IngestConfig(
                    chunk_size=64, overlap=0, parent_size=200, min_chunk_tokens=1))
            self.assertIn("数量不匹配", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_embed_exception_wrapped(self):
        tmp = self._tree()
        try:
            def boom(texts):
                raise ConnectionError("GPU 掉了")
            with self.assertRaises(RuntimeError) as ctx:
                ingest_dir(tmp, boom, config=IngestConfig(
                    chunk_size=64, overlap=0, parent_size=200, min_chunk_tokens=1))
            self.assertIn("嵌入失败", str(ctx.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_dir_yields_empty_index(self):
        tmp = make_tmp()
        try:
            res = self._run(tmp)
            self.assertEqual(res.chunks, [])
            self.assertEqual(res.bm25.search("任意"), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_include_filter_respected(self):
        tmp = self._tree()
        try:
            res = ingest_dir(tmp, fake_embed_batch, config=IngestConfig(
                chunk_size=64, overlap=0, parent_size=200, min_chunk_tokens=1,
                include=[".md"]))
            self.assertEqual(res.stats["documents"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_stats_contents(self):
        tmp = self._tree()
        try:
            res = self._run(tmp)
            for key in ("documents", "chunks", "parents", "total_chars",
                        "avg_chunk_tokens", "config_fingerprint"):
                self.assertIn(key, res.stats)
            self.assertGreater(res.stats["total_chars"], 0)
            self.assertGreater(res.stats["avg_chunk_tokens"], 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ================================================================== persist
class TestPersistAndManifest(unittest.TestCase):
    def _result(self):
        tmp = make_tmp()
        write(os.path.join(tmp, "spec.md"), SPEC)
        return tmp, ingest_dir(tmp, fake_embed_batch, config=IngestConfig(
            chunk_size=64, overlap=0, parent_size=200, min_chunk_tokens=1,
            embedding_model="fake"))

    def test_manifest_written(self):
        tmp, res = self._result()
        try:
            path = save_manifest(res, os.path.join(tmp, "out", "manifest.json"))
            self.assertTrue(os.path.exists(path))
            data = json.load(open(path, encoding="utf-8"))
            self.assertEqual(data["fingerprint"], res.config.fingerprint())
            self.assertEqual(data["config"]["embedding_model"], "fake")
            self.assertIn("built_at", data)
            self.assertIn("stats", data)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_chunks_roundtrip(self):
        tmp, res = self._result()
        try:
            path = save_chunks(res, os.path.join(tmp, "out", "chunks.json"))
            chunks, parents = load_chunks(path)
            self.assertEqual(len(chunks), len(res.chunks))
            self.assertEqual(len(parents), len(res.parents))
            self.assertEqual(chunks[0].chunk_id, res.chunks[0].chunk_id)
            self.assertEqual(chunks[0].parent_id, res.chunks[0].parent_id)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_manifest_json_serializable(self):
        tmp, res = self._result()
        try:
            json.dumps(res.to_manifest(), ensure_ascii=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ================================================================== integration
class TestFullPipelineIntegration(unittest.TestCase):
    def test_ingest_then_evaluate(self):
        """★ 全链路：解析 -> 切分 -> 建库 -> 检索 -> 评测指标。

        这是 Day 3 真正会跑的东西，只是嵌入是假的。
        """
        from core.evaluation import EvalItem, run_retrieval_eval
        from core.pipeline import HybridRetriever, RetrievalConfig

        tmp = make_tmp()
        try:
            write(os.path.join(tmp, "spec.md"), SPEC)
            write(os.path.join(tmp, "extra.md"),
                  "# 验收要求\n需提交精度报告、混淆矩阵与 Failure Case 分析。\n")
            res = ingest_dir(tmp, fake_embed_batch, config=IngestConfig(
                chunk_size=48, overlap=8, parent_size=200, min_chunk_tokens=1,
                embedding_model="fake"))

            retriever = HybridRetriever(
                bm25=res.bm25, vector_store=res.vector_store,
                embed_fn=lambda q: fake_embed_batch([q])[0],
                rerank_fn=None, doc_lookup=lambda d: res.child_text(d),
            )

            # 用真实命中的块做 gold（演示口径；真实项目必须人工标注）
            def gold_for(keyword):
                hits = res.bm25.search(keyword, top_k=1)
                return [hits[0][0]] if hits else []

            items = [
                EvalItem("q1", "mAP 精度要求", gold_for("mAP 精度"), "", "fact", True),
                EvalItem("q2", "数据集怎么划分", gold_for("数据集 划分"), "", "fact", True),
                EvalItem("q3", "ONNX 量化时延", gold_for("ONNX 量化"), "", "term", True),
            ]
            report = run_retrieval_eval(
                items,
                lambda q: retriever.retrieve(q, RetrievalConfig(top_k=5)),
                config_name="ingest-e2e")
            self.assertEqual(report.n, 3)
            self.assertIn("hit@5", report.overall)
            self.assertEqual(report.overall["hit@5"], 1.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ingest_then_agent_tools(self):
        """★ 全链路：建库 -> 把检索封装成 MCP 工具 -> Agent 调用。"""
        from core.agent import AgentEngine
        from core.mcp import ToolRegistry

        tmp = make_tmp()
        try:
            write(os.path.join(tmp, "spec.md"), SPEC)
            res = ingest_dir(tmp, fake_embed_batch, config=IngestConfig(
                chunk_size=64, overlap=0, parent_size=200, min_chunk_tokens=1,
                embedding_model="fake"))

            reg = ToolRegistry()

            @reg.tool(description="检索知识库")
            def kb_search(query: str, top_k: int = 3):
                """检索知识库

                query: 检索词
                top_k: 条数
                """
                hits = res.bm25.search(query, top_k=top_k)
                return {"count": len(hits),
                        "hits": [{"id": d, "text": res.doc_text(d)}
                                 for d, _ in hits]}

            decisions = [{"tool": "kb_search", "args": {"query": "mAP 精度"}},
                         {"final": "精度要求 mAP 不低于 0.90 [1]",
                          "citations": ["c1"]}]
            cursor = {"i": 0}

            def act(state, allowed):
                d = decisions[min(cursor["i"], len(decisions) - 1)]
                cursor["i"] += 1
                return d

            engine = AgentEngine(registry=reg, intent_fn=lambda q: "kb",
                                 plan_fn=lambda q, t: ["检索"],
                                 act_fn=act, route_table={"kb": ["kb_search"]})
            run = engine.run("检测精度要求是多少")
            self.assertTrue(run.ok, run.error)
            self.assertEqual(run.state.tool_calls, 1)
            # 工具返回的是父块正文，应含关键数值
            tool_results = [s for s in run.state.scratchpad
                            if s.get("type") == "tool_call" and s.get("ok")]
            self.assertTrue(tool_results)
            self.assertIn("mAP", json.dumps(tool_results[0]["result"],
                                            ensure_ascii=False))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
