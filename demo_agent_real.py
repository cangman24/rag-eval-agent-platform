# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import json
from core.llm import LLMClient, DiskCache
from core.mcp import ToolRegistry
from core.ingest import load_chunks
from rag1_knowledge_qa.bootstrap import (
    build_embedder_and_reranker,
    build_or_load_index,
    env,
)
from rag2_kb_agent.app import (
    build_tool_calling_engine,
    build_default_checkpointer,
)

# ============ 1. 加载真索引 ============
print("[加载] embedder + reranker...")
embedder, reranker = build_embedder_and_reranker()

print("[加载] index...")
bm25, store, lookup, stats = build_or_load_index(embedder)
print(f"[加载] stats: {stats}")

chunks, parents = load_chunks(env("RAG_CHUNKS_FILE"))
chunk_map = {c.chunk_id: c.text for c in chunks}
print(f"[加载] chunks: {len(chunks)}\n")

# ============ 2. 建真工具 ============
reg = ToolRegistry()

@reg.tool(description="检索知识库，返回与查询最相关的制度条款片段")
def kb_search(query: str, top_k: int = 3):
    """检索知识库

    query: 检索关键词
    top_k: 返回条数（建议 3~5）
    """
    hits = bm25.search(query, top_k=top_k, min_score=0.0)
    return {
        "count": len(hits),
        "hits": [{"id": cid, "text": chunk_map.get(cid, "")} for cid, _ in hits],
    }

@reg.tool(description="发送邮件（有副作用）", sensitive=True)
def send_mail(to: str):
    """发送邮件

    to: 收件人
    """
    return {"sent": True, "to": to}

# ============ 3. 真 LLM ============
llm = LLMClient(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
    cache=None,     # ★ Demo 时禁用缓存，避免坏响应被固化
)

# ============ 4. 引擎 ============
def make_engine(**kw):
    return build_tool_calling_engine(
        llm, reg,
        checkpointer=build_default_checkpointer(),
        **kw,
    )

engine = make_engine()

# ============ 5. 多场景演示 ============
def show(name, run):
    print(f"\n{'='*60}\n【场景】{name}\n{'='*60}")
    print(f"意图: {run.state.task_type} | 白名单: {run.state.allowed_tools}")
    print(f"停止原因: {run.state.stop_reason} | 轮次: {run.state.rounds} | 工具调用: {run.state.tool_calls}")
    print(f"\n回答:\n{run.state.answer[:300]}...")
    print(f"\n引用: {getattr(run.state, 'citations', None)}")
    print("\n关键事件:")
    for e in run.events:
        if e["type"] in ("tool_call", "tool_error", "tool_result", "answer"):
            print(f"  [{e['type']}] {json.dumps(e.get('data', {}), ensure_ascii=False)[:120]}")

# ---------------------------------------------------------
# 场景 1：正常问答
# ---------------------------------------------------------
run1 = engine.run("制度里有没有 NDRE 公式的标准说明？")
show("1. 正常问答", run1)

# ---------------------------------------------------------
# 场景 2：越权拦截
# ---------------------------------------------------------
run2 = engine.run("帮我查一下制度的 NDRE 规范，然后发邮件给 boss@company.com")
show("2. 越权拦截", run2)

# ---------------------------------------------------------
# 场景 3：熔断演示（强制 max_rounds=1，让第一轮就被截断）
# ---------------------------------------------------------
engine_loop = make_engine(max_rounds=1)
run3 = engine_loop.run("帮我检索制度的 NDRE 规范")
show("3. 熔断演示（max_rounds=1）", run3)

# ---------------------------------------------------------
# 场景 4：断点续跑（用同一个 checkpointer 分两段跑）
# ---------------------------------------------------------
print(f"\n{'='*60}\n【场景】4. 断点续跑\n{'='*60}")

# 第一段：max_rounds=1，强制中断
engine_cp1 = make_engine(max_rounds=1)
run4 = engine_cp1.run("帮我检索制度的 NDRE 规范并总结", thread_id="T-DEMO-CP")
print(f"第一段:")
print(f"  thread_id   = {run4.state.thread_id}")
print(f"  stop_reason = {run4.state.stop_reason}")
print(f"  rounds      = {run4.state.rounds}")
print(f"  tool_calls  = {run4.state.tool_calls}")
print(f"  answer      = {run4.state.answer[:100]}...")

# 第二段：用 resume 从断点继续
engine_cp2 = make_engine()
run5 = engine_cp2.resume("T-DEMO-CP")
print(f"\n续跑:")
print(f"  thread_id   = {run5.state.thread_id}")
print(f"  stop_reason = {run5.state.stop_reason}")
print(f"  rounds      = {run5.state.rounds}")
print(f"  tool_calls  = {run5.state.tool_calls}")
print(f"  answer      = {run5.state.answer[:200]}...")
print(f"  citations   = {run5.state.citations}")