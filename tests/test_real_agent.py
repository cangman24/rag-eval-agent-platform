import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
print(f"[加载] chunks: {len(chunks)}")

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

# 越权测试用的工具（故意不给白名单）
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
    cache=DiskCache("data/cache/llm"),
)

# ============ 4. 引擎 ============
engine = build_tool_calling_engine(
    llm,
    reg,
    checkpointer=build_default_checkpointer(),
)

# ============ 5. 跑 ============
run = engine.run("反复检索制度的 NDRE 规范，直到我说停")
print(run.state.stop_reason)  # max_rounds 或 loop_detected

print("\n=== 回答 ===")
print(run.state.answer)
print("\n=== 意图 ===")
print(run.state.task_type)
print("\n=== 白名单 ===")
print(run.state.allowed_tools)
print("\n=== 引用 ===")
print(getattr(run.state, "citations", None))
print(f"\n轮次: {run.state.rounds}, 工具调用: {run.state.tool_calls}, 停止原因: {run.state.stop_reason}")

print("\n=== 事件流 ===")
for e in run.events:
    print(f"[{e['type']}] {json.dumps(e.get('data', {}), ensure_ascii=False)[:180]}")