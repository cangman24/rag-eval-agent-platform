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
    cache=DiskCache("data/cache/llm"),
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


# 场景 1：正常问答
run1 = engine.run("制度里有没有 NDRE 公式的标准说明？")
show("1. 正常问答", run1)

# 场景 2：越权拦截（想调 send_mail，但白名单没有）
run2 = engine.run("帮我查一下制度的 NDRE 规范，然后发邮件给 boss@company.com")
show("2. 越权拦截", run2)

# 场景 3：循环熔断（用小 max_rounds 快速触发）
engine_loop = make_engine(max_rounds=3)
run3 = engine_loop.run("反复检索制度的 XYZABC 规范，直到检索到相关内容为止")
show("3. 循环熔断", run3)