# -*- coding: utf-8 -*-
import os
import sys
# 由于文件移到了 demos/ 目录下，需要取上一级（项目根目录）的路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from core.llm import LLMClient
from core.mcp import ToolRegistry
from core.ingest import load_chunks
from core.agent_eval import (
    load_agent_tasks,
    agent_ablation_configs,
    run_agent_eval,
    render_agent_ablation,
)
from rag1_knowledge_qa.bootstrap import (
    build_embedder_and_reranker,
    build_or_load_index,
    env,
)
from rag2_kb_agent.app import (
    build_tool_calling_ablation_engine,
    rule_based_intent,
    rule_based_plan,
    DEFAULT_ROUTE_TABLE,
)

# ============ 1. 加载索引 ============
print("[加载] index...")
embedder, reranker = build_embedder_and_reranker()
bm25, store, lookup, stats = build_or_load_index(embedder)
chunks, parents = load_chunks(env("RAG_CHUNKS_FILE"))
chunk_map = {c.chunk_id: c.text for c in chunks}
print(f"[加载] chunks: {len(chunks)}")

# ============ 2. 建工具 ============
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

@reg.tool(description="按工单号查询工单状态与处理记录")
def ticket_query(ticket_id: str):
    """查询工单

    ticket_id: 工单号，如 T-100
    """
    store = {
        "T-100": {"ticket_id": "T-100", "status": "closed",
                  "content": "NDRE 规范制定", "owner": "张三"},
        "T-200": {"ticket_id": "T-200", "status": "open",
                  "content": "辐射校正方案评审", "owner": "李四"},
    }
    return store.get(ticket_id, {"ticket_id": ticket_id, "status": "not_found"})

@reg.tool(description="查询业务指标统计值（数量、占比、趋势）")
def metric_query(metric: str, period: str = "近30天"):
    """查询指标

    metric: 指标名，如 工单量 / 违规率 / 占比
    period: 统计周期
    """
    data = {
        ("工单量", "近30天"): 42,
        ("违规率", "近30天"): 0.03,
        ("复核通过率", "近30天"): 0.85,
    }
    value = data.get((metric, period))
    return {"metric": metric, "period": period, "value": value,
            "unit": "起" if "量" in metric else "比例"}

# ============ 3. 加载任务集 ============
tasks = load_agent_tasks("data/agent_tasks.jsonl")
print(f"[加载] 任务数: {len(tasks)}")

# ============ 4. 真 LLM ============
llm = LLMClient(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
    cache=None,
)

# ============ 5. 四档消融 ============
reports = []
for cfg in agent_ablation_configs():
    print(f"\n[跑评测] {cfg.name} ({cfg.label()}) ...")
    
        # 删掉 route_table 传参
    engine = build_tool_calling_ablation_engine(
        llm, reg, cfg,
        intent_fn=rule_based_intent,
        plan_fn=rule_based_plan,
    )
    
    # ★ 核心修复：run_fn 必须返回原生 AgentRun 对象
    report = run_agent_eval(
        tasks,
        run_fn=lambda q, e=engine: e.run(q),
        config_name=f"{cfg.name} ({cfg.label()})",
    )
    reports.append(report)

# ============ 6. 输出六类指标表 ============
print("\n" + "=" * 80)
print("Agent 层消融评测结果（六类指标）")
print("=" * 80)
print(render_agent_ablation(reports))