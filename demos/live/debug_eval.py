import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()

from core.llm import LLMClient
from core.mcp import ToolRegistry
from core.ingest import load_chunks
from rag1_knowledge_qa.bootstrap import build_embedder_and_reranker, build_or_load_index, env
from rag2_kb_agent.app import build_tool_calling_ablation_engine
from core.agent_eval import agent_ablation_configs, load_agent_tasks

# 只拿 1 条任务
tasks = load_agent_tasks("data/agent_tasks.jsonl")
task = tasks[0]
print(f"任务: {task.query}")

# 构建 B 档引擎
embedder, _ = build_embedder_and_reranker()
bm25, _, _, _ = build_or_load_index(embedder)
chunks, _ = load_chunks(env("RAG_CHUNKS_FILE"))
chunk_map = {c.chunk_id: c.text for c in chunks}

reg = ToolRegistry()
@reg.tool(description="检索知识库")
def kb_search(query: str, top_k: int = 3):
    hits = bm25.search(query, top_k=top_k, min_score=0.0)
    return {"count": len(hits), "hits": [{"id": cid, "text": chunk_map.get(cid, "")} for cid, _ in hits]}

llm = LLMClient(api_key=os.environ["DEEPSEEK_API_KEY"], model="deepseek-chat", cache=None)
cfg = [c for c in agent_ablation_configs() if "tools" in c.name][0] # 选 B 档
engine = build_tool_calling_ablation_engine(llm, reg, cfg)

run = engine.run(task.query)
state_dict = run.state.to_dict()
print("\n=== run.state.to_dict() 里的所有键值 ===")
for k, v in state_dict.items():
    print(f"{k}: {v}")

print("\n=== 测试判卷逻辑 ===")
from core.agent_eval import judge_task_success
ok, reason = judge_task_success(task, state_dict)
print(f"是否成功: {ok}, 原因: {reason}")
