# -*- coding: utf-8 -*-
"""demos/live/smoke_real_agent.py —— 真机冒烟脚本（★ 不是单元测试）。

运行前提（缺一不可）：
  · conda 环境 agent_rag（torch / transformers / openai / python-dotenv）
  · models/bge-m3 与 models/bge-reranker-v2-m3 已下载到 models/
  · 已建库：python -m rag1_knowledge_qa.cli ingest --docs data/raw
  · DEEPSEEK_API_KEY 已配置（.env 或环境变量）

用法（会调真模型、花真实 API 费用）：
    python demos/live/smoke_real_agent.py

★ 它刻意不在 tests/ 里：tests/ 的 765 个用例必须保持"零第三方依赖可跑"，
  而这个脚本需要 GPU、模型与 API Key。两者的职责边界是架构决策，不是疏忽。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_env() -> None:
    """惰性 + 容错地加载 .env（没装 dotenv 时手写极简解析）。"""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except ImportError:
        pass
    path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def main() -> None:
    _load_env()

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


if __name__ == "__main__":
    main()
