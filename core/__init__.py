# -*- coding: utf-8 -*-
"""core —— RAG / Agent 共享基建。

本包 21 个模块中 **19 个只用标准库**（agent / agent_eval / agent_langgraph /
bm25 / eval_runner / evaluation / ingest / judge / loaders / mcp / memory /
metrics / pipeline / retrieval / service / sse / text / vectorstore / llm），
目的是让检索、Agent 与评测逻辑可以脱离环境独立单测：

    import core.metrics      # 指标：Hit@K / MRR / NDCG / 幻觉率
    import core.retrieval    # RRF 融合 / 递归切分
    import core.text         # 中文友好分词
    import core.bm25         # BM25（自实现）
    import core.vectorstore  # 向量库抽象 + 内存实现 + ChromaDB 实现
    import core.llm          # LLM 客户端（重试 / 限流 / 磁盘缓存）

外部依赖只出现在 5 处，边界清晰：
    core.embedder / core.reranker  -> 导入即需要 torch / transformers（真模型层）
    core.vectorstore.ChromaVectorStore      -> 延迟导入，需要 chromadb
    core.llm.LLMClient 的默认 transport     -> 延迟导入，需要 openai
    core.loaders 的 pdf / docx 解析          -> 延迟导入，需要 pypdf / python-docx

因此 tests/ 的 765 个用例在裸 Python 上全绿（约 1.2 秒），CI 不需要装 torch。
embedder / reranker 不在"零依赖"范围内是刻意的：它们封装的就是真模型，
其行为必须由真机验证（见 demos/live/），不该混进离线单测的职责里。
"""

__all__ = [
    "agent",
    "agent_eval",
    "agent_langgraph",
    "bm25",
    "embedder",
    "eval_runner",
    "evaluation",
    "ingest",
    "judge",
    "llm",
    "loaders",
    "mcp",
    "memory",
    "metrics",
    "pipeline",
    "reranker",
    "retrieval",
    "service",
    "sse",
    "text",
    "vectorstore",
]
