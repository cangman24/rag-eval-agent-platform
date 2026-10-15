# -*- coding: utf-8 -*-
"""core —— RAG / Agent 共享基建。

本包内所有模块**零第三方依赖**（torch / chromadb / langchain 都不需要），
目的是让检索与评测逻辑可以脱离环境独立单测：

    import core.metrics      # 指标：Hit@K / MRR / NDCG / 幻觉率
    import core.retrieval    # RRF 融合 / 递归切分
    import core.text         # 中文友好分词
    import core.bm25         # BM25（自实现）
    import core.vectorstore  # 向量库抽象 + 内存实现 + ChromaDB 实现
    import core.llm          # LLM 客户端（重试 / 限流 / 磁盘缓存）

需要外部依赖的只有两个，且都是**延迟导入**：
    core.vectorstore.ChromaVectorStore  -> 需要 chromadb
    core.llm.LLMClient 的默认 transport -> 需要 openai
"""

__all__ = [
    "bm25",
    "llm",
    "metrics",
    "retrieval",
    "text",
    "vectorstore",
]
