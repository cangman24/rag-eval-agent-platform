# -*- coding: utf-8 -*-
"""rag1_knowledge_qa —— 项目一：基于 RAG 的垂直领域知识问答系统。

模块分工：
    bootstrap.py    生产装配（需要 GPU + BGE 模型 + DeepSeek Key）
    api_fastapi.py  HTTP/SSE 接口（薄适配，需 fastapi）
    cli.py          命令行入口（不装 fastapi 也能用）

真正干活的逻辑全在 core/ 里，本包只做装配与传输。
"""
