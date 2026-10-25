# -*- coding: utf-8 -*-
"""rag1_knowledge_qa/api_fastapi.py —— FastAPI 接线层（薄适配）。

★ 这一层**只做传输**，不含任何业务逻辑：
      HTTP 请求 -> 拆成 dict -> core.service.RagService -> JSON / SSE

  所以它只有几十行，而且不改动任何已验证的核心代码。

★ 重要：本文件**需要安装 fastapi 才能导入**（开发沙箱里没有）。
  核心逻辑已在 core/service.py 覆盖（全库共 765 项零依赖单测），不依赖本文件验证。
  没装 fastapi 时请直接用 core/service.py 的 RagService —— 它是框架无关的。

启动方式（真机）：
    pip install fastapi uvicorn
    python -m rag1_knowledge_qa.serve          # 见 serve.py
或：
    uvicorn rag1_knowledge_qa.api_fastapi:app_factory --factory --port 8000

接口：
    GET  /health           健康检查 + 索引规模
    POST /chat             一次性问答  {"query": "..."}
    POST /chat/stream      SSE 流式问答
    GET  /chat/stream?q=   SSE 流式（方便 EventSource 联调）
    GET  /debug/retrieve   只检索不生成（调参用）
    GET  /debug/service    服务层自描述（配置指纹、动作列表）
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - 取决于环境
    raise ImportError(
        "本文件需要 fastapi / pydantic：\n"
        "    pip install fastapi uvicorn\n"
        "若暂时不装，请直接使用框架无关的 core.service.RagService。\n"
        f"原始错误: {exc}"
    ) from exc

from core.service import RagService, _hits_from_retrieval, new_request_id
from core.sse import SSE_MEDIA_TYPE

# SSE 响应头：关掉中间层缓冲，否则流式会被攒着一起发
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# --------------------------------------------------------------------- 契约
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    request_id: Optional[str] = Field(None, description="调用方自定义 id，便于链路追踪")


class ChatResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    refused: bool
    citations: list
    hits: list
    latency_ms: float
    error: Optional[str] = None


# --------------------------------------------------------------------- 装配
def build_app(service: RagService, title: str = "Domain RAG QA") -> "FastAPI":
    """把 RagService 挂到 FastAPI 上。

    ★ 用**依赖注入**而不是模块级全局单例：
      同一份代码可以在测试里注入假检索器、在生产里注入真检索器，
      也支持一个进程托管多个知识库（多租户）。
    """
    app = FastAPI(title=title, version=service.config.version)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return service.health()

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> Dict[str, Any]:
        return service.answer(req.query, req.request_id).to_dict()

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> "StreamingResponse":
        def gen() -> Iterator[str]:
            yield from service.answer_stream(req.query, req.request_id)

        return StreamingResponse(gen(), media_type=SSE_MEDIA_TYPE,
                                 headers=SSE_HEADERS)

    @app.get("/chat/stream")
    def chat_stream_get(q: str = Query(..., min_length=1)) -> "StreamingResponse":
        """GET 版：浏览器 `new EventSource('/chat/stream?q=...')` 直接联调。"""
        def gen() -> Iterator[str]:
            yield from service.answer_stream(q, new_request_id())

        return StreamingResponse(gen(), media_type=SSE_MEDIA_TYPE,
                                 headers=SSE_HEADERS)

    @app.get("/debug/retrieve")
    def debug_retrieve(q: str, top_k: int = 5) -> Dict[str, Any]:
        """只跑检索不调 LLM —— Day 3 调检索参数时最常用。"""
        try:
            raw = service.retrieve(q)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"检索失败: {exc}") from exc
        hits = _hits_from_retrieval(raw, service.doc_lookup, top_k)
        return {"query": q, "count": len(hits), "hits": hits}

    @app.get("/debug/service")
    def debug_service() -> Dict[str, Any]:
        return {
            "service": service.config.service_name,
            "version": service.config.version,
            "config": service.config.fingerprint(),
            "llm_configured": service.llm is not None,
            "actions": ["chat", "chat/stream", "debug/retrieve", "health"],
        }

    return app


def app_factory() -> "FastAPI":
    """uvicorn --factory 入口。

    真正的装配逻辑放在 `bootstrap.py`（那里面会 import core.embedder /
    core.reranker，需要你按自己的实现补全）。
    """
    from rag1_knowledge_qa.bootstrap import build_production_service
    return build_app(build_production_service())
