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
    POST /chat             一次性问答  {"query": "...", "session_id": "..."}
    POST /chat/stream      SSE 流式问答（同样接受 session_id）
    GET  /chat/stream?q=   SSE 流式（方便 EventSource 联调）
    GET  /debug/retrieve   只检索不生成（调参用）
    GET  /debug/service    服务层自描述（配置指纹、动作列表）

    ---- 会话持久化（SQLite，见 core/store.py）----
    GET    /sessions           会话列表（最近更新在前）
    POST   /sessions           新建会话 {"title": "..."}
    GET    /sessions/{sid}     会话详情（含全部消息与引用）
    PATCH  /sessions/{sid}     重命名
    DELETE /sessions/{sid}     删除（连带消息与引用）
    GET    /sessions/stats     库统计（会话/消息/引用/被引文档 TOP）
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

from core.store import ChatStore, default_store

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
from core.sse import SSE_MEDIA_TYPE, TurnFolder

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
    session_id: Optional[str] = Field(
        None, description="会话 id；为空则新建一个，响应里回传")


class ChatResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    refused: bool
    citations: list
    hits: list
    latency_ms: float
    error: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None


class SessionRequest(BaseModel):
    title: str = Field("", max_length=200)


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ------------------------------------------------------- 落库（SSE 帧折叠见 core.sse）
def _persist_turn(store: Optional[ChatStore], cid: Optional[str],
                  turn: Dict[str, Any], rid: Optional[str]) -> Dict[str, Optional[str]]:
    """写 assistant 消息（user 消息在请求开始时就已写）。"""
    if store is None or not cid:
        return {"session_id": cid, "message_id": None}
    mid = store.append_message(
        cid, "assistant", turn.get("answer", ""),
        refused=bool(turn.get("refused")),
        latency_ms=turn.get("latency_ms"),
        request_id=rid,
        citations=turn.get("citations") or [],
        meta={"error": turn.get("error"), "engine": "rag"},
    )
    return {"session_id": cid, "message_id": mid}



# --------------------------------------------------------------------- 装配
def build_app(service: RagService, title: str = "Domain RAG QA",
              store: Optional[ChatStore] = None) -> "FastAPI":
    """把 RagService 挂到 FastAPI 上。

    ★ 用**依赖注入**而不是模块级全局单例：
      同一份代码可以在测试里注入假检索器、在生产里注入真检索器，
      也支持一个进程托管多个知识库（多租户）。

    `store` 同样注入：默认走 core.store.default_store()（`data/app.db`），
    测试里传 `ChatStore(":memory:")` 即可，互不干扰。
    """
    app = FastAPI(title=title, version=service.config.version)
    db = store if store is not None else default_store()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        info = service.health()
        info["db"] = db.stats() if db is not None else None
        return info

    # -------------------------------------------------- 会话（持久化）
    @app.get("/sessions/stats")
    def sessions_stats() -> Dict[str, Any]:
        return db.stats() if db is not None else {"enabled": False}

    @app.get("/sessions")
    def list_sessions(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
        if db is None:
            return {"count": 0, "sessions": []}
        items = db.list_conversations(limit=limit)
        return {"count": len(items), "sessions": items}

    @app.post("/sessions")
    def create_session(req: SessionRequest) -> Dict[str, Any]:
        if db is None:
            raise HTTPException(status_code=503, detail="持久化未启用")
        cid = db.create_conversation(req.title)
        return {"session_id": cid, "title": req.title}

    @app.get("/sessions/{sid}")
    def get_session(sid: str) -> Dict[str, Any]:
        if db is None:
            raise HTTPException(status_code=503, detail="持久化未启用")
        conv = db.get_conversation(sid)
        if conv is None:
            raise HTTPException(status_code=404, detail=f"会话不存在: {sid}")
        return conv

    @app.patch("/sessions/{sid}")
    def rename_session(sid: str, req: RenameRequest) -> Dict[str, Any]:
        if db is None:
            raise HTTPException(status_code=503, detail="持久化未启用")
        if not db.rename_conversation(sid, req.title):
            raise HTTPException(status_code=404, detail=f"会话不存在: {sid}")
        return {"session_id": sid, "title": req.title}

    @app.delete("/sessions/{sid}")
    def delete_session(sid: str) -> Dict[str, Any]:
        if db is None:
            raise HTTPException(status_code=503, detail="持久化未启用")
        if not db.delete_conversation(sid):
            raise HTTPException(status_code=404, detail=f"会话不存在: {sid}")
        return {"deleted": sid}

    # -------------------------------------------------- 问答
    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> Dict[str, Any]:
        cid = db.ensure_conversation(req.session_id, title=req.query[:80]) \
            if db is not None else req.session_id
        rid = req.request_id or new_request_id()
        result = service.answer(req.query, rid).to_dict()
        out = dict(result)
        out["request_id"] = result.get("request_id") or rid
        if db is not None and cid:
            db.append_message(cid, "user", req.query, request_id=rid)
            mid = db.append_message(
                cid, "assistant", result.get("answer", ""),
                refused=bool(result.get("refused")),
                latency_ms=result.get("latency_ms"),
                request_id=rid,
                citations=result.get("citations") or [],
                meta={"error": result.get("error"), "engine": "rag"},
            )
            out["session_id"] = cid
            out["message_id"] = mid
        else:
            out["session_id"] = cid
        return out

    @app.post("/chat/stream")
    def chat_stream(req: ChatRequest) -> "StreamingResponse":
        rid = req.request_id or new_request_id()
        cid = db.ensure_conversation(req.session_id, title=req.query[:80]) \
            if db is not None else None
        if db is not None and cid:
            # 先落用户提问：流式中途断开也能在历史里看到"问过什么"
            db.append_message(cid, "user", req.query, request_id=rid)

        def gen() -> Iterator[str]:
            folder = TurnFolder()
            # 第一帧就把 session_id 告诉前端（刷新后能续上同一条会话）
            yield "event: session\ndata: %s\n\n" % json.dumps(
                {"session_id": cid, "request_id": rid}, ensure_ascii=False)
            try:
                for frame in service.answer_stream(req.query, rid):
                    folder.feed(frame)
                    yield frame
            finally:
                # 客户端 abort 时生成器的 GeneratorExit 也会走到这里
                try:
                    _persist_turn(db, cid, folder.result(), rid)
                except Exception:  # noqa: BLE001 - 落库失败不能影响已推送的答案
                    pass

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
