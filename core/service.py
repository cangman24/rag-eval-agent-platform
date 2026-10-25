# -*- coding: utf-8 -*-
"""core/service.py —— 框架无关的 API 服务层。

★ 为什么先写"框架无关的服务层"而不是直接写 FastAPI 路由：

  1. **可测性**：本沙箱里没有 fastapi / starlette / pydantic。如果业务逻辑
     写在路由函数里，我就一行都验证不了。把逻辑抽成"收 dict、回 dict"的
     纯函数后，765 项单测中的问答 / 拒答 / Agent 执行 / SSE 流路径全覆盖。
  2. **可替换性**：以后要换 Flask / Litestar / gRPC，只改一层薄薄的适配；
     业务代码零改动。
  3. **面试可讲**：这就是"业务逻辑与传输层解耦"。被问"你怎么保证接口
     正确"，可以答"核心逻辑是纯函数，接口层只有 40 行适配，逻辑全部单测覆盖"。

真正的 FastAPI 接线在 `rag1_knowledge_qa/api_fastapi.py`，只有几十行。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .agent import AgentRun, EventType
from .llm import REFUSAL_MARKER, build_answer_messages, is_refusal, should_refuse
from .sse import SSEStream


# --------------------------------------------------------------------- 配置
@dataclass
class ServiceConfig:
    """服务层运行参数。"""

    refusal_threshold: float = 0.3      # 低置信拒答阈值
    top_k: int = 5
    recall_k: int = 20
    max_answer_chars: int = 4000
    enable_cache: bool = True
    service_name: str = "domain-rag-qa"
    version: str = "0.1.0"

    def fingerprint(self) -> Dict[str, Any]:
        return {"top_k": self.top_k, "recall_k": self.recall_k,
                "refusal_threshold": self.refusal_threshold}


# --------------------------------------------------------------------- 数据结构
@dataclass
class Citation:
    """一条引用。前端用它渲染可点击的来源。"""

    index: int
    chunk_id: str
    heading_path: str = ""
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "chunk_id": self.chunk_id,
                "heading_path": self.heading_path, "snippet": self.snippet}


@dataclass
class AnswerResult:
    """一次问答的完整结果。"""

    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    refused: bool = False
    hits: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "answer": self.answer,
            "refused": self.refused,
            "citations": [c.to_dict() for c in self.citations],
            "hits": self.hits,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


# --------------------------------------------------------------------- 工具
def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _hits_from_retrieval(result: Any, doc_lookup: Callable[[str], str],
                         top_k: int) -> List[Dict[str, Any]]:
    """把检索结果标准化成 hit 列表。"""
    doc_ids = getattr(result, "doc_ids", None)
    scores = getattr(result, "scores", None)
    if doc_ids is None and isinstance(result, dict):
        doc_ids = result.get("doc_ids", [])
        scores = result.get("scores", [])
    doc_ids = list(doc_ids or [])
    scores = list(scores or [])
    hits: List[Dict[str, Any]] = []
    for i, cid in enumerate(doc_ids[:top_k]):
        hits.append({
            "chunk_id": cid,
            "score": float(scores[i]) if i < len(scores) else 0.0,
            "text": doc_lookup(cid),
        })
    return hits


def build_citations(hits: Sequence[Dict[str, Any]],
                    meta_lookup: Optional[Callable[[str], Dict[str, Any]]] = None,
                    snippet_chars: int = 120) -> List[Citation]:
    """把 hit 列表转成带序号的引用（序号与 Prompt 里的 [1][2] 对应）。"""
    out: List[Citation] = []
    for i, h in enumerate(hits, start=1):
        meta = (meta_lookup(h["chunk_id"]) if meta_lookup else {}) or {}
        text = h.get("text") or ""
        out.append(Citation(
            index=i,
            chunk_id=h["chunk_id"],
            heading_path=meta.get("heading_path", ""),
            snippet=text[:snippet_chars].replace("\n", " "),
        ))
    return out


# --------------------------------------------------------------------- RAG 服务
class RagService:
    """项目一：知识库问答服务（框架无关）。

    Args:
        retriever: 需有 `retrieve(query, cfg) -> {doc_ids, scores}`；
            或直接传 `Callable[[str], Any]`
        doc_lookup: chunk_id -> 喂给模型的正文（**父子块场景下应返回父块**）
        llm: LLMClient（可为 None：无 Key 时只回检索片段，便于本地冒烟）
        config: ServiceConfig
        intent_fn: 可选的意图识别（用于在回答前判断是否该拒答）
    """

    def __init__(
        self,
        retriever: Any,
        doc_lookup: Callable[[str], str],
        llm: Any = None,
        config: Optional[ServiceConfig] = None,
        meta_lookup: Optional[Callable[[str], Dict[str, Any]]] = None,
        retrieval_config: Any = None,
    ) -> None:
        self.retriever = retriever
        self.doc_lookup = doc_lookup
        self.llm = llm
        self.config = config or ServiceConfig()
        self.meta_lookup = meta_lookup
        self.retrieval_config = retrieval_config

    # ---------------------------------------------------------- 检索
    def retrieve(self, query: str) -> Any:
        if callable(self.retriever) and not hasattr(self.retriever, "retrieve"):
            return self.retriever(query)
        if self.retrieval_config is not None:
            return self.retriever.retrieve(query, self.retrieval_config)
        return self.retriever.retrieve(query)

    # ---------------------------------------------------------- 问答
    def answer(self, query: str, request_id: Optional[str] = None) -> AnswerResult:
        rid = request_id or new_request_id()
        started = time.perf_counter()
        result = AnswerResult(query=query, answer="", request_id=rid)

        if not (query or "").strip():
            result.answer = "问题不能为空。"
            result.error = "empty_query"
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        try:
            raw = self.retrieve(query)
        except Exception as exc:  # noqa: BLE001 - 检索失败要变成可读错误
            result.error = f"retrieval_failed: {exc}"
            result.answer = "检索服务暂时不可用，请稍后重试。"
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        hits = _hits_from_retrieval(raw, self.doc_lookup, self.config.top_k)
        result.hits = hits
        result.citations = build_citations(hits, self.meta_lookup)

        # ---- 低置信拒答：宁可不答，也不能编 ----
        score_hits = [{"score": h["score"]} for h in hits]
        if should_refuse(score_hits, self.config.refusal_threshold):
            result.refused = True
            result.answer = f"知识库中{REFUSAL_MARKER}相关内容。"
            result.citations = []
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        # ---- 无 LLM 时降级：直接把检索片段拼出来（便于本地冒烟） ----
        if self.llm is None:
            joined = "\n\n".join(
                f"[{i}] {h['text'][:300]}" for i, h in enumerate(hits, start=1)
            )
            result.answer = f"（未配置 LLM，以下为检索结果）\n{joined}"
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        # ---- 调 LLM 生成（带引用约束） ----
        try:
            messages = build_answer_messages(query, hits)
            llm_result = self.llm.chat(messages)
            answer = (llm_result.text or "").strip()
            result.answer = answer[: self.config.max_answer_chars]
            if is_refusal(answer):
                result.refused = True
                result.citations = []
        except Exception as exc:  # noqa: BLE001 - LLM 挂了要给出可用降级
            result.error = f"llm_failed: {exc}"
            joined = "\n\n".join(
                f"[{i}] {h['text'][:300]}" for i, h in enumerate(hits, start=1)
            )
            result.answer = f"（生成服务异常，以下为检索结果）\n{joined}"

        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    # ---------------------------------------------------------- 流式
    def answer_stream(self, query: str, request_id: Optional[str] = None) -> Iterator[str]:
        """SSE 流：先发检索结果，再发答案；引用只在**确定作答**后才发。

        ★ 顺序很关键：不能先发 citation 再判断拒答。
          真实踩坑（被 test_stream_refusal_path 抓到）：初版在生成答案之前
          就把引用推给前端了，结果是"拒答"场景下前端已经渲染了引用来源——
          用户会看到一个"未找到"的回答配着 3 条来源，自相矛盾。

          正确顺序：检索 -> 生成/判定 -> （不拒答才发 citation）-> answer -> done。
        """
        rid = request_id or new_request_id()
        stream = SSEStream()
        t0 = time.perf_counter()

        yield stream.emit(EventType.START, query=query, request_id=rid)

        try:
            raw = self.retrieve(query)
        except Exception as exc:  # noqa: BLE001
            yield stream.emit(EventType.ERROR, error=f"retrieval_failed: {exc}")
            yield stream.emit(EventType.DONE, request_id=rid)
            return

        hits = _hits_from_retrieval(raw, self.doc_lookup, self.config.top_k)
        yield stream.emit("retrieval", count=len(hits),
                          top_score=hits[0]["score"] if hits else 0.0)

        result = self.answer(query, request_id=rid)

        if result.refused:
            yield stream.emit(EventType.REFUSAL, text=result.answer)
        else:
            # 只有确定作答才发引用，保证"回答-引用"一致
            citations = build_citations(hits, self.meta_lookup)
            for c in citations:
                yield stream.emit("citation", **c.to_dict())
            yield stream.emit(EventType.ANSWER, text=result.answer,
                              citations=[c.chunk_id for c in citations])

        if result.error:
            yield stream.emit(EventType.ERROR, error=result.error)
        yield stream.emit(EventType.DONE, request_id=rid,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))

    # ---------------------------------------------------------- 健康检查
    def health(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "status": "ok",
            "service": self.config.service_name,
            "version": self.config.version,
            "config": self.config.fingerprint(),
            "llm_configured": self.llm is not None,
        }
        store = getattr(self.retriever, "vector_store", None)
        if store is not None and hasattr(store, "count"):
            info["index_size"] = store.count()
        bm25 = getattr(self.retriever, "bm25", None)
        if bm25 is not None and hasattr(bm25, "__len__"):
            info["bm25_size"] = len(bm25)
        return info


# --------------------------------------------------------------------- Agent 服务
class AgentService:
    """项目二：知识库 Agent 服务（框架无关）。"""

    def __init__(self, engine: Any, config: Optional[ServiceConfig] = None) -> None:
        self.engine = engine
        self.config = config or ServiceConfig(service_name="kb-agent-platform")
        self._runs: Dict[str, AgentRun] = {}

    def run(self, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        rid = thread_id or new_request_id()
        run = self.engine.run(query, thread_id=rid)
        self._runs[rid] = run
        return {
            "request_id": rid,
            "thread_id": run.state.thread_id,
            "task_type": run.state.task_type,
            "allowed_tools": run.state.allowed_tools,
            "plan": run.state.plan,
            "answer": run.state.answer,
            "citations": run.state.citations,
            "verified": run.state.verified,
            "finished": run.state.finished,
            "stop_reason": run.state.stop_reason,
            "rounds": run.state.rounds,
            "tool_calls": run.state.tool_calls,
            "latency_ms": round(run.state.elapsed_s() * 1000, 1),
            "error": run.error,
        }

    def resume(self, thread_id: str) -> Dict[str, Any]:
        run = self.engine.resume(thread_id)
        self._runs[thread_id] = run
        return {
            "thread_id": thread_id,
            "answer": run.state.answer,
            "finished": run.state.finished,
            "rounds": run.state.rounds,
            "tool_calls": run.state.tool_calls,
            "stop_reason": run.state.stop_reason,
            "error": run.error,
        }

    def run_stream(self, query: str, thread_id: Optional[str] = None) -> Iterator[str]:
        """Agent 执行的 SSE 流。"""
        rid = thread_id or new_request_id()
        run = self.engine.run(query, thread_id=rid)
        self._runs[rid] = run
        stream = SSEStream()
        for ev in run.events:
            payload = dict(ev.get("data") or {})
            payload["ts"] = ev.get("ts")
            yield stream.emit(ev.get("type") or "message", **payload)

    def trace(self, thread_id: str) -> Dict[str, Any]:
        """取某次执行的完整轨迹（前端轨迹面板 / 事后审计）。"""
        run = self._runs.get(thread_id)
        if run is None:
            return {"thread_id": thread_id, "found": False, "events": []}
        return {
            "thread_id": thread_id,
            "found": True,
            "task_type": run.state.task_type,
            "allowed_tools": run.state.allowed_tools,
            "plan": run.state.plan,
            "rounds": run.state.rounds,
            "tool_calls": run.state.tool_calls,
            "stop_reason": run.state.stop_reason,
            "scratchpad": run.state.scratchpad,
            "events": run.events,
        }

    def tools(self) -> Dict[str, Any]:
        """列出注册的工具与白名单路由表（前端/调试用）。"""
        registry = getattr(self.engine, "registry", None)
        if registry is None:
            return {"tools": [], "routes": {}}
        return {
            "tools": registry.mcp_tools(),
            "routes": {k: list(v) for k, v in
                       getattr(self.engine, "route_table", {}).items()},
        }

    def health(self) -> Dict[str, Any]:
        registry = getattr(self.engine, "registry", None)
        return {
            "status": "ok",
            "service": self.config.service_name,
            "version": self.config.version,
            "tool_count": len(registry) if registry is not None else 0,
            "checkpointer": type(getattr(self.engine, "checkpointer", None)).__name__,
            "max_rounds": getattr(self.engine, "max_rounds", None),
            "max_tool_calls": getattr(self.engine, "max_tool_calls", None),
        }


# --------------------------------------------------------------------- 分发
def dispatch(service: Any, action: str, payload: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """统一入口：把 (action, payload) 分发给某个 service。

    这是给"不想用 Web 框架"的场景准备的 —— 比如 CLI、消息队列消费者，
    或者单元测试。FastAPI 层也复用它，保证两条路径行为一致。
    """
    payload = payload or {}
    if isinstance(service, RagService):
        if action == "answer":
            return service.answer(payload.get("query", ""),
                                  payload.get("request_id")).to_dict()
        if action == "health":
            return service.health()
        return {"error": f"unknown_action: {action}"}

    if isinstance(service, AgentService):
        if action == "run":
            return service.run(payload.get("query", ""), payload.get("thread_id"))
        if action == "resume":
            return service.resume(payload.get("thread_id", ""))
        if action == "trace":
            return service.trace(payload.get("thread_id", ""))
        if action == "tools":
            return service.tools()
        if action == "health":
            return service.health()
        return {"error": f"unknown_action: {action}"}

    return {"error": "unknown_service"}
