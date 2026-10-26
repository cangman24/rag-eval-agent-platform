# -*- coding: utf-8 -*-
"""rag2_kb_agent/agent_api.py —— 项目一（Agent 平台）的 HTTP/SSE 入口。

★ 为什么要有这个文件：

    简历写的是「SSE 实时输出 Agent 推理轨迹，支撑前端可视化面板定位错误节点」。
    在此之前，`rag2_kb_agent/` 只有库级装配（app.py）与 MCP 命令行（mcp_cli.py），
    **没有任何 HTTP 入口** —— 13 类轨迹事件在 `core/agent.py` 里定义了、
    在 `core/agent_langgraph.py` 里也真的 emit 了，但只到内存列表就断了，
    没有任何一条路径能把它送到浏览器。

    本文件补的就是这条路径：把 `engine.run_stream()` 的事件流直接挂到 SSE 上。

★ 与项目二（rag1_knowledge_qa/api_fastapi.py）的分工：
      项目二 SSE 推的是**检索/生成阶段**（start/retrieval/citation/answer）
      项目一 SSE 推的是**Agent 推理轨迹**（intent/plan/tool_call/tool_result/
                                        tool_error/verify/limit/resume/done）
    两者是不同的可观测对象，不要混。

★ 需要 fastapi；没装时给出明确指引（核心逻辑在 core，不依赖本文件验证）。

启动（真机）：
    uvicorn rag2_kb_agent.agent_api:app_factory --factory --port 8100

接口：
    GET  /agent/health            引擎类型、图拓扑、熔断阈值、线程数
    POST /agent/chat              一次性执行（返回完整事件数组 + 最终状态）
    POST /agent/chat/stream       SSE 实时推轨迹事件
    GET  /agent/threads           可回放的 thread 列表
    GET  /agent/trace/{tid}       回放某个 thread 的完整轨迹（含 span 树）
    GET  /agent/spans/{tid}       只取 span 树（甘特条 / 瓶颈 / 错误节点）
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - 取决于环境
    raise ImportError(
        "本文件需要 fastapi / pydantic：\n"
        "    pip install fastapi uvicorn\n"
        f"原始错误: {exc}"
    ) from exc

from core.agent import EventType, InMemoryCheckpointer, JsonFileCheckpointer
from core.agent_langgraph import langgraph_available, make_agent_engine
from core.observability import analyze
from core.sse import SSE_MEDIA_TYPE, stream_agent_events
from rag2_kb_agent.app import (
    DEFAULT_ROUTE_TABLE, rule_based_intent, rule_based_plan,
)
from core.mcp import ToolRegistry

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# --------------------------------------------------------------------- 请求/响应契约
class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: Optional[str] = Field(None, description="省略则自动生成")
    resume: bool = Field(False, description="从检查点续跑")


# --------------------------------------------------------------------- 演示用工具集
def build_demo_registry() -> ToolRegistry:
    """没有真实索引时也能跑通全链路的工具集。

    ★ 里面故意留了一个**不在任何白名单里**的 `send_mail`：
      面板要演示"定位错误节点"，就得有一个真会被拦下的节点可看。
    """
    reg = ToolRegistry()

    @reg.tool(description="检索知识库，返回相关制度条款片段")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 检索关键词
        top_k: 返回条数
        """
        return {"count": 2,
                "hits": [{"id": "制度汇编#p1#c0",
                          "text": "第三章 复核要求：处理结果须经二级复核后方可关闭。"},
                         {"id": "作业规范#p0#c2",
                          "text": "第五条 精度要求：mAP@0.5 不低于 0.85。"}]}

    @reg.tool(description="按工单号查询工单状态与处理记录")
    def ticket_query(ticket_id: str):
        """查询工单

        ticket_id: 工单号，如 T-100
        """
        return {"ticket_id": ticket_id, "status": "closed",
                "handler": "张三", "note": "已完成二级复核"}

    @reg.tool(description="查询业务指标统计值")
    def metric_query(metric: str, period: str = "近30天"):
        """查询指标

        metric: 指标名
        period: 统计周期
        """
        return {"metric": metric, "period": period, "value": 42,
                "trend": "down 12%"}

    @reg.tool(description="发送邮件通知（高危操作，默认不在任何白名单）")
    def send_mail(to: str, subject: str = "通知"):
        """发邮件

        to: 收件人
        subject: 主题
        """
        return {"sent": True, "to": to}

    return reg


# --------------------------------------------------------------------- 无 LLM 决策器
class RuleBasedDecider:
    """不接 LLM 时的决策器：让轨迹面板在**没有 API key 的情况下**也能完整演示。

    ★ 它故意会做两件"错事"，因为演示"定位错误节点"需要真的有错误可定位：
        ① 白名单为空（general 意图）时仍尝试调 kb_search → 越权拦截
        ② 问到"发邮件/通知"时调 send_mail → 越权拦截（该工具不在任何白名单）

    真机接 LLM 时用 `ToolCallingDecider` 替换即可，接口完全一致：
        act_fn(state, allowed) -> {"tool":..., "args":...} | {"final":...}
    """

    def __call__(self, state: Any, allowed: List[str]) -> Dict[str, Any]:
        allowed = list(allowed or [])
        query = getattr(state, "query", "") or ""
        pad = list(getattr(state, "scratchpad", []) or [])
        calls = [s for s in pad if s.get("type") == "tool_call" and s.get("ok")]
        # ★ 被拦下的也算"试过"：不统计它的话，模型会一遍遍重试同一个
        #   越权工具，最后触发 loop_detected 熔断 —— 轨迹里出现 3 个一模一样
        #   的错误节点，反而看不清到底错在哪。
        attempted = {s.get("tool") for s in pad if s.get("type") == "tool_call"}

        # ① 触发高危工具（必被拦，用于演示面板的红色错误节点）
        if (any(k in query for k in ("邮件", "通知", "发送"))
                and "send_mail" not in attempted):
            return {"tool": "send_mail",
                    "args": {"to": "ops@example.com", "subject": query[:30]}}

        # ② 高危工具已被拦 -> 说明权限不足，直接收尾
        if "send_mail" in attempted:
            return {"final": "该请求需要调用 send_mail（高危操作），已被工具白名单"
                             "拦截。当前意图下无可用工具，无法执行，请改用只读查询。",
                    "citations": []}

        # ③ 还没检索过就先检索（若在白名单内）
        if "kb_search" in allowed and not calls:
            return {"tool": "kb_search", "args": {"query": query, "top_k": 3}}

        # ④ 白名单为空：仍然尝试检索 —— 模拟模型越权，会被硬拦截
        if not allowed and not calls:
            return {"tool": "kb_search", "args": {"query": query, "top_k": 3}}

        # ④ 已有观察结果 -> 收尾
        summary = "；".join(
            f"{c.get('tool')} 返回 {str(c.get('result'))[:80]}"
            for c in calls) or "未获取到工具结果"
        return {"final": f"根据工具结果：{summary}。结论：{query[:40]} 已核对完毕。",
                "citations": []}


# --------------------------------------------------------------------- 装配
def build_agent_engine(
    registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[Any] = None,
    llm: Optional[Any] = None,
    route_table: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = "data/checkpoints",
    prefer_langgraph: bool = True,
    max_rounds: int = 6,
    max_tool_calls: int = 12,
) -> Any:
    """装配 Agent 引擎：优先真 LangGraph，装不上自动回退自实现版。

    llm=None 时用 RuleBasedDecider（零成本可演示）；
    传入 llm 则走 ToolCallingDecider（真实 function calling）。
    """
    registry = registry or build_demo_registry()
    if checkpointer is None:
        try:
            checkpointer = JsonFileCheckpointer(checkpoint_dir)
        except OSError:  # pragma: no cover - 只读环境兜底
            checkpointer = InMemoryCheckpointer()

    act_fn: Callable[..., Any]
    if llm is not None:
        from core.memory import LayeredMemory
        from rag2_kb_agent.decisions import ToolCallingDecider
        from rag2_kb_agent.prompts import MessageBuilder
        builder = MessageBuilder(registry=registry)
        memory = LayeredMemory()
        builder.memory_provider = lambda q: memory.build_context(
            q, recent_n=6, recall_k=3)
        act_fn = ToolCallingDecider(llm=llm, registry=registry,
                                    build_messages=builder)
    else:
        act_fn = RuleBasedDecider()

    return make_agent_engine(
        prefer_langgraph=prefer_langgraph,
        registry=registry,
        intent_fn=rule_based_intent,
        plan_fn=rule_based_plan,
        act_fn=act_fn,
        route_table=dict(route_table or DEFAULT_ROUTE_TABLE),
        checkpointer=checkpointer,
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
    )


def _engine_info(engine: Any) -> Dict[str, Any]:
    """引擎自描述：面板顶部要显示"现在跑的是哪一版"。"""
    info: Dict[str, Any] = {
        "engine": type(engine).__name__,
        "langgraph": langgraph_available(),
        "route_table": {k: list(v) for k, v in
                        (getattr(engine, "route_table", {}) or {}).items()},
        "fuses": {"max_rounds": getattr(engine, "max_rounds", None),
                  "max_tool_calls": getattr(engine, "max_tool_calls", None),
                  "total_timeout_s": getattr(engine, "total_timeout_s", None)},
    }
    if hasattr(engine, "graph_nodes"):
        try:
            info["nodes"] = list(engine.graph_nodes())
        except Exception:  # noqa: BLE001 - 自描述失败不该让 /health 挂掉
            info["nodes"] = []
    return info


# --------------------------------------------------------------------- 应用
def app_factory(
    registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[Any] = None,
    llm: Optional[Any] = None,
    checkpoint_dir: str = "data/checkpoints",
    prefer_langgraph: bool = True,
) -> FastAPI:
    """FastAPI app 工厂（`uvicorn ...:app_factory --factory`）。"""
    engine = build_agent_engine(
        registry=registry, checkpointer=checkpointer, llm=llm,
        checkpoint_dir=checkpoint_dir, prefer_langgraph=prefer_langgraph,
    )
    cp = getattr(engine, "checkpointer", None)
    app = FastAPI(title="KB Agent Platform",
                  description="受控 Agent 工作流 —— SSE 推理轨迹",
                  version="1.0.0")

    @app.get("/agent/health")
    def health() -> Dict[str, Any]:
        threads = []
        if cp is not None:
            try:
                threads = list(cp.list_threads())
            except Exception:  # noqa: BLE001
                threads = []
        info = _engine_info(engine)
        info.update({"status": "ok", "threads": len(threads)})
        return info

    @app.get("/agent/threads")
    def threads(limit: int = Query(20, ge=1, le=500)) -> Dict[str, Any]:
        """可回放的 thread 列表。

        ★ 有 state 没 events 的 thread（本文件上线之前跑的 63 个历史检查点）
          也列出来，但标 `has_events=false` —— 面板要据此给出"无轨迹可回放"
          而不是空白时间线。
        """
        if cp is None:
            return {"count": 0, "threads": []}
        items = []
        # ★ 倒序：thread_id 形如 t<毫秒时间戳>，倒序即"最近运行的在前"。
        #   正序时列表会被本文件上线之前跑的 60+ 个历史检查点占满，
        #   刚跑完的那次根本翻不到 —— 回放功能等于不可用。
        all_threads = sorted(cp.list_threads(), reverse=True)
        for tid in all_threads[:limit]:
            events: List[Dict[str, Any]] = []
            try:
                events = list(cp.load_events(tid))
            except Exception:  # noqa: BLE001
                events = []
            state = None
            try:
                state = cp.load(tid)
            except Exception:  # noqa: BLE001
                state = None
            items.append({
                "thread_id": tid,
                "has_events": bool(events),
                "event_count": len(events),
                "finished": bool(getattr(state, "finished", False)),
                "task_type": getattr(state, "task_type", "") or "",
                "query": (getattr(state, "query", "") or "")[:60],
            })
        return {"count": len(items), "threads": items}

    @app.get("/agent/trace/{thread_id}")
    def trace(thread_id: str) -> Dict[str, Any]:
        """回放某个 thread 的完整轨迹（面板"历史回放"用）。"""
        if cp is None:
            raise HTTPException(404, "未配置 checkpointer，无轨迹可回放")
        events: List[Dict[str, Any]] = []
        try:
            events = list(cp.load_events(thread_id))
        except Exception:  # noqa: BLE001
            events = []
        state = None
        try:
            state = cp.load(thread_id)
        except Exception:  # noqa: BLE001
            state = None
        if not events and state is None:
            raise HTTPException(404, f"未找到 thread: {thread_id}")
        return {
            "thread_id": thread_id,
            "events": events,
            "state": state.to_dict() if state is not None else None,
            "summary": _summarize(events, state),
            "spans": _span_payload(_load_spans(cp, thread_id)),
        }

    @app.get("/agent/spans/{thread_id}")
    def spans_only(thread_id: str) -> Dict[str, Any]:
        """只取 span 树（甘特条 / 瓶颈 / 错误节点）。

        与 `/agent/trace/{tid}` 分开是为了让面板可以**只刷新甘特条**
        而不必重新拉一遍完整事件流 —— 回放时事件流可能有几百行。
        """
        if cp is None:
            raise HTTPException(404, "未配置 checkpointer，无 span 可回放")
        spans = _load_spans(cp, thread_id)
        if not spans:
            raise HTTPException(404, f"该 thread 没有 span: {thread_id}")
        return {"thread_id": thread_id, **_span_payload(spans)}

    def _run_and_collect(query: str, thread_id: Optional[str], resume: bool
                         ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        events = list(engine.run_stream(query, thread_id=thread_id,
                                        resume=resume))
        st = engine.last_state
        return {
            "thread_id": getattr(st, "thread_id", thread_id) if st else thread_id,
            "answer": getattr(st, "answer", "") if st else "",
            "events": events,
            "state": st.to_dict() if st is not None else None,
            "summary": _summarize(events, st),
            "spans": _span_payload(getattr(engine, "last_spans", []) or []),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @app.post("/agent/chat")
    def chat(req: AgentChatRequest) -> Dict[str, Any]:
        return _run_and_collect(req.query, req.thread_id, req.resume)

    @app.post("/agent/chat/stream")
    def chat_stream(req: AgentChatRequest) -> StreamingResponse:
        """SSE 实时推轨迹事件。

        ★ 事件是**节点粒度**实时产出的（每个 LangGraph 节点执行完就推），
          不是跑完一次性回放 —— 这是"实时轨迹"与"事后日志"的分界线。

        事件推完后再补一帧 `summary`（答案、轮数、工具调用数、耗时），
        前端拿它渲染顶部汇总条。
        """
        stream = _sse_stream(req.query, req.thread_id, req.resume)
        return StreamingResponse(stream, media_type=SSE_MEDIA_TYPE,
                                 headers=SSE_HEADERS)

    @app.get("/agent/chat/stream")
    def chat_stream_get(
        q: str = Query(..., min_length=1),
        thread_id: Optional[str] = None,
        resume: bool = False,
    ) -> StreamingResponse:
        """GET 版：`new EventSource('/agent/chat/stream?q=...')` 直接联调。"""
        return StreamingResponse(
            _sse_stream(q, thread_id, resume),
            media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)

    def _sse_stream(query: str, thread_id: Optional[str],
                    resume: bool) -> Iterator[str]:
        t0 = time.perf_counter()
        yield from stream_agent_events(
            engine.run_stream(query, thread_id=thread_id, resume=resume))
        st = engine.last_state
        yield _summary_frame(st, t0)
        yield _spans_frame(getattr(engine, "last_spans", []) or [])

    return app


def _summarize(events: List[Dict[str, Any]], state: Any) -> Dict[str, Any]:
    """从事件流里汇总出面板顶部要显示的指标。

    ★ 错误节点统计是"快速定位错误节点"的入口：
      面板可以直接跳到 `error_types` 里列出的那几类事件。
    """
    types = [e.get("type") for e in events or []]
    errors = [e for e in events or [] if e.get("type") in (
        EventType.TOOL_ERROR, EventType.ERROR, EventType.LIMIT)]
    ts = [e.get("ts") for e in events or [] if isinstance(e.get("ts"), (int, float))]
    return {
        "thread_id": getattr(state, "thread_id", "") if state else "",
        "answer": getattr(state, "answer", "") if state else "",
        "rounds": getattr(state, "rounds", 0) if state else 0,
        "tool_calls": getattr(state, "tool_calls", 0) if state else 0,
        "stop_reason": getattr(state, "stop_reason", "") if state else "",
        "event_count": len(events or []),
        "error_count": len(errors),
        "error_types": sorted({e.get("type") for e in errors}),
        "duration_ms": (round((max(ts) - min(ts)) * 1000, 1)
                        if len(ts) >= 2 else 0.0),
        "types": types,
    }


def _load_spans(cp: Any, thread_id: str) -> List[Dict[str, Any]]:
    try:
        return list(cp.load_spans(thread_id))
    except Exception:  # noqa: BLE001 - span 是旁路，取不到就当没有
        return []


def _span_payload(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """span 列表 -> 面板要的一切：树 / 甘特条 / 瓶颈 / 错误节点。

    ★ 为什么要 total_ms 而不是让前端自己算：span 的 t0/t1 是**绝对时钟**，
      前端拿去算占比时容易忽略"根 span 之外还有间隙"，统一在后端算好
      total_ms 与 pct，前后端口径才一致。
    """
    info = analyze(spans or [])
    return {
        "count": info["span_count"],
        "total_ms": info["total_ms"],
        "tree": info["tree"],
        "bottlenecks": info["bottlenecks"],
        "errors": info["errors"],
        "gantt": info["gantt"],
    }


def _spans_frame(spans: List[Dict[str, Any]]) -> str:
    """事件推完后的 span 帧：前端不用二次请求就能画甘特条。"""
    from core.sse import format_sse
    return format_sse(_span_payload(spans), event="spans")


def _summary_frame(state: Any, t0: float) -> str:
    """事件流结束后的汇总帧（前端顶部汇总条用）。"""
    from core.sse import format_sse
    payload = {
        "answer": getattr(state, "answer", "") if state else "",
        "thread_id": getattr(state, "thread_id", "") if state else "",
        "rounds": getattr(state, "rounds", 0) if state else 0,
        "tool_calls": getattr(state, "tool_calls", 0) if state else 0,
        "stop_reason": getattr(state, "stop_reason", "") if state else "",
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return format_sse(payload, event="summary")
