# -*- coding: utf-8 -*-
"""core/sse.py —— Server-Sent Events 格式化与流式推送工具。

对应简历里的「基于 SSE 输出任务状态与工具调用过程，支撑前端轨迹追踪面板」。

★ 为什么把 SSE 格式化独立成模块：
  1. 零依赖 —— 不装 fastapi/starlette 也能单测（字节级断言协议正确性）；
  2. 协议细节容易写错（换行转义、id/retry 字段、空行收尾），
     单独测比在 API 里一起测容易定位；
  3. API 层只负责"把生成器挂到路由上"，逻辑与传输解耦。

SSE 协议要点：
    每条事件形如
        id: 12\\n
        event: tool_call\\n
        data: {"tool":"kb_search"}\\n
        \\n                      <- 空行表示事件结束
    data 内部换行必须逐行加 `data: ` 前缀，否则浏览器解析会截断。
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional

# 标准 SSE 媒体类型
SSE_MEDIA_TYPE = "text/event-stream; charset=utf-8"

# 默认心跳间隔（秒）—— 长任务里防止中间层断连
DEFAULT_HEARTBEAT_S = 15.0


def format_sse(
    data: Any,
    event: Optional[str] = None,
    event_id: Optional[str] = None,
    retry_ms: Optional[int] = None,
) -> str:
    """把一条数据格式化成 SSE 帧（含结尾空行）。

    Args:
        data: 字符串或可 JSON 序列化对象；对象会被序列化成 JSON
        event: 事件名（前端用 addEventListener 监听）
        event_id: 事件 id（浏览器断线重连时通过 Last-Event-ID 回传）
        retry_ms: 建议的重连间隔
    """
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)

    lines: List[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry_ms is not None:
        lines.append(f"retry: {int(retry_ms)}")

    # ★ data 里的换行必须逐行加前缀，否则一条消息会被拆成多条
    for line in payload.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")          # 空行 = 事件结束
    lines.append("")          # 末行留空，join 后形成 "\n\n"
    return "\n".join(lines)


def format_comment(text: str) -> str:
    """SSE 注释帧（`: ...`）。

    用途：① 心跳保活；② 中间代理不该缓存这类帧。
    """
    safe = text.replace("\n", " ")
    return f": {safe}\n\n"


def format_heartbeat(ts: Optional[float] = None) -> str:
    if ts is None:
        return format_comment("heartbeat")
    return format_comment(f"heartbeat {ts:.3f}")


def iter_sse(
    events: Iterable[Dict[str, Any]],
    id_field: str = "seq",
    event_field: str = "type",
    data_field: str = "data",
    heartbeat_every: int = 0,
) -> Iterator[str]:
    """把事件字典序列化成 SSE 帧流。

    Args:
        events: [{"type": "tool_call", "data": {...}}, ...]
            —— 与 core.agent.make_event 的输出结构一致
        heartbeat_every: 每 N 条事件插入一次心跳（0 = 不插）
    """
    for i, ev in enumerate(events):
        if heartbeat_every and i and i % heartbeat_every == 0:
            yield format_heartbeat()
        if not isinstance(ev, dict):
            yield format_sse(ev, event="message")
            continue
        yield format_sse(
            data=ev.get(data_field, ev),
            event=ev.get(event_field) or "message",
            event_id=str(ev.get(id_field)) if ev.get(id_field) is not None else None,
        )


def parse_sse(text: str) -> List[Dict[str, Any]]:
    """解析 SSE 文本成事件列表。

    ★ 这是给**测试**用的（也方便你写前端联调脚本时验证后端输出）。
      有了它就能做往返断言：序列化 -> 解析 -> 内容一致。
    """
    events: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}
    data_lines: List[str] = []

    def flush() -> None:
        nonlocal cur, data_lines
        if not cur and not data_lines:
            return
        ev: Dict[str, Any] = {"data": "\n".join(data_lines)}
        if "event" in cur:
            ev["event"] = cur["event"]
        if "id" in cur:
            ev["id"] = cur["id"]
        if "retry" in cur:
            ev["retry"] = cur["retry"]
        events.append(ev)
        cur, data_lines = {}, []

    for raw in text.split("\n"):
        if raw == "":
            flush()
            continue
        if raw.startswith(":"):
            events.append({"comment": raw[1:].strip()})
            continue
        field, _, value = raw.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data_lines.append(value)
        else:
            cur[field] = value
    flush()
    return events


class SSEStream:
    """带序号与统计的 SSE 发射器。

    序号用于断线重连（浏览器会带 Last-Event-ID 回来，
    服务端据此续发 —— 长任务场景下这是必需的）。
    """

    def __init__(self, event_field: str = "type") -> None:
        self._seq = 0
        self._count = 0
        self.event_field = event_field

    def emit(self, event_type: str, **data: Any) -> str:
        self._seq += 1
        self._count += 1
        return format_sse(data=data, event=event_type, event_id=str(self._seq))

    def comment(self, text: str) -> str:
        return format_comment(text)

    def heartbeat(self, ts: Optional[float] = None) -> str:
        return format_heartbeat(ts)

    @property
    def last_event_id(self) -> int:
        return self._seq

    @property
    def count(self) -> int:
        return self._count

    def emit_many(self, events: Iterable[Dict[str, Any]]) -> Iterator[str]:
        for ev in events:
            if isinstance(ev, dict):
                yield self.emit(ev.get(self.event_field) or "message",
                                **(ev.get("data") or {}))
            else:
                yield self.emit("message", value=ev)


def stream_agent_events(events: Iterable[Dict[str, Any]]) -> Iterator[str]:
    """把 `core.agent` 的事件流直接转成 SSE。

    这是 API 层的一行接线：
        return StreamingResponse(stream_agent_events(run.events),
                                 media_type=SSE_MEDIA_TYPE)
    """
    stream = SSEStream()
    for ev in events:
        if not isinstance(ev, dict):
            yield stream.emit("message", value=str(ev))
            continue
        payload = dict(ev.get("data") or {})
        payload.setdefault("ts", ev.get("ts"))
        yield stream.emit(ev.get("type") or "message", **payload)
