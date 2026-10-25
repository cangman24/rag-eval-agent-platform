# -*- coding: utf-8 -*-
"""core/mcp_server.py —— 把 ToolRegistry 暴露为**标准 MCP server**（JSON-RPC 2.0）。

对应简历里那句：「按 MCP 规范封装内部工具，搭建工具注册中心」——
在 `core/mcp.py`（自定义注册表）之上，这一层补上**协议实现**：
客户端不再需要 import 我们的代码，发 JSON-RPC 消息就能发现与调用工具。

    Agent 侧                          MCP server 侧
    ┌──────────┐   tools/list    ┌───────────────┐
    │ McpClient│ ──────────────> │   McpServer   │ ──> ToolRegistry
    │          │ <────────────── │  (本模块)     │     （core/mcp.py）
    │          │   tools/call    │               │
    └──────────┘ ──────────────> └───────────────┘

★ 与"自定义注册表"的区别（面试会被问）：
    自定义注册表 = 进程内 Python 调用，只有我们自己的 Agent 能用；
    MCP server   = 跨进程/跨语言的标准化接口，Claude Desktop、Cursor、
                  任何 MCP host 都能连上同一个工具服务，生态互操作。

★ 协议范围（刻意最小，对齐 MCP 规范的工具面）：
    initialize               握手：协议版本 + 能力声明 + serverInfo
    tools/list               工具发现：name / description / inputSchema
    tools/call               工具调用：参数经 ToolSpec 校验后执行
    ping                     保活
    notifications/*          通知（无 id，不应答）
    其余方法                 -32601 Method Not Found

★ 传输（两种都实现，零第三方依赖）：
    stdio   子进程间逐行 JSON（MCP 官方本地传输方式）
    HTTP    POST 一个 JSON-RPC  body（走标准库 http.server）

★ 错误语义（MCP 规范容易搞混的一点）：
    「协议层错误」用 JSON-RPC error 对象（未知方法、参数不是对象…）；
    「工具执行错误」（未知工具名、参数校验失败、工具内部异常）用
    tools/call 的**正常返回 + isError: true**——因为工具失败是业务结果，
    不是协议失败，模型需要读到失败原因并自己决定下一步。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .mcp import ToolRegistry

# MCP 协议版本（客户端会带自己的版本，服务端声明自己支持的版本；
# 真实实现应做协商，这里固定声明一个被广泛支持的版本）
PROTOCOL_VERSION = "2025-06-18"

SERVER_NAME = "kb-mcp-server"
SERVER_VERSION = "0.1.0"

# ---------------------------------------------------------------- JSON-RPC 2.0 错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _error(code: int, message: str, data: Any = None, req_id: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


class McpServer:
    """把一个 ToolRegistry 包装成 MCP server。

    allowed 非 None 时，tools/list 与 tools/call 都受白名单约束 ——
    与 Agent 引擎共用同一套"意图 -> 白名单"机制，MCP 不是绕过分权的后门。
    """

    def __init__(self, registry: ToolRegistry, *,
                 allowed: Optional[List[str]] = None,
                 server_name: str = SERVER_NAME,
                 server_version: str = SERVER_VERSION) -> None:
        self.registry = registry
        self.allowed = list(allowed) if allowed is not None else None
        self.server_name = server_name
        self.server_version = server_version
        self._initialized = False

    # ------------------------------------------------------------ 协议方法
    def handle(self, message: Any) -> Optional[Dict[str, Any]]:
        """处理一条 JSON-RPC 消息。

        返回响应 dict；通知（无 id）返回 None。
        本方法**不抛协议异常**——任何意外都转成 -32603，
        因为 stdio 循环里一条消息的崩溃不该带走整个连接。
        """
        if isinstance(message, list):
            # JSON-RPC 允许批量，但 MCP 规范明确不建议（stdio 上无法可靠
            # 区分消息边界与响应顺序）。直接拒绝，逼客户端逐条发。
            return _error(INVALID_REQUEST, "MCP 不支持批量请求（batch），请逐条发送")
        if not isinstance(message, dict):
            return _error(INVALID_REQUEST, "消息必须是 JSON object")

        req_id = message.get("id")
        is_notification = "id" not in message
        method = message.get("method")
        if not isinstance(method, str) or not method:
            if is_notification:
                return None
            return _error(INVALID_REQUEST, "缺少 method 字段", req_id=req_id)

        try:
            result = self._dispatch(method, message.get("params"))
        except _MethodNotFound as exc:
            if is_notification:
                return None
            return _error(METHOD_NOT_FOUND, str(exc), req_id=req_id)
        except _InvalidParams as exc:
            if is_notification:
                return None
            return _error(INVALID_PARAMS, str(exc), req_id=req_id)
        except Exception as exc:  # noqa: BLE001 - 协议层的最后防线
            if is_notification:
                return None
            return _error(INTERNAL_ERROR, f"{type(exc).__name__}: {exc}", req_id=req_id)

        if result is _NO_RESPONSE or is_notification:
            return None
        return _result(req_id, result)

    def _dispatch(self, method: str, params: Any) -> Any:
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method.startswith("notifications/"):
            # 通知：initialized / cancelled / progress 等，一律静默接受
            return _NO_RESPONSE
        if method == "tools/list":
            return {"tools": self.list_tools()}
        if method == "tools/call":
            return self.call_tool(params)
        raise _MethodNotFound(f"未知方法: {method}")

    # ------------------------------------------------------------ 各方法
    def _initialize(self, params: Any) -> Dict[str, Any]:
        if params is not None and not isinstance(params, dict):
            raise _InvalidParams("initialize 的 params 必须是 object")
        self._initialized = True
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.server_name,
                           "version": self.server_version},
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """tools/list：返回 MCP 形态的工具清单。"""
        return self.registry.mcp_tools(self.allowed)

    def call_tool(self, params: Any) -> Dict[str, Any]:
        """tools/call：执行工具，结果按 MCP content 形态返回。

        工具侧的任何失败都走 isError: true 的正常返回（见模块 docstring）。
        """
        if not isinstance(params, dict):
            raise _InvalidParams("tools/call 的 params 必须是 object")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            raise _InvalidParams("tools/call 缺少 name")
        if not isinstance(arguments, dict):
            raise _InvalidParams("tools/call 的 arguments 必须是 object")

        try:
            result = self.registry.call(name, arguments, self.allowed)
        except Exception as exc:  # noqa: BLE001 - 工具失败是业务结果，不是协议错误
            return {"content": [{"type": "text",
                                 "text": f"{type(exc).__name__}: {exc}"}],
                    "isError": True}
        return {"content": [{"type": "text",
                             "text": json.dumps(result, ensure_ascii=False,
                                                default=str)}],
                "isError": False}


class _MethodNotFound(Exception):
    pass


class _InvalidParams(Exception):
    pass


_NO_RESPONSE = object()   # 哨兵：通知类消息不应答


# ---------------------------------------------------------------- 传输：stdio
def serve_stdio(server: McpServer, reader: Any = None, writer: Any = None) -> int:
    """stdio 传输：逐行读 JSON-RPC 消息，逐行写响应。

    reader/writer 可注入（测试用 StringIO）；默认 sys.stdin/stdout。
    读到底（EOF）返回 0；单行解析失败回 -32700 不断连接。
    """
    import sys

    reader = reader if reader is not None else sys.stdin
    writer = writer if writer is not None else sys.stdout

    for line in reader:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            resp = _error(PARSE_ERROR, "JSON 解析失败")
        else:
            resp = server.handle(message)
        if resp is not None:
            writer.write(json.dumps(resp, ensure_ascii=False) + "\n")
            writer.flush()
    return 0


# ---------------------------------------------------------------- 传输：HTTP
def make_http_handler(server: McpServer) -> type:
    """返回一个 BaseHTTPRequestHandler 子类：POST body 即 JSON-RPC 消息。

    用标准库 http.server，不引第三方依赖；单线程够演示与内网使用，
    生产换 uvicorn/fastapi 时协议层（McpServer.handle）一行不用改。
    """
    from http.server import BaseHTTPRequestHandler

    class _McpHttpHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - http.server 的回调名
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                message = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
                resp: Optional[Dict[str, Any]] = _error(PARSE_ERROR, "JSON 解析失败")
            else:
                resp = server.handle(message)
            if resp is None:
                # 通知：HTTP 204，无 body
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - 健康检查
            if self.path.rstrip("/") in ("", "/health"):
                body = json.dumps({"status": "ok",
                                   "server": server.server_name}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # 静默默认访问日志（演示时输出已经很吵了）
            return

    return _McpHttpHandler


def serve_http(server: McpServer, host: str = "127.0.0.1", port: int = 8765) -> None:
    """起一个 HTTP MCP server（阻塞）。"""
    from http.server import ThreadingHTTPServer

    handler = make_http_handler(server)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"MCP HTTP server listening on http://{host}:{port} "
          f"({len(server.list_tools())} tools)")
    httpd.serve_forever()


# ---------------------------------------------------------------- CLI 入口
def _demo_registry() -> ToolRegistry:
    """CLI 自带的演示 registry（真实部署时传入业务 registry）。"""
    reg = ToolRegistry()

    @reg.tool(description="检索知识库，返回与查询最相关的片段")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 检索关键词
        top_k: 返回条数
        """
        return {"count": 1,
                "hits": [{"id": "demo#c1", "text": f"关于「{query}」的演示结果"}]}

    return reg


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MCP server（JSON-RPC 2.0）")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server = McpServer(_demo_registry())
    if args.transport == "http":
        serve_http(server, args.host, args.port)
        return 0
    return serve_stdio(server)


if __name__ == "__main__":
    raise SystemExit(main())
