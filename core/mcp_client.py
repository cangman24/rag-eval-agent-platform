# -*- coding: utf-8 -*-
"""core/mcp_client.py —— MCP 客户端：发现工具、调用工具。

与 `core/mcp_server.py` 配对。客户端只依赖**传输层是一个
"发一条消息、收一条响应"的可调用对象"，因此：

    · InProcessTransport  同进程直连 server.handle（测试用，零开销）
    · StdioTransport      起子进程，逐行 JSON（真实本地部署形态）
    · HttpTransport       POST JSON-RPC（远程部署形态）

Agent 侧用法：

    client = McpClient(StdioTransport([sys.executable, "-m", "core.mcp_server"]))
    tools = client.list_tools()                 # 动态发现，不需要 import 工具代码
    result = client.call_tool("kb_search", {"query": "复核流程"})
    if result["isError"]:
        ...                                      # 工具失败是正常业务分支

★ 为什么值得自建客户端而不只用 requests：
    客户端要的只是"发 JSON、收 JSON"——引一个 HTTP 库来做这件事，
    是把依赖成本花在最不差异化的地方。协议细节（id 配对、通知不应答、
    工具错误用 isError 而不是 JSON-RPC error）才是要写对的部分。
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Callable, Dict, List, Optional

Transport = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


class McpClientError(Exception):
    """客户端侧错误：协议不通、响应无法配对。"""


class McpToolError(Exception):
    """工具执行失败（isError: true）——携带服务端返回的文本原因。"""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"工具 {tool} 调用失败: {message}")
        self.tool = tool
        self.message = message


class McpClient:
    """最小 MCP 客户端。transport 注入，协议逻辑可离线单测。"""

    def __init__(self, transport: Transport, *,
                 protocol_version: str = "2025-06-18") -> None:
        self._transport = transport
        self._protocol_version = protocol_version
        self._next_id = 1
        self.server_info: Dict[str, Any] = {}

    # ------------------------------------------------------------ 协议基础
    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        req_id = self._next_id
        self._next_id += 1
        message: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            message["params"] = params

        resp = self._transport(message)
        if resp is None:
            raise McpClientError(f"{method}: 收到空响应（连接是否已关闭？）")
        if not isinstance(resp, dict):
            raise McpClientError(f"{method}: 响应不是 JSON object")
        if resp.get("id") != req_id:
            raise McpClientError(
                f"{method}: 响应 id 不配对（发 {req_id} 收 {resp.get('id')}）")
        if "error" in resp:
            err = resp["error"] or {}
            raise McpClientError(
                f"{method}: 协议错误 {err.get('code')}: {err.get('message')}")
        return resp.get("result")

    # ------------------------------------------------------------ 高层 API
    def initialize(self) -> Dict[str, Any]:
        result = self._request("initialize", {
            "protocolVersion": self._protocol_version,
            "capabilities": {},
            "clientInfo": {"name": "kb-mcp-client", "version": "0.1.0"},
        })
        self.server_info = result.get("serverInfo", {}) if isinstance(result, dict) else {}
        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self._request("tools/list")
        if not isinstance(result, dict):
            raise McpClientError("tools/list: 返回形态不对")
        return list(result.get("tools") or [])

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """调用工具；isError: true 时抛 McpToolError（原因在 .message）。"""
        result = self._request("tools/call",
                               {"name": name, "arguments": arguments or {}})
        if not isinstance(result, dict):
            raise McpClientError("tools/call: 返回形态不对")
        if result.get("isError"):
            text = _first_text(result)
            raise McpToolError(name, text or "（无错误详情）")
        return _first_text(result)

    def ping(self) -> bool:
        return self._request("ping") == {}


def _first_text(result: Dict[str, Any]) -> str:
    """从 MCP content 数组里取第一段 text。"""
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            return str(item.get("text") or "")
    return ""


# ---------------------------------------------------------------- 传输实现
class InProcessTransport:
    """同进程传输：直接调 McpServer.handle（测试与单机演示用）。"""

    def __init__(self, server: Any) -> None:
        self._server = server

    def __call__(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._server.handle(message)


class StdioTransport:
    """子进程传输：spawn 一个 MCP server 进程，逐行 JSON 对话。

    ★ 这是真实本地部署形态（Claude Desktop / Cursor 连 MCP 就是这个方式），
      所以端到端测试直接用它——协议边界的 bug 只有在真跨进程时才暴露。
    """

    def __init__(self, argv: List[str], *, timeout_s: float = 15.0,
                 cwd: Optional[str] = None) -> None:
        self._argv = list(argv)
        self._timeout_s = timeout_s
        self._cwd = cwd
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> "StdioTransport":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8",
            cwd=self._cwd,          # -m core.mcp_server 需要仓库根在 sys.path
            env=self._clean_env(),
        )

    @staticmethod
    def _clean_env() -> Dict[str, str]:
        import os
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"     # Windows 下 stdout 默认 cp1252
        env["PYTHONUTF8"] = "1"
        return env

    def __call__(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._proc is None or self._proc.stdin is None:
            raise McpClientError("StdioTransport 未启动")
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise McpClientError("server 关闭了 stdout（进程是否已退出？）")
        return json.loads(line)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - 关闭失败不强抛
            self._proc.kill()
        finally:
            self._proc = None


class HttpTransport:
    """HTTP 传输：POST 一个 JSON-RPC body（标准库 urllib，无第三方依赖）。"""

    def __init__(self, url: str, *, timeout_s: float = 15.0) -> None:
        self._url = url
        self._timeout_s = timeout_s

    def __call__(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from urllib.request import Request, urlopen

        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        req = Request(self._url, data=body,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310 - 内网/本机演示
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode("utf-8"))


__all__ = [
    "McpClient", "McpClientError", "McpToolError",
    "InProcessTransport", "StdioTransport", "HttpTransport",
]
