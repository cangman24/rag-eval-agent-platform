# -*- coding: utf-8 -*-
"""tests/test_mcp_server.py —— MCP server 协议层单测（零依赖，离线可跑）。

覆盖三类风险：
    ① 协议形态：initialize / tools/list / tools/call 的请求响应结构
    ② 错误语义：协议错误用 JSON-RPC error，工具失败用 isError（易搞混）
    ③ 传输边界：stdio 逐行解析、坏行不断连接、通知不应答
"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp import ToolRegistry  # noqa: E402
from core.mcp_server import (  # noqa: E402
    INVALID_REQUEST, METHOD_NOT_FOUND, McpServer, serve_stdio,
)


def make_registry():
    reg = ToolRegistry()

    @reg.tool(description="检索知识库")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 关键词
        top_k: 条数
        """
        return {"count": 1, "hits": [{"id": "c1", "text": f"关于{query}"}]}

    @reg.tool(description="会炸的工具")
    def boom(x: str = ""):
        """会炸

        x: 参数
        """
        raise RuntimeError("工具内部错误")

    @reg.tool(description="敏感工具", sensitive=True)
    def send_mail(to: str):
        """发邮件

        to: 收件人
        """
        return {"sent": True, "to": to}

    return reg


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.server = McpServer(make_registry())

    def test_initialize(self):
        resp = self.server.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(resp["id"], 1)
        result = resp["result"]
        self.assertIn("protocolVersion", result)
        self.assertEqual(result["capabilities"]["tools"]["listChanged"], False)
        self.assertEqual(result["serverInfo"]["name"], "kb-mcp-server")

    def test_tools_list_shape(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        self.assertEqual(names, {"kb_search", "boom", "send_mail"})
        kb = next(t for t in tools if t["name"] == "kb_search")
        self.assertIn("inputSchema", kb)
        self.assertEqual(kb["inputSchema"]["type"], "object")
        self.assertIn("query", kb["inputSchema"]["properties"])

    def test_tools_call_success(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "kb_search", "arguments": {"query": "复核"}}})
        result = resp["result"]
        self.assertFalse(result["isError"])
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertIn("复核", result["content"][0]["text"])

    def test_tools_call_unknown_tool_is_error_result(self):
        """★ 错误语义：未知工具是 isError: true 的正常返回，不是 JSON-RPC error。"""
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "不存在", "arguments": {}}})
        self.assertNotIn("error", resp)
        self.assertTrue(resp["result"]["isError"])

    def test_tools_call_tool_exception_is_error_result(self):
        """工具内部异常同样走 isError —— 模型要能读到失败原因。"""
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "boom", "arguments": {"x": "1"}}})
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("RuntimeError", resp["result"]["content"][0]["text"])

    def test_tools_call_bad_arguments_is_error_result(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "kb_search", "arguments": {"query": 123}}})
        self.assertTrue(resp["result"]["isError"])

    def test_unknown_method_is_jsonrpc_error(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 7, "method": "resources/list"})
        self.assertEqual(resp["error"]["code"], METHOD_NOT_FOUND)

    def test_missing_method_is_invalid_request(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 8})
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)

    def test_non_object_message_is_invalid_request(self):
        resp = self.server.handle(["not", "an", "object"])
        self.assertEqual(resp["error"]["code"], INVALID_REQUEST)
        self.assertIn("batch", resp["error"]["message"])

    def test_notification_gets_no_response(self):
        self.assertIsNone(self.server.handle(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_notification_with_unknown_method_no_response(self):
        self.assertIsNone(self.server.handle(
            {"jsonrpc": "2.0", "method": "notifications/whatever"}))

    def test_ping(self):
        resp = self.server.handle({"jsonrpc": "2.0", "id": 9, "method": "ping"})
        self.assertEqual(resp["result"], {})

    def test_internal_error_is_caught_not_raised(self):
        """协议层任何意外都转 -32603，不能炸掉 stdio 连接。"""
        server = McpServer(make_registry())
        server.registry = None      # 制造一个内部错误
        resp = server.handle({"jsonrpc": "2.0", "id": 10, "method": "tools/list"})
        self.assertIn("error", resp)


class TestWhitelist(unittest.TestCase):
    """MCP 不是绕过分权的后门：server 也受白名单约束。"""

    def test_list_filtered_by_allowed(self):
        server = McpServer(make_registry(), allowed=["kb_search"])
        resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual([t["name"] for t in resp["result"]["tools"]], ["kb_search"])

    def test_call_outside_allowed_is_error_result(self):
        server = McpServer(make_registry(), allowed=["kb_search"])
        resp = server.handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "send_mail", "arguments": {"to": "a@b.c"}}})
        self.assertTrue(resp["result"]["isError"])


class TestStdioTransport(unittest.TestCase):
    def test_lines_roundtrip(self):
        server = McpServer(make_registry())
        reader = io.StringIO(
            '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
            '{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
            '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
            "not json at all\n")
        writer = io.StringIO()
        serve_stdio(server, reader=reader, writer=writer)
        lines = [l for l in writer.getvalue().splitlines() if l.strip()]
        self.assertEqual(len(lines), 3)     # 通知与坏行不产多余输出
        import json
        first = json.loads(lines[0])
        self.assertEqual(first["id"], 1)
        self.assertIn("protocolVersion", first["result"])
        bad = json.loads(lines[2])
        self.assertEqual(bad["error"]["code"], -32700)   # 坏行回 PARSE_ERROR

    def test_eof_returns_zero(self):
        server = McpServer(make_registry())
        code = serve_stdio(server, reader=io.StringIO(""), writer=io.StringIO())
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
