# -*- coding: utf-8 -*-
"""tests/test_mcp_client.py —— MCP 客户端单测（零依赖，离线可跑）。

★ 压轴用例是真·跨进程端到端：spawn `python -m core.mcp_server` 子进程，
  经 stdio 完成 initialize -> tools/list -> tools/call 全流程。
  协议边界的 bug（id 配对、换行、编码、通知处理）只有在真跨进程时才暴露，
  进程内替身发现不了——和 LangGraph 那次教训同一模式。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp_server import McpServer  # noqa: E402
from core.mcp_client import (  # noqa: E402
    InProcessTransport, McpClient, McpClientError, McpToolError,
    StdioTransport,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_server():
    from core.mcp import ToolRegistry
    reg = ToolRegistry()

    @reg.tool(description="检索知识库")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 关键词
        top_k: 条数
        """
        return {"count": 1, "hits": [{"id": "c1", "text": f"关于{query}"}]}

    return McpServer(reg)


class TestInProcessClient(unittest.TestCase):
    def setUp(self):
        self.client = McpClient(InProcessTransport(make_server()))

    def test_initialize_records_server_info(self):
        result = self.client.initialize()
        self.assertIn("protocolVersion", result)
        self.assertEqual(self.client.server_info["name"], "kb-mcp-server")

    def test_list_tools(self):
        self.client.initialize()
        tools = self.client.list_tools()
        self.assertEqual([t["name"] for t in tools], ["kb_search"])

    def test_call_tool_returns_text(self):
        self.client.initialize()
        text = self.client.call_tool("kb_search", {"query": "复核"})
        self.assertIn("复核", text)

    def test_call_tool_error_raises_with_reason(self):
        self.client.initialize()
        with self.assertRaises(McpToolError) as ctx:
            self.client.call_tool("不存在", {})
        self.assertIn("不存在", str(ctx.exception))

    def test_id_mismatch_detected(self):
        """响应 id 与请求不配对必须报错——协议层最常见的事故。"""
        def bad_transport(message):
            return {"jsonrpc": "2.0", "id": 999, "result": {}}
        client = McpClient(bad_transport)
        with self.assertRaises(McpClientError):
            client.ping()

    def test_protocol_error_surfaces_message(self):
        def bad_transport(message):
            return {"jsonrpc": "2.0", "id": message["id"],
                    "error": {"code": -32601, "message": "未知方法"}}
        client = McpClient(bad_transport)
        with self.assertRaises(McpClientError) as ctx:
            client.ping()
        self.assertIn("-32601", str(ctx.exception))


class TestStdioEndToEnd(unittest.TestCase):
    """★ 真跨进程：spawn server 子进程，走完整 MCP 会话。"""

    @classmethod
    def setUpClass(cls):
        cls.transport = StdioTransport(
            [sys.executable, "-m", "core.mcp_server"], cwd=ROOT)
        cls.transport.start()
        cls.client = McpClient(cls.transport)

    @classmethod
    def tearDownClass(cls):
        cls.transport.close()

    def test_full_session(self):
        info = self.client.initialize()
        self.assertIn("protocolVersion", info)

        tools = self.client.list_tools()
        self.assertEqual([t["name"] for t in tools], ["kb_search"])

        text = self.client.call_tool("kb_search", {"query": "复核流程", "top_k": 2})
        self.assertIn("复核流程", text)

        self.assertTrue(self.client.ping())

    def test_tool_error_over_stdio(self):
        self.client.initialize()
        with self.assertRaises(McpToolError):
            self.client.call_tool("不存在", {})

    def test_dead_server_detected(self):
        """server 挂掉后，客户端应得到明确错误而不是挂死。"""
        transport = StdioTransport(
            [sys.executable, "-c", "import sys; sys.exit(1)"], cwd=ROOT)
        transport.start()
        client = McpClient(transport)
        with self.assertRaises(McpClientError):
            client.ping()
        transport.close()


if __name__ == "__main__":
    unittest.main()
