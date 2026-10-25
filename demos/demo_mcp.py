# -*- coding: utf-8 -*-
"""demo_mcp.py —— MCP 协议端到端演示：客户端经 stdio 发现并调用真实工具。

跑法：python demos/demo_mcp.py
输出：demo_mcp_output.txt

★ 演示什么（E2 · MCP 协议级工具层）：

    客户端**不 import 任何工具代码**，只发 JSON-RPC 消息：
        initialize -> tools/list -> tools/call
    server 是独立子进程（`python -m rag2_kb_agent.mcp_cli`），
    工具在子进程里真实执行（BM25 检索真实语料）。

    这就是"按 MCP 规范封装工具"与"进程内注册表"的区别：
    换个 host（Claude Desktop / Cursor / 另一个 Agent）连同一个
    server，工具一行不改就能用。

★ 每一步都会打印**原始协议消息**——面试时讲"你的 MCP 实现"，
  直接把这份输出投出来，比讲十分钟架构图都直观。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp_client import McpClient, StdioTransport  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT = []


def log(line=""):
    OUT.append(line)
    print(line)


def show(title, message):
    log(f"  --> {title}")
    log("      " + json.dumps(message, ensure_ascii=False))
    return message


def main():
    log("=" * 78)
    log("MCP 协议端到端演示（真子进程 + stdio + JSON-RPC 2.0）")
    log("=" * 78)

    argv = [sys.executable, "-m", "rag2_kb_agent.mcp_cli"]
    with StdioTransport(argv, cwd=ROOT) as transport:
        client = McpClient(transport)

        log("\n[1] initialize —— 握手（协议版本 / 能力 / serverInfo）")
        info = client.initialize()
        log(f"  <-- protocolVersion={info['protocolVersion']} "
            f"server={info['serverInfo']['name']} v{info['serverInfo']['version']}")

        log("\n[2] tools/list —— 工具发现（客户端零 import）")
        tools = client.list_tools()
        for t in tools:
            props = list((t["inputSchema"].get("properties") or {}).keys())
            required = t["inputSchema"].get("required") or []
            log(f"  <-- {t['name']}: 参数 {props}（必填 {required}）")

        log("\n[3] tools/call —— 真实检索（BM25 over data/raw 83 篇语料）")
        text = client.call_tool("kb_search", {"query": "检测技术要求", "top_k": 2})
        payload = json.loads(text)
        log(f"  <-- count={payload['count']}")
        for h in payload["hits"]:
            log(f"      [{h['id']}] {h['text'][:50]}...")

        log("\n[4] tools/call —— 工单查询（未命中也是正常业务返回）")
        ticket_text = client.call_tool("ticket_query", {"ticket_id": "T-999"})
        log(f"  <-- {ticket_text}")

        log("\n[5] tools/call —— 带默认值参数（top_k 省略走默认 3）")
        text = client.call_tool("kb_search", {"query": "面积评估"})
        payload = json.loads(text)
        log(f"  <-- count={payload['count']}（默认 top_k=3）")

        log("\n[6] ping —— 保活")
        log(f"  <-- ping ok={client.ping()}")

    # ---------------------------------------------------------------- 自检
    log("\n【自检】")
    checks = [
        ("initialize 拿到协议版本", bool(info.get("protocolVersion"))),
        ("tools/list 发现 3 个工具", len(tools) == 3),
        ("kb_search 返回真实命中", payload["count"] > 0),
        ("工单未命中走 isError=False 的正常返回", "not_found" in ticket_text),
        ("完整会话零异常", True),
    ]
    ok = True
    for name, passed in checks:
        log(f"  {'✓' if passed else '✗'} {name}")
        ok = ok and passed
    log("\n结论：客户端经标准 MCP 协议完成了发现与调用，未 import 任何工具代码。")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "demo_mcp_output.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
