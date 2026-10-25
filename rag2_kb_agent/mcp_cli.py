# -*- coding: utf-8 -*-
"""rag2_kb_agent/mcp_cli.py —— 用**真实业务 registry** 起 MCP server。

与 core/mcp_server.py 的分工（接缝设计）：
    core/mcp_server.py   只管协议（JSON-RPC 2.0），零依赖，可离线单测
    本文件               只管装配：语料 -> 索引 -> ToolRegistry -> McpServer

用法：
    python -m rag2_kb_agent.mcp_cli --transport stdio     # 本地子进程传输
    python -m rag2_kb_agent.mcp_cli --transport http --port 8765

★ 索引说明：走 data/raw 的 83 篇真实语料 + 标准库 BM25，
  不依赖 GPU/嵌入模型——MCP 协议层与检索质量解耦，
  换 BGE-M3 + ChromaDB 时只改 build_index 一个函数。
  生产部署把 fake_embed_batch 换成 BgeEmbedder.encode_batch 即可。
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence

from core.bm25 import BM25Index  # noqa: F401 - 类型说明用（IngestResult.bm25 即此类型）
from core.ingest import IngestConfig, IngestResult, ingest_dir
from core.mcp import ToolRegistry
from core.mcp_server import McpServer, serve_http, serve_stdio
from core.text import tokenize
from core.vectorstore import l2_normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")

# 演示工单库：真实部署时来自工单系统 API，这里内置几条样例
DEMO_TICKETS = {
    "T-100": {"ticket_id": "T-100", "status": "closed", "owner": "审批组",
              "summary": "检测精度复测申请，已闭环"},
    "T-101": {"ticket_id": "T-101", "status": "open", "owner": "运维组",
              "summary": "边缘设备推理时延超标待定位"},
}


def fake_embed_batch(texts: Sequence[str]) -> List[List[float]]:
    """批量假嵌入（字符哈希 + L2 归一化）。真实项目换成 BGE-M3。"""
    out = []
    for t in texts:
        v = [0.0] * 64
        for tok in tokenize(t):
            h = sum(ord(c) * (i + 1) for i, c in enumerate(tok))
            v[h % 64] += 1.0
        out.append(l2_normalize(v))
    return out


def build_index(raw_dir: str = RAW_DIR) -> IngestResult:
    """真实语料 -> 父子块 + BM25 索引（IngestResult 即索引视图）。

    build_registry_from_index 只认 bm25 / chunks / child_text / doc_text
    四个属性 —— IngestResult 恰好全有，这就是接缝设计的用处。
    """
    cfg = IngestConfig(
        chunk_size=512, overlap=64, parent_size=2048, min_chunk_tokens=8,
        embedding_model="fake-hash-64d（演示；生产换 BGE-M3）",
    )
    return ingest_dir(raw_dir, fake_embed_batch, config=cfg, verbose=False)


def build_server(allowed: Optional[List[str]] = None) -> McpServer:
    """装配：真实索引 -> 业务工具 registry -> MCP server。"""
    from rag2_kb_agent.app import build_registry_from_index

    index = build_index()
    # IngestResult 没有 tickets 字段；工单库由本文件注入（演示数据）
    index.tickets = dict(DEMO_TICKETS)  # type: ignore[attr-defined]
    registry: ToolRegistry = build_registry_from_index(index)
    return McpServer(registry, allowed=allowed)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="业务 registry 的 MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server = build_server()
    if args.transport == "http":
        serve_http(server, args.host, args.port)
        return 0
    return serve_stdio(server)


if __name__ == "__main__":
    raise SystemExit(main())
