# -*- coding: utf-8 -*-
"""rag1_knowledge_qa/server.py —— uvicorn 启动入口。

用法：
    uvicorn rag1_knowledge_qa.server:app --host 0.0.0.0 --port 8000
    # 或
    python -m rag1_knowledge_qa.server

★ 为什么单独放一个 server.py 而不是直接用 api_fastapi:app：

  `api_fastapi.py` 用的是 **app factory 模式**（`build_app(service)`）——
  这是对的（便于测试时注入假 service），但 uvicorn 的 `--reload` 和
  `module:app` 约定需要模块级 `app` 对象。

  这里做两件事把它接起来：
    ① 模块级构造 service（走 bootstrap），再 build_app
    ② 把装配失败变成**可读的启动错误**，而不是一大段 import traceback

  另外注意：装配本身会加载 BGE 模型（约 6 秒），所以启动慢是正常的。
  生产环境应当预热（见 docker-compose 的 healthcheck）。
"""
from __future__ import annotations

import os
import sys
from typing import Any

from .api_fastapi import build_app


def _load_env() -> None:
    """与 cli.py 一致：惰性 + 容错地加载 .env。

    ★ 不能写成模块顶层 `from dotenv import load_dotenv`：
      那会让"没装 python-dotenv 的解释器"连 import 本模块都失败，
      与 core/ 的零依赖可测设计冲突（cli.py 上踩过这个坑）。
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except ImportError:
        pass
    try:
        path = os.path.join(os.getcwd(), ".env")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


def create_app() -> Any:
    """构造 FastAPI 应用（供 uvicorn factory 模式或测试使用）。"""
    _load_env()
    from .bootstrap import build_production_service

    service = build_production_service()
    return build_app(service)


def _fatal(message: str, exc: BaseException | None = None) -> None:
    print("=" * 74, file=sys.stderr)
    print("启动失败", file=sys.stderr)
    print("=" * 74, file=sys.stderr)
    print(message, file=sys.stderr)
    if exc is not None:
        print(f"\n原始错误：{type(exc).__name__}: {exc}", file=sys.stderr)
    print("\n排查顺序：", file=sys.stderr)
    print("  1. 模型是否就绪     ls models/bge-m3", file=sys.stderr)
    print("  2. 索引是否已建     ls data/vectors/chunks.json", file=sys.stderr)
    print("     未建则：python -m rag1_knowledge_qa.cli ingest --docs data/raw",
          file=sys.stderr)
    print("  3. 依赖是否装全     pip install -e \".[ml,serve]\"", file=sys.stderr)
    print("     注意 transformers 必须是 4.44.2（见 pyproject.toml）", file=sys.stderr)
    print("=" * 74, file=sys.stderr)


try:
    app = create_app()
except Exception as exc:  # noqa: BLE001 - 启动期任何异常都要给出可读提示
    _fatal(
        "装配服务失败。最常见的原因是"
        "（a）BGE 模型未下载、（b）索引未建、（c）缺 fastapi/torch。",
        exc,
    )
    raise


def main() -> int:
    import uvicorn

    host = os.environ.get("RAG_HOST", "0.0.0.0")
    port = int(os.environ.get("RAG_PORT", "8000"))
    reload = os.environ.get("RAG_RELOAD", "0") == "1"
    uvicorn.run("rag1_knowledge_qa.server:app", host=host, port=port,
                reload=reload, log_level=os.environ.get("RAG_LOG_LEVEL", "info"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
