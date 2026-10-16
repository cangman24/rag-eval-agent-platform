# -*- coding: utf-8 -*-
"""rag1_knowledge_qa/cli.py —— 命令行入口。

★ 为什么要有 CLI：**不装 fastapi 也能干活**。
  Day 2~5 你主要在做建库、调参、跑评测，这些用命令行比起服务快得多。
  CLI 直接复用 core.service，与 HTTP 接口行为一致（同一条代码路径）。

用法：
    # 建库（需要 core/embedder.py）
    python -m rag1_knowledge_qa.cli ingest --docs data/raw

    # 单次问答（无 Key 时自动降级为检索片段）
    python -m rag1_knowledge_qa.cli ask "端侧推理时延要求是多少"

    # 流式问答（打印 SSE 帧，可用于前端联调对拍）
    python -m rag1_knowledge_qa.cli ask "..." --stream

    # 只检索不生成（调 chunk_size / top_k 时最常用）
    python -m rag1_knowledge_qa.cli retrieve "mAP 精度"

    # 健康检查
    python -m rag1_knowledge_qa.cli health
"""
# 注释可以写在这里
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

from core.service import RagService


def _load_env() -> None:
    """把 .env 读进 os.environ。

    ★ 这段必须是**惰性 + 容错**的，不能写成模块顶层 `from dotenv import load_dotenv`。

      原因（真实踩坑）：`core/` 的设计目标是「零第三方依赖、用系统 Python 就能跑
      全部单测」。如果在模块顶层 import dotenv，那么**任何 import 本模块的行为**
      （包括 `tests/test_cli.py`）都会在没装 dotenv 的解释器里直接 ImportError ——
      结果就是 Day 1 的验收项「689 项测试全绿」变成 1 个 error、总数从 689 掉到 670。

      改成函数内导入 + try/except 之后：
        · 装了 python-dotenv -> 正常读 .env
        · 没装              -> 退化为极简解析（下面手写的 KEY=VALUE），测试照常通过
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except ImportError:
        pass
    # 没有 python-dotenv 就手写一个极简解析：KEY=VALUE，忽略注释与空行。
    # 这样即使依赖没装，本地跑 CLI 也能读到 .env。
    import os
    try:
        env_path = os.path.join(os.getcwd(), ".env")
        if not os.path.exists(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as f:
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


# 让 Windows 控制台能正确输出中文（原先的 reconfigure 保留）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# --------------------------------------------------------------------- 输出
def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_health(service: RagService, args: argparse.Namespace) -> int:
    _print(service.health())
    return 0


def cmd_ask(service: RagService, args: argparse.Namespace) -> int:
    if args.stream:
        # 直接打印 SSE 帧原文：便于与前端对拍协议
        for frame in service.answer_stream(args.query):
            sys.stdout.write(frame)
        return 0
    result = service.answer(args.query)
    if args.json:
        _print(result.to_dict())
    else:
        print(f"问题: {result.query}")
        print(f"回答: {result.answer}")
        # ★ 这段不要注释掉 —— 它是「引用溯源」这个卖点的唯一可见证据。
        #   踩坑记录：这段曾被临时注释，结果 result.citations 明明有 2 条
        #   （build_citations 实测正常），终端却什么都不显示，
        #   tests/test_cli.py 的断言也随之失败。排查时很迷惑：
        #   service 层是对的，坏在展示层。
        #   引用为空只有两种情况：真拒答（refused=True），或检索无命中。
        #   而这两种情况 answer 里都会明确说"未找到"，所以不需要额外提示。
        if result.citations:
            print("\n引用:")
            for c in result.citations:
                head = f"（{c.heading_path}）" if c.heading_path else ""
                print(f"  [{c.index}] {c.chunk_id}{head}")
                print(f"      {c.snippet[:80]}")
        flag = "  [拒答]" if result.refused else ""
        print(f"\n耗时 {result.latency_ms:.0f} ms{flag}")
        if result.error:
            print(f"错误: {result.error}")
    return 0


def cmd_retrieve(service: RagService, args: argparse.Namespace) -> int:
    from core.service import _hits_from_retrieval
    raw = service.retrieve(args.query)
    hits = _hits_from_retrieval(raw, service.doc_lookup, args.top_k)
    if args.json:
        _print({"query": args.query, "count": len(hits), "hits": hits})
    else:
        print(f"query: {args.query}   命中 {len(hits)} 条")
        for i, h in enumerate(hits, start=1):
            print(f"  [{i}] {h['chunk_id']}  score={h['score']:.4f}")
            print(f"      {h['text'][:100]}")
    return 0


def cmd_ingest(service: Optional[RagService], args: argparse.Namespace) -> int:
    """建库独立于 service（不需要 LLM）。"""
    from core.ingest import IngestConfig, ingest_dir, save_chunks, save_manifest

    try:
        from core.embedder import BgeEmbedder  # type: ignore
    except ImportError:
        print("建库需要 core/embedder.py。请按 rag1_knowledge_qa/bootstrap.py "
              "顶部 docstring 的参考实现补全后再试。", file=sys.stderr)
        return 2

    embedder = BgeEmbedder(args.model)
    cfg = IngestConfig(chunk_size=args.chunk_size, overlap=args.overlap,
                       parent_size=args.parent_size,
                       embedding_model=args.model)
    result = ingest_dir(args.docs, embedder.encode_batch, config=cfg, verbose=True)
    save_chunks(result, args.chunks_out)
    save_manifest(result, args.manifest_out)
    if hasattr(result.vector_store, "persist"):
        result.vector_store.persist(args.vectors_out)
    _print(result.to_manifest())
    return 0


# --------------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag1_knowledge_qa.cli",
        description="垂直领域知识问答系统 · 命令行入口")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="健康检查")

    a = sub.add_parser("ask", help="问答")
    a.add_argument("query")
    a.add_argument("--stream", action="store_true", help="打印 SSE 帧")
    a.add_argument("--json", action="store_true", help="输出 JSON")

    r = sub.add_parser("retrieve", help="只检索不生成")
    r.add_argument("query")
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--json", action="store_true")

    i = sub.add_parser("ingest", help="建库（需要 core/embedder.py）")
    i.add_argument("--docs", default="data/raw")
    i.add_argument("--model", default="models/bge-m3")
    i.add_argument("--chunk-size", type=int, default=512)
    i.add_argument("--overlap", type=int, default=64)
    i.add_argument("--parent-size", type=int, default=2048)
    i.add_argument("--chunks-out", default="data/vectors/chunks.json")
    i.add_argument("--vectors-out", default="data/vectors/vectors.json")
    i.add_argument("--manifest-out", default="reports/ingest_manifest.json")

    return p


def main(argv: Optional[list] = None) -> int:
    # ★ 在任何装配动作之前先把 .env 读进来 —— 否则 bootstrap 里
    #   `os.environ.get("DEEPSEEK_API_KEY")` 会是空的，服务静默降级成
    #   "未配置 LLM 模式"（只回检索片段），而你以为是模型答得不好。
    _load_env()

    args = build_parser().parse_args(argv)

    if args.command == "ingest":
        return cmd_ingest(None, args)

    # 其余命令都需要 service —— 走 bootstrap（真机装配）
    try:
        from rag1_knowledge_qa.bootstrap import build_production_service
        service = build_production_service()
    except Exception as exc:  # noqa: BLE001
        print(f"装配服务失败: {exc}\n"
              f"提示：先用 `python -m rag1_knowledge_qa.cli ingest --docs data/raw` "
              f"建库，并确认 core/embedder.py 与 core/reranker.py 已补全。",
              file=sys.stderr)
        return 2

    handlers = {"health": cmd_health, "ask": cmd_ask, "retrieve": cmd_retrieve}
    return handlers[args.command](service, args)


if __name__ == "__main__":
    raise SystemExit(main())
