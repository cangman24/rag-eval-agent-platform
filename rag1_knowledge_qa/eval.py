# -*- coding: utf-8 -*-
"""rag1_knowledge_qa/eval.py —— 一条命令跑完整评测。

用法：
    # 完整：四档消融 + LLM-as-Judge
    python -m rag1_knowledge_qa.eval --eval-set data/eval/eval.jsonl \\
        --ablation --judge --out reports

    # 只跑检索消融（不花钱）
    python -m rag1_knowledge_qa.eval --eval-set data/eval/eval.jsonl --ablation

    # 看最近一次 run 的指标
    python -m rag1_knowledge_qa.eval --show

    # 对比两次 run（改了参数后看有没有变好）
    python -m rag1_knowledge_qa.eval --compare reports/run-A reports/run-B

★ 为什么做成命令而不是脚本：
  Day 3 你要反复跑（改 chunk_size / 改召回参数 / 改 prompt），
  命令化之后每次都是同一条操作、同一套口径、同一份指纹，
  报告落在 reports/<run_id>/ 下可以逐次对比 —— 这才叫"可复现评测数据"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, List, Optional

from core.eval_runner import (
    EvalRunConfig, compare_runs, latest_run_dir, load_items_or_raise, run_eval,
)
from core.pipeline import ablation_configs


# --------------------------------------------------------------------- 装载
def _load_service(args: argparse.Namespace) -> Any:
    """装配真实检索服务（需要 BGE 模型与已建好的索引）。"""
    from rag1_knowledge_qa.bootstrap import build_production_service
    return build_production_service()


def _build_factory(service: Any, args: argparse.Namespace) -> Callable:
    """`cfg -> (query -> 检索结果)`，按档切换链路开关。

    ★ 关键：这里通过覆盖 retrieval_config 实现"逐路可开关"，
      而不是重写一遍检索逻辑 —— 保证消融跑的就是线上同一条链路。
    """
    base = service.retrieval_config

    def factory(cfg: Any) -> Callable[[str], Any]:
        # 把消融配置合并到基础配置上
        merged = type(base)(**{
            **{k: getattr(base, k) for k in vars(base)},
            "use_vector": cfg.use_vector,
            "use_bm25": cfg.use_bm25,
            "use_rerank": cfg.use_rerank,
            "use_rules": cfg.use_rules,
            "top_k": cfg.top_k,
            "recall_k": cfg.recall_k,
        })

        def retrieve(query: str) -> Any:
            return service.retriever.retrieve(query, merged)

        return retrieve

    return factory


def _build_answer_and_judge(service: Any, args: argparse.Namespace):
    """构造 answer_fn 与 judge_fn（含把检索片段喂给评委）。"""
    if not args.judge and not args.answers:
        return None, None, None

    from core.judge import LLMJudge, context_from_hits
    from core.service import _hits_from_retrieval

    hits_cache: dict = {}

    def answer_fn(query: str) -> str:
        result = service.answer(query)
        hits_cache[query] = result.hits
        return result.answer

    judge_fn = None
    judge_report = None
    if args.judge:
        if service.llm is None:
            print("[warn] 未配置 LLM，无法做 LLM-as-Judge；"
                  "只跑拒答正确率。", file=sys.stderr)
        else:
            def context_provider(query: str) -> str:
                # 复用 answer_fn 已算出的 hits，避免重复检索
                hits = hits_cache.get(query)
                if hits is None:
                    raw = service.retrieve(query)
                    hits = _hits_from_retrieval(raw, service.doc_lookup,
                                                service.config.top_k)
                return context_from_hits(hits)

            judge = LLMJudge(service.llm, context_provider=context_provider)
            judge_fn = judge
            judge_report = judge.report

    return answer_fn, judge_fn, judge_report


# --------------------------------------------------------------------- 子命令
def cmd_run(args: argparse.Namespace) -> int:
    items = load_items_or_raise(args.eval_set)
    service = _load_service(args)
    factory = _build_factory(service, args)
    configs = ablation_configs(top_k=args.top_k, recall_k=args.recall_k) \
        if args.ablation else ablation_configs(top_k=args.top_k,
                                               recall_k=args.recall_k)[2:3]

    answer_fn, judge_fn, judge_report = _build_answer_and_judge(service, args)

    ingest_meta = {}
    for attr in ("stats",):
        maybe = getattr(service.retriever, attr, None)
        if isinstance(maybe, dict):
            ingest_meta = maybe

    cfg = EvalRunConfig(
        eval_set_path=args.eval_set,
        out_dir=args.out,
        top_k=args.top_k,
        recall_k=args.recall_k,
        k_list=tuple(args.k_list),
        embedding_model=os.environ.get("BGE_M3_PATH", "unknown"),
        ingest_meta=ingest_meta,
        judge_enabled=bool(judge_fn),
        judge_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        notes=args.notes or "",
    )
    result = run_eval(items, configs, factory, cfg,
                      answer_fn=answer_fn, judge_fn=judge_fn,
                      judge_report=judge_report,
                      progress=lambda m: print(m))

    print()
    print("=" * 70)
    print(f"运行完成：{result.run_id}")
    print(f"  总结报告：{result.summary_path}")
    print(f"  运行指纹：{result.manifest_path}")
    print()
    for r in result.retrieval_reports:
        ov = r.overall
        print(f"  {r.config_name:<40} hit@5={ov.get('hit@5', 0):.4f}  "
              f"mrr@5={ov.get('mrr@5', 0):.4f}")
    if result.answer_metrics:
        am = result.answer_metrics
        print(f"  生成层：拒答正确率={am.get('refusal_accuracy', 0):.2f}", end="")
        # ★ 评委没跑时**绝不能**打印 0.00 ——
        #   那会被读成"忠实度 0 分 / 幻觉率 0%"，
        #   前者是灾难、后者是完美，而真相是"根本没测"。
        #   区分"测出来是 0"与"没测"是评测报告最基本的诚实要求。
        judged = am.get("judged", 0)
        if judged:
            print(f"  忠实度={am.get('groundedness_mean', 0):.2f}"
                  f"  幻觉率={am.get('hallucination_rate', 0):.2%}"
                  f"  相关={am.get('relevance_mean', 0):.2f}"
                  f"  准确={am.get('accuracy_mean', 0):.2f}"
                  f"  完整={am.get('completeness_mean', 0):.2f}"
                  f"  (judged={judged:.0f})")
        else:
            print("  忠实度=未测评  幻觉率=未测评   （未启用 LLM-as-Judge，"
                  "需 --judge 且配置 DEEPSEEK_API_KEY）")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = args.run or latest_run_dir(args.out)
    if not run_dir:
        print(f"未找到任何 run（在 {args.out} 下）", file=sys.stderr)
        return 1
    manifest_path = os.path.join(run_dir, "eval_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"未找到 manifest：{manifest_path}", file=sys.stderr)
        return 1
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)

    print(f"run_id      : {m['run_id']}")
    print(f"timestamp   : {m['timestamp']}")
    print(f"embedding   : {m['ingest'].get('embedding_model')}")
    # ★ 建库参数与指纹必须打印：否则看到指标波动时无法判断是不是换了分块参数
    ing = m.get("ingest") or {}
    if ing.get("config_fingerprint") or ing.get("chunk_size"):
        parts = []
        for key in ("chunk_size", "overlap", "parent_size"):
            if ing.get(key) is not None:
                parts.append(f"{key}={ing[key]}")
        if ing.get("config_fingerprint"):
            parts.append(f"fingerprint={ing['config_fingerprint']}")
        print(f"ingest      : {'  '.join(parts)}")
    print(f"eval_set    : {m['eval_set']['n']} 条 {m['eval_set']['by_type']}")
    if m["eval_set"].get("sha256"):
        print(f"eval sha256 : {m['eval_set']['sha256']}")
    print()
    print(f"{'配置':<42}{'hit@5':>9}{'mrr@5':>9}{'ndcg@5':>9}{'P95(ms)':>10}")
    print("-" * 80)
    for r in m.get("results", []):
        ov = r.get("overall", {})
        lat = r.get("latency", {})
        print(f"{r['config']:<42}{ov.get('hit@5', 0):>9.4f}"
              f"{ov.get('mrr@5', 0):>9.4f}{ov.get('ndcg@5', 0):>9.4f}"
              f"{lat.get('p95_ms', 0):>10.1f}")
    if m.get("answer_metrics"):
        print()
        print("生成层指标：")
        for k, v in m["answer_metrics"].items():
            print(f"  {k:<22}{v}")
    if m.get("judge"):
        print()
        print(f"评委：{m['judge']}")
    problems = (m.get("eval_set_health") or {}).get("problems") or []
    if problems:
        print()
        print(f"⚠️ 评测集体检 {len(problems)} 个问题（前 5 条）：")
        for p in problems[:5]:
            print(f"  · {p}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    text = compare_runs(args.run_a, args.run_b)
    print(text)
    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n已写入 {args.out_file}")
    return 0


# --------------------------------------------------------------------- 入口
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag1_knowledge_qa.eval",
        description="领域知识问答系统 · 评测入口（可复现、带运行指纹）")
    p.add_argument("--eval-set", default="data/eval/eval.jsonl",
                   help="评测集 JSONL 路径")
    p.add_argument("--out", default="reports", help="报告输出根目录")
    p.add_argument("--ablation", action="store_true",
                   help="跑四档消融（否则只跑默认档）")
    p.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge")
    p.add_argument("--answers", action="store_true",
                   help="只跑生成层（不启用评委）")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--recall-k", type=int, default=20)
    p.add_argument("--k-list", type=int, nargs="*", default=[5],
                   help="额外统计的 k 值，如 --k-list 5 3 10")
    p.add_argument("--notes", default="", help="本次运行的备注（写进 manifest）")
    p.add_argument("--show", action="store_true", help="只打印最近一次 run 的指标")
    p.add_argument("--run", default=None, help="配合 --show 指定 run 目录")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                   help="对比两次 run")
    p.add_argument("--out-file", default=None, help="对比结果写入文件")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        args.run_a, args.run_b = args.compare
        return cmd_compare(args)
    if args.show:
        return cmd_show(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
