# -*- coding: utf-8 -*-
"""校验评测报告的「数字证据链」：语料版本 + 模型归属 + 向量维度。

为什么要有这个脚本
------------------
报告里的数字（Hit@5 / MRR / 幻觉率）只有在一件事成立时才有意义：
**它对应的是当前这份语料、当前这个模型**。
靠"我记得跑的时候没改语料"来支撑是可笑的——记忆不可复现，机器校验可以。

本脚本做三件事：
  1. 语料：索引里的每个 doc 是否仍在磁盘上、内容是否逐字一致；
  2. 模型：manifest 是否写明了模型名（unknown 等于没写）；
  3. 维度：embedding_dim 是否实测且等于 1024（BGE-M3 的特征值）。

用法
----
    python tools/verify_evidence_chain.py                    # 默认 reports/latest
    python tools/verify_evidence_chain.py reports/run-xxx
    python tools/verify_evidence_chain.py --json run-xxx     # 机器可读

退出码：0 = 证据链完整；1 = 有缺口（缺什么打印什么）；2 = 报告不存在/无法解析。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM = lambda s: re.sub(r"\s+", "", s)  # noqa: E731

# BGE-M3 的输出维度。写成常量不是为了"断言它一定是 1024"，
# 而是为了在它被悄悄换掉（比如换成了 768 维的老模型）时能立刻报警。
EXPECTED_DIM = 1024


# --------------------------------------------------------------------- 语料
def check_corpus(chunks_path: str, raw_dir: str) -> Tuple[int, int, List[str]]:
    """索引语料 vs 磁盘语料。返回 (索引篇数, 当前篇数, 结论行)。"""
    if not os.path.exists(chunks_path):
        # ★ 索引用 .gitignore 排除（它可由 data/raw + 一条 ingest 命令重建），
        #   所以"仓库里没有索引"是设计决定，不是证据缺口 —— 记 △ 并说明怎么补齐，
        #   否则分仓（无索引产物）会永远告警，告警就变成噪音了。
        return 0, 0, [f"△ 索引文件不存在：{chunks_path}"
                      f"（索引不随仓库分发；先跑 ingest 再校验，可补齐这一项）"]

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f).get("chunks", [])

    by: Dict[str, List[dict]] = {}
    for c in chunks:
        by.setdefault(c.get("doc_id", ""), []).append(c)
    cur = {f for f in os.listdir(raw_dir) if f.endswith(".md")} \
        if os.path.isdir(raw_dir) else set()

    notes = [f"索引 {len(by)} 篇 / 当前 {len(cur)} 篇 / {len(chunks)} 块"]
    drift = sorted(set(by) - cur)
    added = sorted(cur - set(by))
    if drift:
        notes.append(f"✗ 语料已被删除（报告数字基于旧语料）: {drift}")
    if added:
        notes.append(f"△ 索引未包含（报告数字不含这些文档）: {added}")

    bad = []
    for doc, items in by.items():
        p = os.path.join(raw_dir, doc)
        if not os.path.exists(p) or not os.path.isfile(p):
            bad.append(doc)
            continue
        with open(p, encoding="utf-8") as f:
            text = NORM(f.read())
        # ★ 必须去空白再比：chunk 会重排换行，直接子串匹配会全是假阳性。
        #   假阳性比假阴性危险——它会让你把真数字误判成假的。
        if any(NORM(x.get("text", "")) not in text for x in items):
            bad.append(doc)
    notes.append(("✗ 内容已变更: %s" % bad) if bad else "✓ 内容逐字一致")
    return len(by), len(cur), notes


# --------------------------------------------------------------------- 模型
def check_model(manifest: Dict[str, Any]) -> List[str]:
    """模型归属：名字必须写明，维度必须实测。"""
    out: List[str] = []
    info = manifest.get("model_info") or {}
    ing = manifest.get("ingest") or {}
    emb = info.get("embedding_model") or ing.get("embedding_model")
    dim = info.get("embedding_dim")
    rr = info.get("reranker_model")

    if emb in (None, "", "unknown"):
        out.append("✗ embedding_model 缺失/为 unknown → 模型归属无法自证")
    else:
        out.append(f"✓ embedding_model = {emb}")

    if dim is None:
        out.append("△ embedding_dim 未记录（重跑后由脚本实测写入）")
    elif int(dim) != EXPECTED_DIM:
        out.append(f"✗ embedding_dim = {dim}，期望 {EXPECTED_DIM}"
                   f"（换模型了？报告口径需重算）")
    else:
        out.append(f"✓ embedding_dim = {dim}")

    if rr in (None, "", "unknown"):
        out.append("△ reranker_model 未记录（无法自证精排是否加载）")
    else:
        out.append(f"✓ reranker_model = {rr}")
        if info.get("reranker_loaded") is False:
            out.append("✗ reranker_loaded=False → 本次 run 没有精排")
    return out


# --------------------------------------------------------------------- 主流程
def verify(run_dir: str, chunks_path: str, raw_dir: str) -> Tuple[int, Dict[str, Any]]:
    """返回 (退出码 0/1/2, 结构化结果)。"""
    manifest_path = os.path.join(run_dir, "eval_manifest.json")
    if not os.path.exists(manifest_path):
        return 2, {"error": f"manifest 不存在：{manifest_path}"}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        return 2, {"error": f"manifest 解析失败：{exc}"}

    n_idx, n_cur, corpus_notes = check_corpus(chunks_path, raw_dir)
    model_notes = check_model(manifest)
    ok = not any(n.startswith("✗") for n in corpus_notes + model_notes)

    return (0 if ok else 1), {
        "run_id": manifest.get("run_id"),
        "timestamp": manifest.get("timestamp"),
        "corpus": corpus_notes,
        "model": model_notes,
        "indexed_docs": n_idx,
        "current_docs": n_cur,
        "ok": ok,
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="评测报告数字证据链校验")
    p.add_argument("run_dir", nargs="?", default="reports/latest",
                   help="run 目录（含 eval_manifest.json）")
    p.add_argument("--chunks", default=os.path.join(ROOT, "data/vectors/chunks.json"))
    p.add_argument("--raw", default=os.path.join(ROOT, "data/raw"))
    p.add_argument("--json", action="store_true", help="输出 JSON（给 CI 用）")
    args = p.parse_args(argv)

    code, res = verify(args.run_dir, args.chunks, args.raw)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return code
    if "error" in res:
        print(f"[error] {res['error']}", file=sys.stderr)
        return code
    print(f"[run] {res['run_id']} @ {res['timestamp']}")
    print("[corpus]")
    for n in res["corpus"]:
        print("  " + n)
    print("[model]")
    for n in res["model"]:
        print("  " + n)
    print(f"[result] 证据链{'完整' if res['ok'] else '有缺口'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
