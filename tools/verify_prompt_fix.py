# -*- coding: utf-8 -*-
"""tools/verify_prompt_fix.py —— 针对"检索命中却仍拒答"的条目做定向复测。

为什么要有这个脚本（而不是直接重跑全量评测）：
  全量 119 条要跑十几分钟。而问题已定位到**特定 21 条**，
  先复测这 21 条能在 2 分钟内回答"提示词改动到底有没有用"。

  这是定位类修复的标准做法：先用最小样本验证方向，再决定要不要全量重跑。
  如果这 21 条仍然全拒答，说明改 prompt 无效，得换思路（改召回或改模型），
  那就没必要花十几分钟跑全量。

用法：
    python tools/verify_prompt_fix.py data/eval/eval.jsonl
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag1_knowledge_qa.bootstrap import build_production_service


def main() -> int:
    eval_set = sys.argv[1] if len(sys.argv) > 1 else "data/eval/eval.jsonl"
    items = [json.loads(line) for line in open(eval_set, encoding="utf-8") if line.strip()]
    service = build_production_service()
    if service.llm is None:
        print("✗ LLM 未装配", file=sys.stderr)
        return 1

    print("第一遍：找出「检索已命中 gold 但被拒答」的条目 ...")
    targets = []
    for it in items:
        if not it["is_answerable"]:
            continue
        r = service.answer(it["query"])
        gold = set(it.get("gold_ids") or [])
        got = {h["chunk_id"] for h in r.hits}
        if r.refused and (gold & got):
            targets.append(it)

    print(f"  命中该条件的条目：{len(targets)} 条")
    if not targets:
        print("  没有需要复测的条目，说明提示词改动后已无此问题。")
        return 0

    print()
    print("第二遍：重新回答这些条目，看是否仍拒答 ...")
    print()
    print(f"  {'qid':<6} {'问题':<44} {'top分':>7}  结果")
    print("  " + "-" * 76)
    still = []
    fixed = 0
    for it in targets:
        r = service.answer(it["query"])
        score = r.hits[0]["score"] if r.hits else float("nan")
        status = "仍拒答 ✗" if r.refused else "已作答 ✓"
        if r.refused:
            still.append(it["qid"])
        else:
            fixed += 1
        print(f"  {it['qid']:<6} {it['query'][:42]:<44} {score:>7.2f}  {status}")

    print()
    print("=" * 78)
    n = len(targets)
    print(f"定向复测结果：{fixed}/{n} 已从「拒答」变为「作答」"
          f"（{fixed / n:.1%}）")
    if still:
        print(f"仍拒答 {len(still)} 条：{', '.join(still[:15])}")
        print()
        print("若仍有较多条拒答，说明单纯改 prompt 不够，需要检查：")
        print("  · 召回是否命中了**正确**的块（命中 gold 的哪一段？）")
        print("  · 该块是否真的包含答案（可能是 eval 标注的 gold 不准）")
    print("=" * 78)

    os.makedirs("reports", exist_ok=True)
    with open("reports/prompt_fix_check.json", "w", encoding="utf-8") as f:
        json.dump({"targets": len(targets), "fixed": fixed,
                   "still_refused": still}, f, ensure_ascii=False, indent=2)
    print("\n已写入 reports/prompt_fix_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
