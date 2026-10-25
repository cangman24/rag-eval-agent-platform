# -*- coding: utf-8 -*-
"""tools/diagnose_refusal.py —— 诊断拒答判定的混淆矩阵。

为什么需要它：
  拒答正确率 0.60 是个**混合指标**，它同时包含两个方向相反的错误：

      过度拒答（可答的拒了）  -> 用户问什么都答不上来，体验灾难
      漏拒答  （不可答的答了） -> 编造答案，对知识库产品是更严重的问题

  两者性质完全不同，修法也不同，所以必须拆开看。

★ 顺带会暴露一个潜在缺陷：
  评测里判"是否拒答"用的是 `"未找到" in answer`（宽匹配）。
  这会把"回答里引用了这句话"也误判成拒答，例如：

      模型回答："知识库中未找到相关记录的规定是……"   <- 这是作答，不是拒答

  本脚本同时用两种判据统计，差值就是宽匹配的误判量。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag1_knowledge_qa.bootstrap import build_production_service


def main() -> int:
    eval_set = sys.argv[1] if len(sys.argv) > 1 else "data/eval/eval.jsonl"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    items = [json.loads(line) for line in open(eval_set, encoding="utf-8") if line.strip()]
    if limit:
        items = items[:limit]

    print(f"评测集 {len(items)} 条，开始逐条诊断（会真实调用 LLM）...")
    service = build_production_service()
    if service.llm is None:
        print("✗ LLM 未装配，无法诊断（检查 DEEPSEEK_API_KEY）", file=sys.stderr)
        return 1

    # 混淆矩阵：真实可答性 × 系统行为
    m = {
        "ans_refused": 0,      # 可答 -> 拒答（过度拒答，体验灾难）
        "ans_answered": 0,     # 可答 -> 作答（正确）
        "unans_refused": 0,    # 不可答 -> 拒答（正确）
        "unans_answered": 0,   # 不可答 -> 作答（漏拒答，可能编造）
    }
    # 宽判据 vs 严格判据的差异
    loose_only = []        # 只有宽松判据认为拒答的（疑似误判）
    over_refused = []      # 过度拒答的明细（最有价值的排查清单）
    leak_answered = []     # 该拒没拒的明细

    for i, it in enumerate(items, 1):
        try:
            r = service.answer(it["query"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {it['qid']} 调用失败: {exc}")
            continue

        answer = r.answer or ""
        loose = "未找到" in answer          # 评测当前用的宽判据
        strict = bool(r.refused)            # service 的权威判定（含 should_refuse）
        if loose and not strict:
            loose_only.append((it["qid"], it["query"][:40], answer[:70]))

        if it["is_answerable"]:
            if strict:
                m["ans_refused"] += 1
                over_refused.append(
                    (it["qid"], it["query"][:46],
                     [h["chunk_id"] for h in r.hits][:2],
                     it["gold_ids"][:2]))
            else:
                m["ans_answered"] += 1
        else:
            if strict:
                m["unans_refused"] += 1
            else:
                m["unans_answered"] += 1
                leak_answered.append((it["qid"], it["query"][:46], answer[:70]))

        if i % 20 == 0:
            print(f"  已处理 {i}/{len(items)}")

    print()
    print("=" * 76)
    print("拒答混淆矩阵（按 service 的权威判定 refused 字段）")
    print("=" * 76)
    print(f"  可答   {m['ans_answered']:>4} 条作答 ✓    {m['ans_refused']:>4} 条拒答 ✗（过度拒答）")
    print(f"  不可答 {m['unans_refused']:>4} 条拒答 ✓    {m['unans_answered']:>4} 条作答 ✗（漏拒答）")
    total = sum(m.values())
    correct = m["ans_answered"] + m["unans_refused"]
    print()
    print(f"  拒答正确率 = {correct}/{total} = {correct / total:.4f}" if total else "  无数据")

    ans_total = m["ans_answered"] + m["ans_refused"]
    unans_total = m["unans_refused"] + m["unans_answered"]
    if ans_total:
        print(f"  可答覆盖率（不误拒）= {m['ans_answered']}/{ans_total} "
              f"= {m['ans_answered'] / ans_total:.4f}")
    if unans_total:
        print(f"  越界拒答率（该拒就拒）= {m['unans_refused']}/{unans_total} "
              f"= {m['unans_refused'] / unans_total:.4f}")

    if loose_only:
        print()
        print("=" * 76)
        print(f"★ 判据缺陷：{len(loose_only)} 条被宽判据（\"未找到\" in answer）误判为拒答")
        print("  这些是**作答**，但回答里恰好包含了'未找到'三个字。")
        print("  评测口径应从宽匹配改为 service 的 is_refusal()（限前 60 字 + 完整话术）")
        print("=" * 76)
        for qid, q, a in loose_only[:8]:
            print(f"  {qid} {q}")
            print(f"       回答: {a}")

    if over_refused:
        print()
        print("=" * 76)
        print(f"过度拒答明细（{len(over_refused)} 条）—— 这是最该优先修的")
        print("=" * 76)
        print(f"  {'qid':<6} {'问题':<48} {'检索到的':<26} 期望")
        for qid, q, got, want in over_refused[:20]:
            print(f"  {qid:<6} {q:<48} {str(got):<26} {want}")

    if leak_answered:
        print()
        print("=" * 76)
        print(f"漏拒答明细（{len(leak_answered)} 条）—— 越界问题被作答（有编造风险）")
        print("=" * 76)
        for qid, q, a in leak_answered[:12]:
            print(f"  {qid} {q}")
            print(f"       回答: {a}")

    out = {
        "confusion": m,
        "refusal_accuracy": (correct / total) if total else 0.0,
        "answerable_cover_rate": (m["ans_answered"] / ans_total) if ans_total else 0.0,
        "out_of_scope_refusal_rate": (m["unans_refused"] / unans_total) if unans_total else 0.0,
        "loose_judgement_false_positives": len(loose_only),
        "over_refused_qids": [x[0] for x in over_refused],
        "leaked_qids": [x[0] for x in leak_answered],
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/refusal_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print("已写入 reports/refusal_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
