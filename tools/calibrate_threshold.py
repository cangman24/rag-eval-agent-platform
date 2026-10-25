# -*- coding: utf-8 -*-
"""tools/calibrate_threshold.py —— 标定拒答阈值。

为什么必须标定而不是拍一个数：
  RAG_REFUSAL_THRESHOLD 的判据是「精排后最高分 < 阈值 就拒答」。
  而 bge-reranker 的输出是**未归一化的 logit**：正例约 1~6，反例为负，
  不是 [0,1] 概率。所以：

    · 阈值设 0.3  -> 大量本该作答的问题落到负分区 -> 过度拒答
    · 阈值设 -8   -> 越界问题也会被硬答 -> 漏拒答（编造）

  ★ 实测证据（本脚本就是为拿到它而写的）：
      阈值 0.3 时可答覆盖率只有 51.5%，而越界拒答率 100%。
      即"太保守"——把一半能答的问题拒了。

  正确做法是**用评测集扫一遍阈值**，看两个方向的取舍曲线，
  然后按业务偏好选点：知识库产品通常宁可多答（用户能自己判断），
  但越界拒答不能掉太多。

用法：
    python tools/calibrate_threshold.py data/eval/eval.jsonl
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag1_knowledge_qa.bootstrap import build_production_service


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    i = min(int(len(sorted_vals) * p / 100.0), len(sorted_vals) - 1)
    return sorted_vals[i]


def main() -> int:
    eval_set = sys.argv[1] if len(sys.argv) > 1 else "data/eval/eval.jsonl"
    items = [json.loads(line) for line in open(eval_set, encoding="utf-8") if line.strip()]
    service = build_production_service()

    ans, unans = [], []
    for it in items:
        try:
            raw = service.retrieve(it["query"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {it['qid']} 检索失败: {exc}")
            continue
        top = max(raw.scores) if raw.scores else float("-inf")
        (ans if it["is_answerable"] else unans).append((it["qid"], top))

    a_scores = sorted(s for _, s in ans)
    u_scores = sorted(s for _, s in unans)

    print("=" * 78)
    print("精排最高分分布（这是标定阈值的唯一依据）")
    print("=" * 78)
    print(f"  可答样本 {len(a_scores)} 条")
    for p in (1, 5, 10, 25, 50, 75, 90, 95):
        print(f"    P{p:<3} = {percentile(a_scores, p):>9.4f}")
    print(f"    min = {a_scores[0]:>9.4f}    max = {a_scores[-1]:>9.4f}")
    print()
    print(f"  越界样本 {len(u_scores)} 条")
    for p in (50, 75, 90, 95, 99):
        print(f"    P{p:<3} = {percentile(u_scores, p):>9.4f}")
    print(f"    min = {u_scores[0]:>9.4f}    max = {u_scores[-1]:>9.4f}")

    print()
    print("=" * 78)
    print("阈值扫描：不同阈值下的双向取舍")
    print("=" * 78)
    print(f"  {'阈值':>8} {'可答覆盖率':>12} {'越界拒答率':>12} {'综合正确率':>12}  说明")
    print("  " + "-" * 70)

    best = None
    for thr in (-10, -8, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.3, 1, 2, 3):
        # 判据：最高分 < 阈值 -> 拒答
        cover = sum(1 for s in a_scores if s >= thr) / len(a_scores) if a_scores else 0
        refuse = sum(1 for s in u_scores if s < thr) / len(u_scores) if u_scores else 0
        total = sum(1 for s in a_scores if s >= thr) + sum(1 for s in u_scores if s < thr)
        n = len(a_scores) + len(u_scores)
        acc = total / n if n else 0
        note = ""
        if abs(thr - 0.3) < 1e-9:
            note = "<- 当前默认值（明显偏保守）"
        if best is None or acc > best[1]:
            best = (thr, acc, cover, refuse)
        print(f"  {thr:>8.1f} {cover:>12.4f} {refuse:>12.4f} {acc:>12.4f}  {note}")

    print()
    thr, acc, cover, refuse = best
    print(f"★ 综合正确率最高的阈值：{thr}（正确率 {acc:.4f}，"
          f"可答覆盖 {cover:.4f}，越界拒答 {refuse:.4f}）")
    print()
    print("选点建议（不是纯看正确率，要看业务偏好）：")
    print("  · 知识库问答 >> 宁可多答：用户看到来源能自己判断，被拒答则完全没信息")
    print("     -> 选「可答覆盖率」高、越界拒答率仍可接受的点")
    print("  · 合规/医疗等高风险场景 >> 宁可不答：编造答案的代价远大于拒答")
    print("     -> 选越界拒答率接近 1.0 的点，接受可答覆盖率下降")
    print()
    print("改法：在 .env 里设置 RAG_REFUSAL_THRESHOLD=<选定值>，然后重跑评测。")

    os.makedirs("reports", exist_ok=True)
    out = {
        "answerable_scores": a_scores,
        "out_of_scope_scores": u_scores,
        "best_threshold": thr,
        "best_accuracy": acc,
        "answerable_cover_at_best": cover,
        "oos_refusal_at_best": refuse,
    }
    with open("reports/threshold_calibration.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print("已写入 reports/threshold_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
