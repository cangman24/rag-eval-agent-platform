# -*- coding: utf-8 -*-
"""tools/diagnose_refusal2.py —— 修正口径后重测拒答混淆矩阵。

★ 修正了什么（这是本轮最重要的修复）：

  `run_answer_eval` 判"是否拒答"用的是：

      refusal_marker = "未找到"        # 宽匹配
      refused = "未找到" in answer

  而 service 的权威判据是 `core.llm.is_refusal()`：

      REFUSAL_PHRASE = "知识库中未找到"
      return REFUSAL_PHRASE in text[:60]     # 限前 60 字 + 完整话术

  两者差异会造成**两个方向的误判**：

    宽匹配多判：回答里引用了"未找到"三个字（如"…未找到相关记录的规定是…"）
                -> 明明是作答，被算成拒答 -> 拒答正确率虚低
    宽匹配少判：模型用别的话术拒答（如"无法回答""片段中未包含"）
                -> 明明是拒答，被算成作答 -> 拒答正确率虚高

  ★ 所以本脚本同时统计三种口径，把差异显式暴露出来，
    而不是默默用一个不准的数去填报告。

用法：
    python tools/diagnose_refusal2.py data/eval/eval.jsonl [limit]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag1_knowledge_qa.bootstrap import build_production_service


def is_refusal_strict(answer: str) -> bool:
    """与 service 一致的权威判据。"""
    from core.llm import is_refusal
    return is_refusal(answer)


def is_refusal_loose(answer: str) -> bool:
    """评测框架当前用的宽判据（保留以便对比）。"""
    return "未找到" in (answer or "")


def main() -> int:
    eval_set = sys.argv[1] if len(sys.argv) > 1 else "data/eval/eval.jsonl"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    items = [json.loads(l) for l in open(eval_set, encoding="utf-8") if l.strip()]
    if limit:
        items = items[:limit]

    service = build_production_service()
    if service.llm is None:
        print("✗ LLM 未装配", file=sys.stderr)
        return 1

    print(f"逐条诊断 {len(items)} 条（每条约 3~5 秒）...")

    # 三种口径各自的混淆矩阵
    mats = {k: {"a_ok": 0, "a_bad": 0, "u_ok": 0, "u_bad": 0}
            for k in ("loose", "strict", "service")}
    disagree = []          # 宽判据与严格判据不一致的明细
    over_refused = []      # 检索命中却仍被拒答（最该修的）

    for i, it in enumerate(items, 1):
        try:
            r = service.answer(it["query"])
        except Exception as exc:  # noqa: BLE001
            print(f"  {it['qid']} 失败: {exc}")
            continue
        ans = r.answer or ""
        flags = {
            "loose": is_refusal_loose(ans),
            "strict": is_refusal_strict(ans),
            "service": bool(r.refused),
        }
        for name, refused in flags.items():
            m = mats[name]
            if it["is_answerable"]:
                m["a_bad" if refused else "a_ok"] += 1
            else:
                m["u_ok" if refused else "u_bad"] += 1

        if flags["loose"] != flags["strict"]:
            disagree.append((it["qid"], it["query"][:38], ans[:60],
                             "宽松=拒答/严格=作答" if flags["loose"]
                             else "宽松=作答/严格=拒答"))
        # 检索命中 gold 却被拒答 -> 说明不是检索的锅
        gold = set(it.get("gold_ids") or [])
        got = {h["chunk_id"] for h in r.hits}
        if it["is_answerable"] and flags["service"] and (gold & got):
            over_refused.append((it["qid"], it["query"][:44],
                                 sorted(gold & got)[0], round(r.hits[0]["score"], 2)))

    print()
    print("=" * 78)
    print("三种判据下的混淆矩阵对比")
    print("=" * 78)
    label = {"loose": '宽松 "未找到" in answer（评测框架现状）',
             "strict": "严格 is_refusal()（与 service 一致）",
             "service": "service.refused 字段（含 should_refuse 阈值）"}
    for name in ("loose", "strict", "service"):
        m = mats[name]
        n = sum(m.values())
        correct = m["a_ok"] + m["u_ok"]
        print(f"\n【{label[name]}】")
        print(f"  可答   {m['a_ok']:>4} 作答✓   {m['a_bad']:>4} 拒答✗")
        print(f"  不可答 {m['u_ok']:>4} 拒答✓   {m['u_bad']:>4} 作答✗")
        if n:
            print(f"  拒答正确率 = {correct}/{n} = {correct / n:.4f}")
            a_tot = m["a_ok"] + m["a_bad"]
            u_tot = m["u_ok"] + m["u_bad"]
            if a_tot:
                print(f"    可答覆盖率（不误拒）= {m['a_ok']}/{a_tot} "
                      f"= {m['a_ok'] / a_tot:.4f}")
            if u_tot:
                print(f"    越界拒答率（该拒就拒）= {m['u_ok']}/{u_tot} "
                      f"= {m['u_ok'] / u_tot:.4f}")

    if disagree:
        print()
        print("=" * 78)
        print(f"★ 判据不一致 {len(disagree)} 条（这就是评测数字失真的来源）")
        print("=" * 78)
        for qid, q, a, kind in disagree[:12]:
            print(f"  {qid} [{kind}] {q}")
            print(f"       {a}")

    if over_refused:
        print()
        print("=" * 78)
        print(f"★ 检索命中 gold 却仍被拒答：{len(over_refused)} 条")
        print("  这些**不是检索问题**，是生成层过度保守（prompt / 话术标定）")
        print("=" * 78)
        for qid, q, g, sc in over_refused[:20]:
            print(f"  {qid} {q:<46} gold={g} score={sc}")

    out = {
        "matrices": mats,
        "refusal_accuracy": {
            k: (sum(m.values()) and (m["a_ok"] + m["u_ok"]) / sum(m.values()))
            for k, m in mats.items()
        },
        "disagreement_count": len(disagree),
        "over_refused_despite_hit": len(over_refused),
        "over_refused_qids": [x[0] for x in over_refused],
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/refusal_diagnosis2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print("已写入 reports/refusal_diagnosis2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
