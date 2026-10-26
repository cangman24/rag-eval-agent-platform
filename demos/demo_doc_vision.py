# -*- coding: utf-8 -*-
"""demo_doc_vision.py —— 多模态文档理解端到端演示（E3）。

跑法：python demos/demo_doc_vision.py
输出：demo_doc_vision_output.txt

★ 演示什么：

    一份 PDF 里有三种"模型读不到、但人一眼就懂"的信息：
      ① 版式结构：哪块是标题、哪块是正文、阅读顺序是什么
      ② 表格：按行列对齐重建成网格，可以"第 2 行第 3 列"地寻址
      ③ 图表：柱子的**高度**里藏着数据，文字层里根本没有

    这个 demo 走完整链路：
        stdlib 生成 PDF（带 ground truth）
          -> pypdf 提 span / 填充矩形 + 坐标翻转
          -> 聚类与判定（标题/段落/表格/图表）
          -> 注册成 MCP 工具
          -> 脚本化 Agent 调工具回答"哪天工单量最高"，并给出引用

    ★ 关键卖点：答案里的数字**不是模型编的**，是从 PDF 的矢量算子
      量出来的，并且与生成时写死的 ground truth 逐项对照 —— 有据可查。

★ 缺 pypdf 时（CI 零依赖环境）会友好降级并 exit 0：
    算法层（聚类/判定/网格重建）在零依赖下照样跑，只是不解析真文件。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT = []
CHECKS = []


def log(line=""):
    OUT.append(line)
    print(line)


def check(name, ok, detail=""):
    CHECKS.append((name, ok))
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))


def main():
    log("=" * 78)
    log("多模态文档理解演示（PDF -> 版面/表格/图表 -> Agent 可引用事实）")
    log("=" * 78)

    try:
        import pypdf  # noqa: F401
        has_pypdf = True
    except ImportError:
        has_pypdf = False

    # ---------------------------------------------------------------- 算法层
    log("\n[0] 算法层自检（零依赖，不碰文件）")
    from core.doc_vision import Span, analyze_spans

    spans = [Span("Quarterly Review", 60, 100, 300, 118, 18.0, True),
             Span("body line one", 60, 140, 300, 150, 10.0),
             Span("body line two", 60, 154, 300, 164, 10.0)]
    layout = analyze_spans(spans)
    kinds = [e.kind for e in layout.elements]
    log(f"  元素序列: {kinds}")
    check("标题按字号识别（正文为众数基准）", kinds.count("heading") == 1, str(kinds))

    if not has_pypdf:
        log("\n[!] 未安装 pypdf —— 真文件解析路径跳过（算法层已验证）。")
        log("    装上就能看完整链路：pip install pypdf")
        check("零依赖环境友好降级", True)
        _write_output()
        _report()
        return 0

    # ---------------------------------------------------------------- 真文件
    from tools.make_demo_pdf import build_demo_pdf

    tmp = os.path.join(ROOT, "demos", "data")
    os.makedirs(tmp, exist_ok=True)
    pdf_path = os.path.join(tmp, "doc_vision_demo.pdf")
    gt_path = os.path.join(tmp, "doc_vision_demo_gt.json")

    log("\n[1] 生成带 ground truth 的迷你 PDF（纯标准库，无第三方依赖）")
    gt = build_demo_pdf(pdf_path, gt_path)
    log(f"  文件: {os.path.relpath(pdf_path, ROOT)}")
    log(f"  gt: 标题「{gt['title']}」/ 表格 {len(gt['table'])} 行 "
        f"/ 柱状图 {len(gt['bars'])} 根")

    from core.doc_vision_tools import build_doc_vision_registry

    reg = build_doc_vision_registry()
    log(f"\n[2] 注册为 MCP 工具: {reg.names()}")

    log("\n[3] doc_analyze —— 版面结构（element_id 供引用）")
    doc = reg.call("doc_analyze", {"path": pdf_path, "max_pages": 1})
    page = doc["pages"][0]
    for el in page["elements"]:
        text = el["text"].replace("\n", " ")[:52]
        log(f"  {el['id']:>6} {el['kind']:<9} {text}")
    check("版面未降级", not page["degraded"], page.get("reason", ""))
    check("识别到标题", any(e["kind"] == "heading" for e in page["elements"]))
    check("识别到表格", any(e["kind"] == "table" for e in page["elements"]))
    check("识别到图表", any(e["kind"] == "figure" and "chart" in e
                            for e in page["elements"]))

    log("\n[4] doc_read_table —— 表格重建（跨行对齐判据）")
    table = reg.call("doc_read_table", {"path": pdf_path, "page": 1, "index": 0})
    log(f"  {table['id']}: {table['rows']} 行 x {table['columns']} 列")
    for row in table["grid"]:
        log("    " + " | ".join(f"{c:<9}" for c in row))
    check("网格与 ground truth 完全一致", table["grid"] == gt["table"])
    answer_cells = [r for r in table["grid"][1:] if r[0] == "Compute"]
    check("可按行列寻址（Compute 行）",
          answer_cells == [["Compute", "15", "41", "56"]], str(answer_cells))

    log("\n[5] doc_read_chart —— 图表读数（矢量精确，非像素估计）")
    chart = reg.call("doc_read_chart", {"path": pdf_path, "page": 1})
    for b in chart["bars"]:
        log(f"    {str(b['label']):<5} {b['value']:>7.2f}  "
            f"(confidence={b['confidence']})")
    check("图表读数未降级", not chart["degraded"], chart.get("reason", ""))
    max_err = max(abs(b["value"] - want["value"])
                  for b, want in zip(chart["bars"], gt["bars"], strict=False))
    check("读数与 gt 最大偏差 %.2f < 1.5" % max_err, max_err < 1.5)
    check("柱体带类目标签",
          [b["label"] for b in chart["bars"]] == [b["label"] for b in gt["bars"]])

    # ---------------------------------------------------------------- Agent
    log("\n[6] Agent 闭环 —— 用工具里的数据回答，并给出引用")
    from core.agent import AgentEngine

    calls = {"n": 0}

    def act(state, allowed):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool": "doc_read_chart", "args": {"path": pdf_path, "page": 1}}
        bars = state.scratchpad[-1].get("result", {}).get("bars", [])
        top = max(bars, key=lambda b: b["value"])
        return {"final": f"{top['label']} 的工单量最高，为 {top['value']:.1f}",
                "citations": [f"p1:chart:{top['index']}"]}

    engine = AgentEngine(
        registry=reg,
        intent_fn=lambda q: "chart_read",
        plan_fn=lambda q, t: ["解析版式", "读图表", "作答"],
        act_fn=act,
        route_table={"chart_read": ["doc_read_chart"]},
    )
    run = engine.run("哪天的工单量最高？")
    log("  问题: 哪天的工单量最高？")
    log(f"  答案: {run.state.answer}")
    log(f"  引用: {run.state.citations}")
    log(f"  轮次: {run.state.rounds} / 工具调用: {run.state.tool_calls}")

    want_label = max(gt["bars"], key=lambda b: b["value"])["label"]
    check("答案来自工具数据（非编造）", want_label in run.state.answer, run.state.answer)
    check("带引用", bool(run.state.citations))
    check("工具调用次数 = 1", run.state.tool_calls == 1)

    log("\n[7] 越权防护 —— 白名单外的工具调不动")
    try:
        reg.call("doc_read_table", {"path": pdf_path}, allowed=["doc_analyze"])
        check("白名单拦截", False, "未被拦截")
    except Exception as exc:  # noqa: BLE001
        check("白名单拦截", True, type(exc).__name__)

    _write_output()
    return _report()


def _write_output():
    path = os.path.join(ROOT, "demo_doc_vision_output.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
    try:
        os.remove(os.path.join(ROOT, "demos", "data", "doc_vision_demo.pdf"))
    except OSError:
        pass


def _report():
    failed = [n for n, ok in CHECKS if not ok]
    log("\n" + "=" * 78)
    log(f"自检 {len(CHECKS) - len(failed)}/{len(CHECKS)} 通过" +
        ("" if not failed else f"，失败: {failed}"))
    log("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
