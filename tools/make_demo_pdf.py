# -*- coding: utf-8 -*-
"""tools/make_demo_pdf.py —— 生成 doc_vision 的演示/测试 PDF（纯标准库）。

跑法：
    python tools/make_demo_pdf.py --out data/demo_doc.pdf
    python tools/make_demo_pdf.py --out data/demo_doc.pdf --gt data/demo_doc_gt.json

★ 为什么自己写 PDF 生成器：
    doc_vision 需要一份"版式已知"的文档来验证提取结果——标题在哪、
    表格几行几列、柱状图每根多高，全都已知，才能断言"提对了"。
    现成库（reportlab）不在零依赖 CI 环境里；PDF 的文本/线条/填充矩形
    都是几十字节的内容流操作，自己写生成器反而让**测试 fixture 完全可控**。

★ 为什么演示文档用英文：标准 14 字体（Helvetica 系）不含 CJK 字形，
    嵌入中文字体需要 CIDFont + cmap 解析（另一个工程）。
    版式分析算法本身与语言无关（输入是 span 列表），中文语料走既有
    markdown 管线，互不影响。

产出的 PDF（单页 A4）：
    标题（18pt bold）
    正文段落（10pt 三行）
    H2 + 4 列 × 5 行表格（实线网格 + 加粗表头）
    H2 + 矢量柱状图（5 根彩柱 + 坐标轴 + 刻度标签 0~100）
    图注（9pt 斜体区域用普通字体代替）
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

PAGE_W, PAGE_H = 595.0, 842.0     # A4

# ---------------------------------------------------------------- 演示文档内容
TITLE = "Quarterly Infrastructure Operations Review"
PARA = [
    "This report summarizes ticket throughput, resolution outcomes, and",
    "service-level compliance for the observed period. All figures are",
    "derived from the ticketing system export and reviewed by the on-call team.",
]
TABLE_HEADER = ["Category", "Open", "Closed", "Total"]
TABLE_ROWS = [
    ["Network", "12", "34", "46"],
    ["Storage", "8", "27", "35"],
    ["Compute", "15", "41", "56"],
    ["Access", "5", "19", "24"],
]
CHART_TITLE = "Weekly Ticket Volume"
CHART_BARS = [("Mon", 40.0), ("Tue", 65.0), ("Wed", 55.0),
              ("Thu", 80.0), ("Fri", 70.0)]
CHART_TICKS = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
CHART_CAPTION = "Figure 1: Weekly ticket volume (tickets per day)"


def _esc(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class PdfBuilder:
    """攒对象、算 xref、写文件。只支持本生成器用到的算子子集。"""

    def __init__(self) -> None:
        self.objects: List[bytes] = [b""]     # 1-based

    def add(self, body: bytes) -> int:
        self.objects.append(body)
        return len(self.objects) - 1

    def text(self, x: float, y: float, size: float, s: str, bold: bool = False) -> str:
        font = "/F2" if bold else "/F1"
        return (f"BT {font} {size} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm "
                f"({_esc(s)}) Tj ET\n")

    def line(self, x0: float, y0: float, x1: float, y1: float,
             width: float = 0.7) -> str:
        return (f"0 0 0 RG {width:.2f} w {x0:.1f} {y0:.1f} m "
                f"{x1:.1f} {y1:.1f} l S\n")

    def fill_rect(self, x: float, y: float, w: float, h: float,
                  rgb: Tuple[float, float, float]) -> str:
        return (f"{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} rg "
                f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re f\n")

    def build(self, path: str) -> None:
        content = self._content_stream()
        font1 = self.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                         b"/Encoding /WinAnsiEncoding >>")
        font2 = self.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                         b"/Encoding /WinAnsiEncoding >>")
        stream = self.add(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content + b"endstream")
        page = self.add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.0f} {PAGE_H:.0f}] "
            f"/Resources << /Font << /F1 {font1} 0 R /F2 {font2} 0 R >> >> "
            f"/Contents {stream} 0 R >>".encode())
        pages = self.add(
            f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>".encode())
        catalog = self.add(f"<< /Type /Catalog /Pages {pages} 0 R >>".encode())

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i in range(1, len(self.objects)):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + self.objects[i] + b"\nendobj\n"
        xref_at = len(out)
        n = len(self.objects)
        out += f"xref\n0 {n}\n".encode()
        out += b"0000000000 65535 f \n"
        for i in range(1, n):
            out += f"{offsets[i]:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {n} /Root {catalog} 0 R >>\n"
                f"startxref\n{xref_at}\n%%EOF\n").encode()
        with open(path, "wb") as f:
            f.write(bytes(out))

    # ------------------------------------------------------------ 页面内容
    def _content_stream(self) -> bytes:
        ops: List[str] = []
        gt: Dict[str, Any] = {"title": TITLE, "table": [TABLE_HEADER] + TABLE_ROWS,
                              "bars": []}

        # 标题
        ops.append(self.text(60, 780, 18, TITLE, bold=True))
        y = 750
        for line in PARA:
            ops.append(self.text(60, y, 10, line))
            y -= 14

        # ---- 表格 ----
        y -= 16
        ops.append(self.text(60, y, 14, "Ticket Status by Category", bold=True))
        y -= 18
        col_x = [60.0, 220.0, 320.0, 420.0, 520.0]
        row_h = 18.0
        table_top = y
        for r, row in enumerate([TABLE_HEADER] + TABLE_ROWS):
            ry = table_top - r * row_h
            for c, cell in enumerate(row):
                ops.append(self.text(col_x[c] + 4, ry - 13, 10, cell,
                                     bold=(r == 0)))
            ops.append(self.line(col_x[0], ry - row_h, col_x[-1], ry - row_h))
        for cx in col_x:
            ops.append(self.line(cx, table_top, cx, table_top - 5 * row_h))
        y = table_top - 5 * row_h - 30

        # ---- 矢量柱状图 ----
        ops.append(self.text(60, y, 14, CHART_TITLE, bold=True))
        y -= 20
        plot_left, plot_right = 110.0, 520.0
        plot_bottom, plot_top = y - 180.0, y
        # 坐标轴
        ops.append(self.line(plot_left, plot_bottom, plot_right, plot_bottom, 1.0))
        ops.append(self.line(plot_left, plot_bottom, plot_left, plot_top, 1.0))
        # 刻度（标签 + 网格线）
        tick_y: List[Tuple[float, float]] = []
        for tv in CHART_TICKS:
            ty = plot_bottom + (tv / 100.0) * (plot_top - plot_bottom)
            tick_y.append((ty, tv))
            ops.append(self.line(plot_left - 4, ty, plot_left, ty, 0.5))
            ops.append(self.text(plot_left - 38, ty - 3, 8, f"{tv:g}"))
        # 柱体
        bar_w = 44.0
        slot = (plot_right - plot_left) / len(CHART_BARS)
        bars_gt = []
        for i, (label, value) in enumerate(CHART_BARS):
            bx = plot_left + i * slot + (slot - bar_w) / 2
            bh = (value / 100.0) * (plot_top - plot_bottom)
            ops.append(self.fill_rect(bx, plot_bottom, bar_w, bh, (0.20, 0.40, 0.75)))
            ops.append(self.text(bx + 12, plot_bottom - 12, 8, label))
            bars_gt.append({"label": label, "value": value,
                            "rect": [bx, plot_bottom, bx + bar_w,
                                     plot_bottom + bh]})
        gt["bars"] = bars_gt
        gt["ticks"] = [{"y_pixel_pdf": ty, "value": v} for ty, v in tick_y]
        gt["plot_box_pdf"] = [plot_left, plot_bottom, plot_right, plot_top]

        # 图注
        ops.append(self.text(60, plot_bottom - 30, 9, CHART_CAPTION))

        self._gt = gt
        return "".join(ops).encode("latin-1", "replace")

    def ground_truth(self) -> Dict[str, Any]:
        return getattr(self, "_gt", {})


def build_demo_pdf(path: str, gt_path: str = "") -> Dict[str, Any]:
    """生成演示 PDF（+ 可选 ground truth JSON）。返回 ground truth。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    b = PdfBuilder()
    b.build(path)
    gt = b.ground_truth()
    if gt_path:
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=1)
    return gt


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="生成 doc_vision 演示 PDF")
    parser.add_argument("--out", default="data/demo_doc.pdf")
    parser.add_argument("--gt", default="",
                        help="ground truth JSON 输出路径（默认与 PDF 同目录）")
    args = parser.parse_args(argv)
    gt_path = args.gt or os.path.splitext(args.out)[0] + "_gt.json"
    gt = build_demo_pdf(args.out, gt_path)
    print(f"已生成 {args.out}（标题 {len(gt.get('title', ''))} 字符，"
          f"表格 {len(gt.get('table', []))} 行，柱状图 {len(gt.get('bars', []))} 根）")
    print(f"ground truth -> {gt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
