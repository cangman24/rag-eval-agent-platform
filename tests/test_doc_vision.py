# -*- coding: utf-8 -*-
"""tests/test_doc_vision.py —— doc_vision 版式分析算法单测（零依赖可跑）。

覆盖三类风险：
    ① 结构判定：标题/段落/表格/图像区的识别与阅读顺序
    ② 表格重建：乱序 span -> 正确网格（列聚类 + 行聚类）
    ③ 降级：空页 / 无标题特征 / 图表无柱 —— 显式降级不猜

图表读数（read_bar_chart）需要 numpy：CI 零依赖环境跳过，真机跑。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.doc_vision import (  # noqa: E402
    Span, analyze_spans, read_bar_chart, rebuild_table,
)

try:
    import numpy as np  # noqa: F401 - 仅探测可用性
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def S(text, x0, y0, x1, y1, size=10.0, bold=False):
    return Span(text=text, x0=x0, y0=y0, x1=x1, y1=y1, size=size, bold=bold)


class TestAnalyzeSpans(unittest.TestCase):
    def test_empty_page_degrades(self):
        layout = analyze_spans([])
        self.assertTrue(layout.degraded)
        self.assertEqual(layout.reason, "页面无文字")

    def test_heading_detected_by_font_size(self):
        spans = [
            S("Quarterly Review", 60, 100, 300, 118, size=18, bold=True),
            S("First body line of the report.", 60, 140, 350, 150),
            S("Second body line continues here.", 60, 154, 350, 164),
        ]
        layout = analyze_spans(spans)
        headings = layout.by_kind("heading")
        self.assertEqual(len(headings), 1)
        self.assertEqual(headings[0].text, "Quarterly Review")
        self.assertEqual(headings[0].level, 1)
        paras = layout.by_kind("paragraph")
        self.assertEqual(len(paras), 1)
        self.assertIn("Second body", paras[0].text)

    def test_no_false_heading_when_all_same_size(self):
        """★ 没有字号差异时不许瞎猜标题——宁可不分级，不可乱分级。"""
        spans = [S(f"line {i}", 60, 100 + i * 14, 300, 110 + i * 14, size=10)
                 for i in range(5)]
        layout = analyze_spans(spans)
        self.assertEqual(layout.by_kind("heading"), [])
        self.assertEqual(len(layout.by_kind("paragraph")), 1)

    def test_table_detected_and_grid_rebuilt(self):
        spans = []
        cols = [60.0, 160.0, 260.0, 360.0]
        header = ["Category", "Open", "Closed", "Total"]
        for c, h in enumerate(header):
            spans.append(S(h, cols[c], 200, cols[c] + 80, 210, size=10, bold=True))
        rows = [["Network", "12", "34", "46"],
                ["Storage", "8", "27", "35"],
                ["Compute", "15", "41", "56"]]
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                spans.append(S(cell, cols[c], 220 + r * 16, cols[c] + 60,
                               230 + r * 16, size=10))
        layout = analyze_spans(spans)
        tables = layout.by_kind("table")
        self.assertEqual(len(tables), 1)
        grid = tables[0].extra["grid"]
        self.assertEqual(grid[0], header)
        self.assertEqual(grid[1], ["Network", "12", "34", "46"])
        self.assertEqual(tables[0].extra["columns"], 4)
        self.assertIn("| Category | Open | Closed | Total |",
                      tables[0].extra["markdown"])

    def test_ragged_lines_not_a_table(self):
        """★ 跨行对齐判据的价值：段落每行也有多个 x 簇，但行间参差。

        表格与段落的分界不在"一行有几个簇"，而在**这些簇是否跨行一致**。
        """
        spans = []
        # 三行散文，每行 3 个词片段，但 x 位置各行都不同
        xs = [(60, 120, 200), (70, 150, 240), (80, 130, 210)]
        for r, row in enumerate(xs):
            for w, x0 in enumerate(row):
                spans.append(S(f"word{r}{w}", x0, 100 + r * 16, x0 + 40,
                               110 + r * 16))
        layout = analyze_spans(spans)
        self.assertEqual(layout.by_kind("table"), [])
        self.assertEqual(len(layout.by_kind("paragraph")), 1)

    def test_figure_boxes_become_figure_elements(self):
        spans = [S("Chart below", 60, 100, 200, 110)]
        layout = analyze_spans(spans, figure_boxes=[(60, 150, 400, 350)])
        figures = layout.by_kind("figure")
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0].bbox, (60, 150, 400, 350))
        self.assertIn("ocr", figures[0].extra["needs"])

    def test_reading_order_top_to_bottom(self):
        spans = [S("bottom", 60, 300, 200, 310),
                 S("top", 60, 100, 200, 110)]
        layout = analyze_spans(spans)
        texts = [e.text for e in layout.elements if e.kind == "paragraph"]
        self.assertEqual(texts, ["top", "bottom"])

    def test_two_column_hint_in_reason(self):
        """★ 双栏要**多行**证据：左右各两行、彼此 x 不重叠、y 共享横带。"""
        spans = [S("left column line one", 60, 100, 250, 110),
                 S("left column line two", 60, 116, 250, 126),
                 S("right column line one", 320, 100, 520, 110),
                 S("right column line two", 320, 116, 520, 126)]
        layout = analyze_spans(spans)
        self.assertIn("双栏", layout.reason)

    def test_single_row_labels_not_two_column(self):
        """★ 同一行的散落标签（图表 x 轴类目）不许判成双栏。"""
        spans = [S("Mon", 141, 494, 154, 502),
                 S("Tue", 223, 494, 241, 502),
                 S("Wed", 305, 494, 323, 502)]
        layout = analyze_spans(spans)
        self.assertNotIn("双栏", layout.reason)

    def test_to_dict_roundtrip_shape(self):
        # ★ 必须同时给正文：整页只有一种字号时不允许判标题
        #   （见 test_no_false_heading 与 _body_size 的踩坑记录）
        spans = [S("T", 60, 100, 200, 118, size=18),
                 S("body", 60, 140, 200, 150, size=10)]
        d = analyze_spans(spans).to_dict()
        self.assertIn("elements", d)
        self.assertEqual(d["elements"][0]["kind"], "heading")
        self.assertEqual(d["elements"][0]["level"], 1)
        self.assertEqual(d["elements"][1]["kind"], "paragraph")


class TestRebuildTable(unittest.TestCase):
    def test_unordered_spans_rebuild_grid(self):
        spans = [
            S("46", 360, 220, 400, 230),
            S("Network", 60, 220, 150, 230),
            S("12", 160, 220, 200, 230),
            S("34", 260, 220, 300, 230),
            S("Total", 360, 200, 440, 210),
            S("Open", 160, 200, 220, 210),
            S("Closed", 260, 200, 360, 210),
            S("Category", 60, 200, 150, 210),
        ]
        table = rebuild_table(spans)
        self.assertEqual(table["grid"][0], ["Category", "Open", "Closed", "Total"])
        self.assertEqual(table["grid"][1], ["Network", "12", "34", "46"])
        self.assertEqual(table["rows"], 2)

    def test_empty_column_dropped(self):
        spans = [S("a", 60, 200, 100, 210),
                 S("", 200, 200, 260, 210),      # 空 span 被过滤
                 S("b", 60, 220, 100, 230)]
        table = rebuild_table([s for s in spans if s.text])
        self.assertEqual(table["columns"], 1)

    def test_empty_input(self):
        self.assertEqual(rebuild_table([])["grid"], [])


@unittest.skipUnless(HAS_NUMPY, "numpy 不可用（CI 零依赖环境跳过，真机跑）")
class TestReadBarChart(unittest.TestCase):
    """合成像素阵列测读数逻辑——不需要 PIL，只需要 numpy。"""

    def _make_chart(self, values, plot_box=(20, 10, 90, 50)):
        import numpy as np
        img = np.full((60, 100, 3), 255, dtype=np.uint8)
        left, top, right, bottom = plot_box
        # 白色背景 + 灰色坐标轴
        img[bottom, left:right] = (0, 0, 0)
        img[top:bottom, left] = (0, 0, 0)
        # 柱体：蓝色，高度按值线性映射
        n = len(values)
        slot = (right - left) / n
        for i, v in enumerate(values):
            x0 = int(left + i * slot + 2)
            x1 = int(left + (i + 1) * slot - 2)
            h = int((v / 100.0) * (bottom - top))
            img[bottom - h:bottom, x0:x1] = (51, 102, 191)
        return img

    def test_reads_three_bars(self):
        img = self._make_chart([75.0, 37.5, 87.5])
        result = read_bar_chart(
            img, plot_box=(20, 10, 90, 50),
            y_ticks=[(10, 100.0), (50, 0.0)])
        self.assertFalse(result["degraded"], result.get("reason"))
        self.assertEqual(len(result["bars"]), 3)
        values = [b["value"] for b in result["bars"]]
        for got, want in zip(values, [75.0, 37.5, 87.5], strict=False):
            self.assertAlmostEqual(got, want, delta=1.5)
        for b in result["bars"]:
            self.assertGreater(b["confidence"], 0.7)

    def test_blank_image_degrades(self):
        import numpy as np
        img = np.full((60, 100, 3), 255, dtype=np.uint8)
        result = read_bar_chart(img, plot_box=(20, 10, 90, 50),
                                y_ticks=[(10, 100.0), (50, 0.0)])
        self.assertTrue(result["degraded"])
        self.assertIn("柱体", result["reason"])

    def test_insufficient_ticks_degrades(self):
        img = self._make_chart([50.0])
        result = read_bar_chart(img, plot_box=(20, 10, 90, 50),
                                y_ticks=[(10, 100.0)])
        self.assertTrue(result["degraded"])
        self.assertIn("刻度", result["reason"])


if __name__ == "__main__":
    unittest.main()
