# -*- coding: utf-8 -*-
"""tests/test_doc_vision_pdf.py —— doc_vision 的**真 PDF 链路**测试。

零依赖测试（tests/test_doc_vision.py）测的是算法：喂进去的是手写的 span。
但"算法对"不等于"链路通"——真 PDF 还要走：

    内容流 -> pypdf 提取 span/矩形 -> 坐标翻转 -> 聚类 -> 判定

每一步都可能出错，而且错法各不相同（最典型的就是 PDF 的 y 轴向上、
屏幕坐标 y 轴向下，混用会让阅读顺序整个颠倒）。

所以这里用 tools/make_demo_pdf.py 生成一份**带 ground truth 的迷你 PDF**
（纯标准库生成，CI 也能生成），再让 analyze_pdf 跑真链路，逐项对照 gt。

需要 pypdf（可选依赖）：CI 零依赖环境自动跳过，真机跑。
与 TestRealLangGraph 同一套纪律：跳过要显式、原因要写清楚。
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.doc_vision import analyze_pdf, spans_from_pdf  # noqa: E402
from tools.make_demo_pdf import build_demo_pdf  # noqa: E402

try:
    import pypdf  # noqa: F401
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


@unittest.skipUnless(HAS_PYPDF, "pypdf 不可用（CI 零依赖环境跳过，真机跑）")
class TestRealPdfChain(unittest.TestCase):
    """生成 PDF -> pypdf 解析 -> 版式分析，对照 ground truth。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="doc_vision_")
        cls.pdf = os.path.join(cls.tmp, "demo.pdf")
        cls.gt_path = os.path.join(cls.tmp, "demo_gt.json")
        cls.gt = build_demo_pdf(cls.pdf, cls.gt_path)
        cls.pages = analyze_pdf(cls.pdf)

    # ---------------------------------------------------------------- 提取层
    def test_spans_extracted_with_page_size(self):
        pages = spans_from_pdf(self.pdf)
        self.assertTrue(pages)
        self.assertGreater(pages[0]["page_size"][0], 0)
        self.assertTrue(pages[0]["spans"], "pypdf 没提取到任何文字")

    def test_coords_are_screen_oriented(self):
        """★ 坐标翻转回归：屏幕坐标 y 向下，标题必须在正文**上方**（y 更小）。

        PDF 原生 y 向上，若不翻转，标题的 y 会大于正文，阅读顺序整个颠倒。
        """
        spans = spans_from_pdf(self.pdf)[0]["spans"]
        title = next(s for s in spans if "Quarterly" in s.text)
        body = next(s for s in spans if s.text.strip() == "Open")
        self.assertLess(title.y0, body.y0,
                        "坐标没翻转：标题应在正文上方（y 更小）")

    def test_filled_rects_extracted(self):
        """柱体是 re+f 算子，必须能从内容流里取出来。"""
        rects = spans_from_pdf(self.pdf)[0]["rects"]
        self.assertGreaterEqual(len(rects), len(self.gt["bars"]),
                                "内容流里的填充矩形比 gt 的柱数还少")

    # ---------------------------------------------------------------- 结构层
    def test_title_detected_as_heading(self):
        headings = self.pages[0].by_kind("heading")
        self.assertTrue(headings, "标题没被识别出来")
        self.assertEqual(headings[0].text.strip(), self.gt["title"])
        self.assertEqual(headings[0].level, 1)

    def test_table_grid_matches_ground_truth(self):
        tables = self.pages[0].by_kind("table")
        self.assertEqual(len(tables), 1, "表格没被识别（或识别成多块）")
        grid = tables[0].extra["grid"]
        self.assertEqual(grid, self.gt["table"],
                         "重建网格与 gt 不一致（列聚类或行聚类有偏差）")

    def test_table_markdown_is_wellformed(self):
        md = self.pages[0].by_kind("table")[0].extra["markdown"]
        lines = md.strip().splitlines()
        self.assertEqual(len(lines), len(self.gt["table"]) + 1)
        self.assertTrue(lines[1].startswith("|"))
        self.assertIn("---", lines[1])

    # ---------------------------------------------------------------- 图表层
    def test_vector_chart_values_match_ground_truth(self):
        """矢量柱状图读数：**精确几何**，与 gt 的偏差应该在 1 以内。"""
        figures = [e for e in self.pages[0].elements
                   if e.kind == "figure" and "chart" in (e.extra or {})]
        self.assertTrue(figures, "图表没被挂成 figure 元素")
        bars = figures[0].extra["chart"]["bars"]
        self.assertEqual(len(bars), len(self.gt["bars"]))
        for got, want in zip(bars, self.gt["bars"], strict=False):
            self.assertAlmostEqual(got["value"], want["value"], delta=1.5,
                                   msg=f"第 {want['label']} 根柱读数偏差过大")

    def test_chart_reading_is_not_degraded(self):
        figures = [e for e in self.pages[0].elements
                   if e.kind == "figure" and "chart" in (e.extra or {})]
        self.assertFalse(figures[0].extra["chart"]["degraded"],
                         figures[0].extra["chart"].get("reason"))

    # ---------------------------------------------------------------- 端到端
    def test_ground_truth_file_is_valid_json(self):
        with open(self.gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        self.assertEqual(gt["title"], self.gt["title"])

    def test_pages_are_not_degraded(self):
        for p in self.pages:
            self.assertFalse(p.degraded, p.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
