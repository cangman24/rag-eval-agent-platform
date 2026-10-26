# -*- coding: utf-8 -*-
"""core/doc_vision_tools.py —— 把 doc_vision 包装成 **Agent 可调用的工具**。

为什么值得单独一层：
    core/doc_vision.py 是**算法**（吃 span、吐版图），不知道文件、不知道协议；
    这里负责三件事：
      ① 路径 -> 版面：读文件、翻页、拼成 Agent 能消费的紧凑结构
      ② 失败降级：缺 pypdf / 文件不存在 / 页号越界 -> 返回带 reason 的
         降级结果，**不抛异常**（工具抛异常会把 Agent 的整轮对话打断）
      ③ 结构化寻址：给每个元素发 element_id，Agent 引用时说的是
         "第 2 页 el-3（表格）第 3 行"，而不是复述一大段文字

注册进 ToolRegistry 后即可被 MCP 协议层（core/mcp_server.py）暴露给任意
MCP 客户端 —— 这是"多模态文档理解"从算法走到产品的那一步。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .doc_vision import (
    MissingDependency, analyze_pdf, read_bars_from_rects, spans_from_pdf,
)
from .mcp import ToolRegistry

# 单页返回的元素数上限（防止整本 PDF 一次性塞爆上下文）
MAX_ELEMENTS_PER_PAGE = 40


def _degraded(reason: str, **extra: Any) -> Dict[str, Any]:
    out = {"degraded": True, "reason": reason}
    out.update(extra)
    return out


def _element_brief(el: Any) -> Dict[str, Any]:
    brief = {
        "id": el.element_id,
        "kind": el.kind,
        "text": el.text,
        "bbox": [round(v, 1) for v in el.bbox],
        "confidence": el.confidence,
    }
    if el.level:
        brief["level"] = el.level
    if el.kind == "table":
        brief["columns"] = el.extra.get("columns")
        brief["rows"] = el.extra.get("rows")
    if el.kind == "figure" and "chart" in (el.extra or {}):
        chart = el.extra["chart"]
        brief["chart"] = {
            "bars": len(chart.get("bars") or []),
            "unit": chart.get("unit"),
            "degraded": chart.get("degraded", True),
        }
    return brief


def register_doc_vision_tools(registry: ToolRegistry) -> ToolRegistry:
    """往工具注册表里挂上三个文档理解工具，返回同一个 registry 便于链式调用。"""

    @registry.tool(description="分析 PDF 版面：标题/段落/表格/图表分区与阅读顺序")
    def doc_analyze(path: str, max_pages: int = 5) -> Dict[str, Any]:
        """分析 PDF 的版面结构，返回每页的元素清单（含 element_id 供引用）

        path: PDF 文件路径
        max_pages: 最多分析几页（默认 5）
        """
        if not os.path.exists(path):
            return _degraded(f"文件不存在: {path}")
        try:
            pages = analyze_pdf(path, max_pages=max_pages)
        except MissingDependency as exc:
            return _degraded(str(exc), hint="pip install pypdf")
        except Exception as exc:  # noqa: BLE001 - 工具层不抛，交给 Agent 决策
            return _degraded(f"解析失败: {type(exc).__name__}: {exc}")
        out_pages: List[Dict[str, Any]] = []
        for i, layout in enumerate(pages):
            els = [_element_brief(e) for e in layout.elements[:MAX_ELEMENTS_PER_PAGE]]
            out_pages.append({
                "page": i + 1,
                "degraded": layout.degraded,
                "reason": layout.reason,
                "truncated": len(layout.elements) > MAX_ELEMENTS_PER_PAGE,
                "elements": els,
            })
        return {"degraded": False, "pages": out_pages,
                "n_pages": len(out_pages)}

    @registry.tool(description="读取 PDF 里的柱状图数据（矢量精确 / 位图近似）")
    def doc_read_chart(path: str, page: int = 1) -> Dict[str, Any]:
        """把 PDF 页面里的柱状图读成数据表 [(标签, 数值), ...]

        path: PDF 文件路径
        page: 页码，从 1 开始
        """
        if not os.path.exists(path):
            return _degraded(f"文件不存在: {path}")
        try:
            raw = spans_from_pdf(path)
        except MissingDependency as exc:
            return _degraded(str(exc), hint="pip install pypdf")
        except Exception as exc:  # noqa: BLE001
            return _degraded(f"解析失败: {type(exc).__name__}: {exc}")
        idx = page - 1
        if not (0 <= idx < len(raw)):
            return _degraded(f"页码越界: 共 {len(raw)} 页，请求第 {page} 页")
        page_data = raw[idx]
        width = page_data["page_size"][0]
        # 刻度标签：页面左侧的纯数字文字（y 轴刻度）
        numeric = [s for s in page_data["spans"]
                   if s.text.strip().replace(".", "").replace("-", "").isdigit()]
        ticks = [((s.y0 + s.y1) / 2, s.text) for s in numeric
                 if s.x0 < width * 0.35]
        # 类目标签：非数值文字，用于给柱体命名（"周四"）
        labels = [((s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2, s.text.strip())
                  for s in page_data["spans"] if s not in numeric]
        chart = read_bars_from_rects(page_data["rects"], ticks, label_spans=labels)
        if chart["degraded"]:
            return _degraded(chart.get("reason", "图表读数降级"),
                             bars=chart.get("bars") or [])
        return {"degraded": False, "bars": chart["bars"],
                "n_bars": len(chart["bars"]), "unit": chart.get("unit")}

    @registry.tool(description="读取 PDF 里的表格：返回二维网格，可按行列寻址")
    def doc_read_table(path: str, page: int = 1, index: int = 0) -> Dict[str, Any]:
        """把 PDF 页面里的表格读成二维网格

        path: PDF 文件路径
        page: 页码，从 1 开始
        index: 该页第几个表格，从 0 开始
        """
        if not os.path.exists(path):
            return _degraded(f"文件不存在: {path}")
        try:
            pages = analyze_pdf(path)
        except MissingDependency as exc:
            return _degraded(str(exc), hint="pip install pypdf")
        except Exception as exc:  # noqa: BLE001
            return _degraded(f"解析失败: {type(exc).__name__}: {exc}")
        idx = page - 1
        if not (0 <= idx < len(pages)):
            return _degraded(f"页码越界: 共 {len(pages)} 页，请求第 {page} 页")
        tables = pages[idx].by_kind("table")
        if not tables:
            return _degraded("该页没有识别到表格")
        if not (0 <= index < len(tables)):
            return _degraded(f"表格序号越界: 该页共 {len(tables)} 个表格")
        t = tables[index]
        return {"degraded": False, "id": t.element_id,
                "columns": t.extra["columns"], "rows": t.extra["rows"],
                "grid": t.extra["grid"], "markdown": t.extra["markdown"]}

    return registry


def build_doc_vision_registry() -> ToolRegistry:
    """只含文档理解工具的独立注册表（便于单独起 MCP server）。"""
    return register_doc_vision_tools(ToolRegistry())


__all__ = [
    "register_doc_vision_tools", "build_doc_vision_registry",
    "MAX_ELEMENTS_PER_PAGE", "MissingDependency",
]
