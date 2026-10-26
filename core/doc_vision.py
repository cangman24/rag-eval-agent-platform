# -*- coding: utf-8 -*-
"""core/doc_vision.py —— 多模态文档理解：版式分析 + 表格重建 + 图表读数。

对应简历里那句（E3 · 差异化核心）：
「将多模态文档理解封装为 MCP 工具（版面检测 + 表格解析 + 图表结构化），
实现上传含图表 PDF -> 定位图表 -> 基于图表内容问答 + 引用的端到端链路，
低置信自动降级纯文本检索」。

★ 设计主线：**结构化注入，而不是自然语言转述**。

    常见做法是把 PDF 整页丢给多模态模型，让它"描述"页面 —— 拿到的是
    一段不可校验的散文。本模块反过来：把页面拆成**带坐标、带类型、
    带阅读顺序的结构化元素**（heading / paragraph / table / figure），
    表格还原成网格、图表读出成数据表，再注入 Agent 上下文。
    每一个数字都能回溯到像素或坐标，模型只需要"基于结构化事实回答"，
    不需要"猜页面里有什么"。

★ 三层结构（接缝设计，每层可独立验证）：

    ① 纯算法层（本文件，零第三方依赖，CI 全覆盖）
       analyze_spans() / rebuild_table() / read_bar_chart()
       输入是普通 Python 数据结构（Span 列表 / PIL  Image 二选一），
       所以测试不需要 PDF 也不需要图像库。
    ② 适配器层（懒导入，真机路径）
       spans_from_pdf()   用 pypdf 提"文字 + 坐标 + 字号 + 图像块"
       chart_pixels()     用 PIL 解出 PDF 内嵌图像
    ③ 工具层（MCP）
       doc_vision 工具经 core/mcp_server 暴露，Agent 经 JSON-RPC 调用。

★ 关于"图表理解用 VLM 还是自训小模型"（面试必问）的诚实回答：
    本模块的图表读数走**规则式**（连通区域 + 像素测量 + 坐标轴线性映射），
    只对干净渲染的垂直柱状图可靠，且明确返回 confidence；
    照片级、3D、堆叠、曲线图一律低置信降级 —— 降级路径比覆盖率重要。
    版面检测同理：当前是几何规则（投影 + 对齐聚类），生产环境换成
    YOLO 微调版面模型时只需实现同一组接口（见 spans_from_* 的契约），
    算法层一行不改。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------- 数据模型
@dataclass
class Span:
    """一个文字片段：文字 + 包围盒 + 字号。

    坐标系约定（★ 踩坑记录）：PDF 原点在左下、y 向上；图像原点在左上、
    y 向下。本模块内部统一使用**屏幕坐标系（原点左上，y 向下）**，
    适配器负责转换 —— 混用两种坐标是版面分析最经典的错源。
    """

    text: str
    x0: float
    y0: float          # 上边
    x1: float
    y1: float          # 下边
    size: float = 0.0  # 字号（pt）；0 表示未知
    bold: bool = False

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)


@dataclass
class PageElement:
    """页面里的一个结构化元素。

    kind: heading / paragraph / table / figure / chart
    level: 标题层级（正文 0，H1=1 ...）；其余类型为 0
    element_id: 稳定 id（如 "el-3"），Agent 引用溯源用
    """

    kind: str
    bbox: Tuple[float, float, float, float]      # (x0, y0, x1, y1)
    text: str = ""
    level: int = 0
    confidence: float = 1.0
    element_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.element_id,
            "kind": self.kind,
            "bbox": [round(v, 2) for v in self.bbox],
            "confidence": round(self.confidence, 3),
        }
        if self.kind == "heading":
            d["level"] = self.level
        if self.text:
            d["text"] = self.text
        if self.extra:
            d.update(self.extra)
        return d


@dataclass
class PageLayout:
    """整页的分析结果。degraded=True 时 elements 可能不完整，reason 说明原因。"""

    elements: List[PageElement] = field(default_factory=list)
    degraded: bool = False
    reason: str = ""
    page_size: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_size": list(self.page_size),
            "degraded": self.degraded,
            "reason": self.reason,
            "elements": [e.to_dict() for e in self.elements],
        }

    def by_kind(self, kind: str) -> List[PageElement]:
        return [e for e in self.elements if e.kind == kind]


# --------------------------------------------------------------------- 行/块聚类
def _cluster_lines(spans: Sequence[Span], y_tol: float = 3.0) -> List[List[Span]]:
    """把 y 中心相近的 span 归成一行，行内按 x 排序。

    y_tol 的经验值：同字号行距通常 >= 1.2*size，3pt 足以合并基线与
    上标/下标而不粘起相邻行。
    """
    rows: List[List[Span]] = []
    for s in sorted(spans, key=lambda s: ((s.y0 + s.y1) / 2, s.x0)):
        cy = (s.y0 + s.y1) / 2
        for row in rows:
            rcy = sum((r.y0 + r.y1) / 2 for r in row) / len(row)
            if abs(cy - rcy) <= max(y_tol, 0.5 * s.size):
                row.append(s)
                break
        else:
            rows.append([s])
    for row in rows:
        row.sort(key=lambda s: s.x0)
    return rows


def _merge_row(row: Sequence[Span]) -> Tuple[str, float, float, float, float, float, bool]:
    """一行 span 合并成 (text, x0, y0, x1, y1, max_size, any_bold)。"""
    text = " ".join(s.text.strip() for s in row if s.text.strip())
    x0 = min(s.x0 for s in row)
    y0 = min(s.y0 for s in row)
    x1 = max(s.x1 for s in row)
    y1 = max(s.y1 for s in row)
    size = max((s.size for s in row), default=0.0)
    bold = any(s.bold for s in row)
    return text, x0, y0, x1, y1, size, bold


def _cluster_row_indexes(rows: Sequence[Sequence[Span]], gap_factor: float = 0.8
                         ) -> List[List[int]]:
    """行 -> 块（返回**行下标**分组）：行间距超过 gap_factor*行高 视为新块。

    返回下标而不是 span 集合，是为了让调用方能拿回"这个块从第几行开始"
    —— 阅读顺序要按行号排，光有 span 集合会丢掉这个信息。
    """
    if not rows:
        return []
    merged = [_merge_row(r) for r in rows]
    blocks: List[List[int]] = []
    prev_bottom: Optional[float] = None
    prev_size = 0.0
    for i, m in enumerate(merged):
        top = m[2]
        size = m[5] or 10.0
        new_block = prev_bottom is None or (top - prev_bottom) > gap_factor * max(size, prev_size)
        if new_block:
            blocks.append([])
        blocks[-1].append(i)
        prev_bottom = m[4]
        prev_size = size
    return blocks


def _cluster_blocks(rows: Sequence[Sequence[Span]], gap_factor: float = 0.8
                    ) -> List[List[Span]]:
    """行 -> 块：行间距超过 gap_factor*行高 视为新块（投影分割的 1D 版）。"""
    return [[s for i in idxs for s in rows[i]]
            for idxs in _cluster_row_indexes(rows, gap_factor)]


# --------------------------------------------------------------------- 标题判定
def _body_size(sizes: Sequence[float]) -> float:
    """正文基准字号 = **众数**（出现最多的字号），平票取较小者。

    ★ 踩坑记录：一开始用中位数，块数为偶数时取上半位——
      一页只有「标题 18pt + 一段正文 10pt」两个块时中位数是 18，
      标题拿自己当基准，判不出标题（实测：heading 测试直接 0 个）。
      正文在页数上占绝对多数，众数才是对的统计口径；平票取小
      （宁可把基准定低，让真正的大字号浮出来）。
    """
    known = [round(s, 1) for s in sizes if s > 0]
    if not known:
        return 0.0
    counts: Dict[float, int] = {}
    for s in known:
        counts[s] = counts.get(s, 0) + 1
    top = max(counts.values())
    return min(s for s, c in counts.items() if c == top)


def _heading_levels(block_sizes: Sequence[float], body_size: float) -> List[int]:
    """字号 -> 标题层级。只有显著大于正文的字号才算标题（防瞎猜）。"""
    levels: List[int] = []
    seen = sorted({round(s, 1) for s in block_sizes if s > body_size * 1.12},
                  reverse=True)
    rank = {s: i + 1 for i, s in enumerate(seen[:3])}   # 最多三级
    for s in block_sizes:
        levels.append(rank.get(round(s, 1), 0))
    return levels


# --------------------------------------------------------------------- 表格区域
def _row_col_anchors(row: Sequence[Span], x_tol: float) -> List[float]:
    return _cluster_values([s.x0 for s in row], tol=x_tol)


def _is_table_row(row: Sequence[Span], min_cols: int, x_tol: float) -> bool:
    """单行是否"像表格行"：至少 min_cols 个 span 且 x 起点聚成 >= min_cols 列。"""
    return len(row) >= min_cols and len(_row_col_anchors(row, x_tol)) >= min_cols


def _table_regions(rows: Sequence[Sequence[Span]], *,
                   min_cols: int, min_rows: int,
                   x_tol: float = 6.0,
                   anchor_ratio: float = 0.6) -> List[Tuple[int, int]]:
    """在**行序列**上找表格区域，返回 [start_row, end_row) 区间列表。

    ★ 踩坑记录：表格判定不能放在"块"上做。表头行与数据行之间行距稍大
      就会被 _cluster_blocks 切成两个块，表头于是被当成普通段落，
      重建出的网格直接少了首行（实测：4 列表格只剩 3 行数据）。

    真正的表格特征是**跨行对齐**，用这个来防误判：
      - 先找连续"像表格行"的行（每行 >= min_cols 列）
      - 再把整组 x 起点聚类，只保留出现在 >= anchor_ratio 行里的列锚点
      - 锚点数 >= min_cols 且对齐到锚点的行数 >= min_rows 才算表格

      普通段落虽然一行也可能有多个 x 簇，但**行与行之间位置参差**，
      凑不出跨行一致的列锚点 —— 这是表格与段落最可靠的分界线。
    """
    regions: List[Tuple[int, int]] = []
    i, n = 0, len(rows)
    while i < n:
        if not _is_table_row(rows[i], min_cols, x_tol):
            i += 1
            continue
        j = i
        while j < n and _is_table_row(rows[j], min_cols, x_tol):
            j += 1
        group = rows[i:j]
        if len(group) >= min_rows:
            anchors = _cluster_values([s.x0 for r in group for s in r], tol=x_tol)
            # 每个锚点出现在多少行里
            hit = [0] * len(anchors)
            for r in group:
                for x0 in (s.x0 for s in r):
                    k = min(range(len(anchors)), key=lambda q: abs(x0 - anchors[q]))
                    if abs(x0 - anchors[k]) <= x_tol:
                        hit[k] += 1
            shared = [a for a, h in zip(anchors, hit, strict=True)
                  if h >= anchor_ratio * len(group)]
            if len(shared) >= min_cols:
                aligned = sum(
                    1 for r in group
                    if sum(1 for x0 in (s.x0 for s in r)
                           if any(abs(x0 - a) <= x_tol for a in shared)) >= min_cols)
                if aligned >= min_rows:
                    regions.append((i, j))
        i = j
    return regions


# --------------------------------------------------------------------- 双栏检测
def _two_column_hint(spans: Sequence[Span], *,
                     min_gutter: float = 24.0) -> bool:
    """在 **span 层**检测双栏：找一条没有文字跨越的竖直"栏沟"。

    ★ 踩坑记录：不能在元素层做。左右两栏同一基线的文字行，
      y 中心相差 < 3pt，会被 _cluster_lines 并成**同一行**，
      进而并成同一个块、同一个元素 —— 元素层永远看不到"两个元素
      左右并排"，检测器静默失效（实测：reason 恒为空字符串）。

    做法：候选栏沟取相邻 span 边界之间的空隙（宽 >= min_gutter），
    且沟两侧都有文字、y 区间重叠 —— 这才是"并排两栏"的定义。
    """
    if len(spans) < 2:
        return False
    edges = sorted({round(s.x0) for s in spans} | {round(s.x1) for s in spans})
    for g0, g1 in zip(edges, edges[1:], strict=False):
        if g1 - g0 < min_gutter:
            continue
        left = [s for s in spans if s.x1 <= g0]
        right = [s for s in spans if s.x0 >= g1]
        if not left or not right:
            continue
        left_hits: List[Span] = []
        right_hits: List[Span] = []
        for a in left:
            for b in right:
                # ★ 必须**严格正重叠**：左右两栏要共享一条横带。
                #   一开始给了 -6pt 容差（想兼容两栏基线错开），结果
                #   图表的 y 轴刻度 "0" 与 x 轴类目 "Mon" 只差 0.2pt
                #   就被算成并排 —— 单栏页面被误报成双栏。
                if min(a.y1, b.y1) - max(a.y0, b.y0) > 0:
                    if a not in left_hits:
                        left_hits.append(a)
                    if b not in right_hits:
                        right_hits.append(b)
        # ★ 两侧都要有**至少两个不同的行**：一栏是"一竖列文字"，不是
        #   "一行散落的标签"。图表的 x 轴类目（Mon/Tue/Wed…）全在同一
        #   行上彼此隔开 —— 按 span 个数数会中招（实测误报过两次），
        #   必须按 y 带数行数。
        if _n_rows(left_hits) >= 2 and _n_rows(right_hits) >= 2:
            return True
    return False


def _n_rows(spans: Sequence[Span], tol: float = 3.0) -> int:
    """这些 span 分布在几个不同的行（y 带）上。"""
    ys = sorted((s.y0 + s.y1) / 2 for s in spans)
    if not ys:
        return 0
    n = 1
    for prev, cur in zip(ys, ys[1:], strict=False):
        if cur - prev > tol:
            n += 1
    return n


# --------------------------------------------------------------------- 表格重建
def _cluster_values(values: Sequence[float], tol: float) -> List[float]:
    """一维聚类：返回每组的代表值（均值）。tol 通常取 2~4pt。"""
    groups: List[List[float]] = []
    for v in sorted(values):
        if groups and abs(v - groups[-1][-1]) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def rebuild_table(spans: Sequence[Span], x_tol: float = 6.0, y_tol: float = 4.0
                  ) -> Dict[str, Any]:
    """把表格区域内的 span 重排成网格（纯对齐算法，不需要表格线）。

    列 = x0 聚类；行 = y0 聚类；单元格取该行该列的所有文字。
    ★ 返回 markdown 与二维数组两种形态：前者给人看，后者给 Agent 用
      （"第 2 行第 3 列的值"这种提问需要结构化寻址）。
    """
    rows = _cluster_lines(spans, y_tol=y_tol)
    row_anchors = [min(s.x0 for s in r) for r in rows]
    col_anchors = _cluster_values([s.x0 for s in spans], tol=x_tol)
    grid: List[List[str]] = []
    for r in rows:
        cells = ["" for _ in col_anchors]
        for s in r:
            ci = min(range(len(col_anchors)),
                     key=lambda i: abs(s.x0 - col_anchors[i]))
            cells[ci] = (cells[ci] + " " + s.text.strip()).strip()
        grid.append(cells)
    # 丢弃全空列（右边界噪声）
    keep = [i for i in range(len(col_anchors)) if any(row[i] for row in grid)]
    grid = [[row[i] for i in keep] for row in grid]
    return {
        "columns": len(keep),
        "rows": len(grid),
        "grid": grid,
        "markdown": _grid_to_markdown(grid),
        "row_anchors": [round(v, 1) for v in row_anchors],
    }


def _grid_to_markdown(grid: Sequence[Sequence[str]]) -> str:
    if not grid:
        return ""
    header = grid[0]
    out = ["| " + " | ".join(c or " " for c in header) + " |",
           "|" + "---|" * len(header)]
    for row in grid[1:]:
        out.append("| " + " | ".join(c or " " for c in row) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------- 图表读数
def read_bar_chart(pixels: Any, *,
                   plot_box: Tuple[int, int, int, int],
                   y_ticks: Sequence[Tuple[int, float]],
                   x_step: Optional[float] = None,
                   bar_color: Optional[Tuple[int, int, int]] = None,
                   color_tol: int = 40,
                   min_area: int = 30) -> Dict[str, Any]:
    """规则式垂直柱状图读数（PIL/numpy 路径，懒导入由调用方保证）。

    Args:
        pixels: HxWx3 的类数组（PIL Image 转 numpy 即可）
        plot_box: 绘图区像素盒 (left, top, right, bottom)（坐标轴内侧）
        y_ticks: [(y_pixel, value), ...] 至少两个刻度用于线性映射
        x_step: 相邻柱中心的像素间距；None 时按连通区域自动估计
        bar_color: 柱体颜色；None 时取"最醒目的非灰非白彩色"
        color_tol: 颜色匹配容差（RGB 各通道）
        min_area: 小于该面积的色块忽略（噪点）

    Returns:
        {"bars": [{"index", "value", "confidence", "pixels"}...],
         "degraded": bool, "reason": str}

    ★ 局限（写进返回的 reason，不藏）：
        - 只支持垂直柱；堆叠/横向/3D/曲线一律低置信或降级
        - 依赖干净渲染；JPEG 边缘会产生 ±1px 换算误差（如实报告）
        - 生产环境换 CV 模型时保持同一返回契约即可
    """
    try:
        import numpy as np
    except ImportError:
        return {"bars": [], "degraded": True, "reason": "numpy 不可用"}

    arr = np.asarray(pixels)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {"bars": [], "degraded": True, "reason": "图像不是 RGB"}
    arr = arr[:, :, :3].astype(int)
    h, w, _ = arr.shape
    left, top, right, bottom = plot_box
    left, top = max(0, left), max(0, top)
    right, bottom = min(w - 1, right), min(h - 1, bottom)
    if right - left < 10 or bottom - top < 10:
        return {"bars": [], "degraded": True, "reason": "绘图区过小"}

    if len(y_ticks) < 2:
        return {"bars": [], "degraded": True, "reason": "y 轴刻度不足，无法标定"}
    # 线性标定：pixel -> value（y 像素向下增大，value 向上增大）
    (py0, v0), (py1, v1) = y_ticks[0], y_ticks[-1]
    if py1 == py0:
        return {"bars": [], "degraded": True, "reason": "y 轴刻度重合"}
    slope = (v1 - v0) / (py1 - py0)

    def px2val(py: float) -> float:
        return v0 + (py - py0) * slope

    if bar_color is None:
        # 取绘图区内最醒目的彩色（饱和度优先，灰白除外）
        region = arr[top:bottom + 1, left:right + 1]
        mx = region.max(axis=2)
        mn = region.min(axis=2)
        sat = mx - mn
        mask = (sat > 30) & (mx > 40)
        if not mask.any():
            return {"bars": [], "degraded": True, "reason": "绘图区内未找到彩色柱体"}
        colored = region[mask]
        bar_color = tuple(int(v) for v in colored.mean(axis=0))

    region = arr[top:bottom + 1, left:right + 1]
    dist = np.abs(region - np.array(bar_color)).sum(axis=2)
    mask = dist <= color_tol * 3

    # 列投影 -> 柱体（列连通段）
    col_has = mask.any(axis=0)
    segments: List[Tuple[int, int]] = []
    x = 0
    width = mask.shape[1]
    while x < width:
        if col_has[x]:
            x0 = x
            while x < width and col_has[x]:
                x += 1
            if x - x0 >= 2:
                segments.append((x0, x - 1))
        else:
            x += 1
    # 面积过滤（噪点）
    bars_px: List[Tuple[int, int, int]] = []
    for x0, x1 in segments:
        cols = mask[:, x0:x1 + 1]
        ys = np.where(cols.any(axis=1))[0]
        if len(ys) == 0:
            continue
        y_top = int(ys.min())
        area = int(cols.sum())
        if area < min_area:
            continue
        bars_px.append((x0 + left, x1 + left, y_top + top))
        bars_px[-1] = bars_px[-1]

    if not bars_px:
        return {"bars": [], "degraded": True, "reason": "未检测到柱体"}

    bars = []
    for i, (bx0, bx1, by_top) in enumerate(bars_px):
        value = px2val(by_top)
        # 置信度：柱体区域的色彩纯度（越纯说明边缘越干净、读数越可靠）
        col_area = mask[:, bx0 - left:bx1 - left + 1].sum()
        purity = col_area / max(1, (bx1 - bx0 + 1) * (bottom - by_top + 1))
        bars.append({
            "index": i,
            "x_pixel": (bx0 + bx1) // 2,
            "top_pixel": by_top,
            "value": round(value, 3),
            "confidence": round(max(0.0, min(1.0, float(purity))), 3),
        })
    return {
        "bars": bars,
        "bar_color": list(bar_color),
        "calibration": {"pixel_per_unit": round(slope, 4),
                        "y_ticks": [[p, v] for p, v in y_ticks]},
        "degraded": False,
        "reason": "",
    }


# --------------------------------------------------------------------- 主入口
def analyze_spans(spans: Sequence[Span], *,
                  page_size: Tuple[float, float] = (0.0, 0.0),
                  figure_boxes: Optional[Sequence[Tuple[float, float, float, float]]] = None,
                  min_table_cols: int = 3,
                  min_table_rows: int = 2) -> PageLayout:
    """版式分析主入口（纯算法，零依赖）。

    Args:
        spans: 文字片段（屏幕坐标：原点左上，y 向下）
        page_size: (width, height)；0 表示未知
        figure_boxes: 图像块包围盒（PDF 提取时给出）；没有则为空
        min_table_cols/min_table_rows: 判定表格的最小规模（防把普通
            多列文字误判成表格）

    降级原则：**任何不确定都显式降级并说明原因**，不猜。
    """
    spans = [s for s in spans if s.text and s.text.strip()]
    if not spans:
        return PageLayout(degraded=True, reason="页面无文字", page_size=page_size)

    if not page_size[0]:
        page_size = (max(s.x1 for s in spans), max(s.y1 for s in spans))

    rows = _cluster_lines(spans)
    row_meta = [_merge_row(r) for r in rows]

    # 正文基准字号：正文占绝对多数，用众数（见 _body_size 的踩坑记录）
    body_size = _body_size([m[5] for m in row_meta])

    # ① 表格区域（在行序列上判定，跨行对齐是唯一可靠特征）
    regions = _table_regions(rows, min_cols=min_table_cols, min_rows=min_table_rows)
    table_at = {}
    for start, end in regions:
        table_at[start] = end

    # ② 非表格行 -> 块 -> 标题/段落
    #    只在非表格行上做块聚类：否则表格上方/下方的正文会被表格行粘住，
    #    而且表格区域自己会被判成一个"块"再错判成段落。
    plain_idx = [i for i in range(len(rows)) if i not in table_at]
    row_blocks = _cluster_row_indexes([rows[i] for i in plain_idx])
    blocks: List[Tuple[int, List[Span], int]] = []   # (起行号, span 集合, 层级)
    sizes_of_blocks = []
    for idxs in row_blocks:
        spans_of_block = [s for k in idxs for s in rows[plain_idx[k]]]
        sizes_of_blocks.append(max((s.size for s in spans_of_block), default=0.0))
    levels = _heading_levels(sizes_of_blocks, body_size)
    for idxs, level in zip(row_blocks, levels, strict=True):
        spans_of_block = [s for k in idxs for s in rows[plain_idx[k]]]
        blocks.append((plain_idx[idxs[0]], spans_of_block, level))

    # ③ 按行号顺序组装（阅读顺序 = 版面顺序，表格整体占它那段行）
    pieces: List[Tuple[int, PageElement]] = []
    for start, end in regions:
        group = [s for r in rows[start:end] for s in r]
        _, x0, y0, x1, y1, _, _ = _merge_row(group)
        table = rebuild_table(group)
        pieces.append((start, PageElement(
            kind="table", bbox=(x0, y0, x1, y1), text=table["markdown"],
            element_id="", confidence=0.85,
            extra={"columns": table["columns"], "rows": table["rows"],
                   "grid": table["grid"], "markdown": table["markdown"]})))
    for start_row, spans_of_block, level in blocks:
        text, x0, y0, x1, y1, size, bold = _merge_row(spans_of_block)
        if level > 0 and len(text) <= 60:
            el = PageElement(kind="heading", bbox=(x0, y0, x1, y1), text=text,
                             level=level, element_id="",
                             confidence=0.9 if bold or size > body_size * 1.3 else 0.75)
        else:
            el = PageElement(kind="paragraph", bbox=(x0, y0, x1, y1), text=text,
                             element_id="", confidence=0.95)
        pieces.append((start_row, el))

    pieces.sort(key=lambda p: p[0])
    elements: List[PageElement] = []
    for _, el in pieces:
        el.element_id = f"el-{len(elements)}"
        elements.append(el)

    # 图像块 -> figure / chart
    for fb in figure_boxes or []:
        elements.append(PageElement(
            kind="figure", bbox=tuple(fb), element_id=f"el-{len(elements)}",
            text="（图像区域，内容需经 OCR/图表读数进一步解析）",
            confidence=0.6,
            extra={"needs": ["ocr", "chart_read"]}))

    # 阅读顺序：块按 y 再按 x
    elements.sort(key=lambda e: (round(e.bbox[1], 1), e.bbox[0]))

    # 双栏检测只看**正文**：表格单元格之间天然有宽缝，形态上和"栏沟"
    # 一模一样（实测：单栏 PDF 里的 'Category' 与 'Open' 之间隔了 33pt，
    # 被误判成双栏）。表格由表格检测器负责，这里必须排除掉。
    table_spans = {id(s) for start, end in regions
                   for r in rows[start:end] for s in r}
    prose = [s for s in spans if id(s) not in table_spans]
    reason = ""
    if _two_column_hint(prose):
        reason = "疑似双栏版面，阅读顺序为近似值"

    return PageLayout(elements=elements, degraded=False, reason=reason,
                      page_size=page_size)


# --------------------------------------------------------------------- PDF 适配器
def _image_placements(page: Any, reader: Any,
                      page_height: float) -> List[Tuple[float, float, float, float]]:
    """从页面内容流解析内嵌图像的放置位置。

    PDF 里图像经 `q a b c d e f cm /ImX Do Q` 绘制：cm 矩阵把单位方形
    映射到 (e, f)-(e+a, f+d)（PDF 坐标，原点左下、y 向上）。
    这里转成屏幕坐标（y 向下）返回。

    ★ 踩坑记录：pypdf 的 page.images 只给图像数据、不给位置；
      位置在内容流里。第一版图省事返回"整页"当 figure box，
      等于把版面分析的功劳全归给运气——不行，得真解析。
    """
    placements: List[Tuple[float, float, float, float]] = []
    try:
        import pypdf  # noqa: F401 - 确认依赖可用
        from pypdf.generic import ContentStream, NameObject
    except ImportError as exc:
        raise MissingDependency("解析 PDF 需要 pypdf：pip install pypdf") from exc

    try:
        contents = page.get_contents()
        if contents is None:
            return placements
        if isinstance(contents, list):
            raw = b"".join(
                c.get_data() if hasattr(c, "get_data") else bytes(c) for c in contents)
        else:
            raw = contents.get_data() if hasattr(contents, "get_data") else bytes(contents)
        stream = ContentStream(raw, reader)
    except Exception:  # noqa: BLE001 - 解析失败只丢图像位置，不丢文字
        return placements

    last_cm: List[float] = []
    for operands, operator in stream.operations:
        op = str(operator)
        if op == "cm" and len(operands) >= 6:
            try:
                last_cm = [float(o) for o in operands[-6:]]
            except (TypeError, ValueError):
                last_cm = []
        elif op == "Do" and operands:
            name = operands[-1]
            is_image = isinstance(name, NameObject) and not str(name).startswith("/Form")
            if not (is_image and len(last_cm) == 6):
                continue
            a, b, c, d, e, f = last_cm
            # 单位方形经线性变换；忽略旋转/斜切（b=c=0 才是我们支持的 upright 图）
            x0, x1 = e, e + a
            y0, y1 = f, f + d
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            placements.append((x0, page_height - y1, x1, page_height - y0))
    return placements


def _make_text_visitor(sink: List[Span], page_height: float):
    """生成 pypdf 的 visitor_text 回调：把文字 span 收进 sink。

    PDF 坐标（原点左下、y 向上）在这里翻转成屏幕坐标（y 向下）——
    两种坐标混用是版面分析最经典的错源，翻转只在这一处做。
    """
    def visitor_text(text, cm, tm, font_dict, font_size):
        if not text.strip():
            return
        # tm[4]/tm[5] 是文字原点（PDF 坐标，y 向上）
        x, y = tm[4], tm[5]
        size = abs(font_size) if font_size else 10.0
        # 宽度估算：0.55*size/字 是中英文折中；只影响包围盒右边界，
        # 不影响结构判定（标题/表格都看不右边界）
        width = len(text) * size * 0.55
        sink.append(Span(text=text, x0=x, y0=page_height - y - size * 0.8,
                         x1=x + width, y1=page_height - y + size * 0.3,
                         size=size,
                         bold="Bold" in str(font_dict.get("/BaseFont", ""))))
    return visitor_text


def spans_from_pdf(path: str, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
    """pypdf 适配器：PDF -> 每页 {spans, figure_boxes, page_size}。

    ★ 懒导入：CI 零依赖环境不装 pypdf，调用方得到明确的 MissingDependency
      而不是 ImportError 栈。
    ★ 坐标转换：pypdf 的 text matrix 原点在左下、y 向上；这里转成屏幕
      坐标（y 向下）——两种坐标混用是版面分析最经典的错源。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise MissingDependency(
            "解析 PDF 需要 pypdf：pip install pypdf") from exc

    reader = PdfReader(path)
    pages: List[Dict[str, Any]] = []
    all_pages = reader.pages[:max_pages] if max_pages else reader.pages
    for pno, page in enumerate(all_pages):
        mb = page.mediabox
        pw, ph = float(mb.width), float(mb.height)
        spans: List[Span] = []
        figures = _image_placements(page, reader, ph)

        # ★ 用工厂函数绑定 spans/ph：直接把 def 写在循环里会捕获循环变量
        #   （ruff B023）——目前同步调用侥幸不出错，一旦将来改成延迟回调
        #   （比如并发解析多页）就会全部写到最后一页的列表里。
        visitor_text = _make_text_visitor(spans, ph)

        try:
            page.extract_text(visitor_text=visitor_text)
        except Exception:  # noqa: BLE001 - 单页失败不该杀掉整篇
            pass

        pages.append({"page": pno, "spans": spans, "figure_boxes": figures,
                      "rects": _filled_rects(page, reader, ph),
                      "page_size": (pw, ph)})
    return pages


class MissingDependency(Exception):
    """可选依赖缺失（与 core/loaders.py 同一约定）。"""


def attach_chart_elements(layout: PageLayout,
                          rects: Sequence[Tuple[float, float, float, float, str]],
                          tick_spans: Sequence[Tuple[float, str]],
                          label_spans: Optional[Sequence[Tuple[float, float, str]]] = None
                          ) -> PageLayout:
    """把矢量图表读数挂成版图里的 figure 元素（带 element_id 供引用）。

    版图分析看到的是文字；图表是矩形算子。两者在这里合并，
    输出才是 Agent 能直接消费的"一页结构化事实"。
    """
    chart = read_bars_from_rects(rects, tick_spans, label_spans=label_spans)
    if chart["degraded"] or not chart["bars"]:
        return layout
    xs = [r[0] for r in rects] + [r[2] for r in rects]
    ys = [r[1] for r in rects] + [r[3] for r in rects]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    layout.elements.append(PageElement(
        kind="figure", bbox=bbox, element_id=f"el-{len(layout.elements)}",
        text=f"柱状图（{len(chart['bars'])} 根，矢量精确读数）",
        confidence=1.0,
        extra={"chart": chart}))
    layout.elements.sort(key=lambda e: (round(e.bbox[1], 1), e.bbox[0]))
    return layout


def analyze_pdf(path: str, max_pages: Optional[int] = None) -> List[PageLayout]:
    """PDF -> 每页 PageLayout（适配器 + 分析器，真机路径）。"""
    out: List[PageLayout] = []
    for page in spans_from_pdf(path, max_pages=max_pages):
        layout = analyze_spans(page["spans"], page_size=page["page_size"],
                               figure_boxes=page["figure_boxes"])
        # 刻度标签：数值文字且在页面左侧（图表 y 轴旁）
        numeric = [s for s in page["spans"]
                   if s.text.strip().replace(".", "").replace("-", "").isdigit()]
        ticks = [((s.y0 + s.y1) / 2, s.text) for s in numeric
                 if s.x0 < page["page_size"][0] * 0.35]
        # 类目标签：非数值文字（x 轴下的 "Mon"/"周四" 之类），
        # 没有它柱体就没有名字，Agent 答不出"哪天最高"
        labels = [((s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2, s.text.strip())
                  for s in page["spans"] if s not in numeric]
        attach_chart_elements(layout, page["rects"], ticks, labels)
        out.append(layout)
    return out


# --------------------------------------------------------------------- 矢量图表
# 内容流分词：数字 or 算子（f* 这类带星号的算子也要能匹配）
_CONTENT_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[A-Za-z][A-Za-z*'\"]{0,3}")
_COLOR_OPS = {"rg", "g", "k", "sc", "scn", "RG", "G", "K", "SC", "SCN"}
_FILL_OPS = {"f", "F", "f*", "b", "b*", "B", "B*"}


def _filled_rects(page: Any, reader: Any,
                  page_height: float) -> List[Tuple[float, float, float, float, str]]:
    """从内容流提取**填充矩形**（re + f/f* 算子）：(x0, y0, x1, y1, color)。

    矢量柱状图的柱体就是填充矩形。与图像路径互补：
    矢量图读数**精确**（算子即真相），位图读数**近似**（像素测量）。
    """
    try:
        import pypdf  # noqa: F401 - 仅确认依赖可用，解析走内容流文本
    except ImportError as exc:
        raise MissingDependency("解析 PDF 需要 pypdf：pip install pypdf") from exc

    rects: List[Tuple[float, float, float, float, str]] = []
    try:
        contents = page.get_contents()
        if contents is None:
            return rects
        items = contents if isinstance(contents, (list, tuple)) else [contents]
        chunks = []
        for c in items:
            chunks.append(c.get_data() if hasattr(c, "get_data") else bytes(c))
        raw = b"".join(chunks)
    except Exception:  # noqa: BLE001 - 单页内容流取不到就当没有矢量图
        return rects

    # ★ 踩坑记录：不要 `ContentStream(raw, reader)`。
    #   pypdf 6.x 的 page.get_contents() **已经返回 ContentStream**，
    #   再拿它的字节去构造一个，会抛
    #   AttributeError: 'bytes' object has no attribute 'get_object'
    #   ——被 except 吞掉后表现是"静默返回 0 个矩形"，图表读数全线降级，
    #   而日志里什么都没有（最难查的那类失败）。
    #
    #   这里改为自己分词扫描**已解压**的内容流：只认 re / 填充 / 设色
    #   三类算子，不依赖 pypdf 的算子对象模型，跨版本都稳。
    pending: List[float] = []
    last_fill: Optional[str] = None
    nums: List[float] = []
    for token in _CONTENT_TOKEN_RE.findall(raw.decode("latin-1", "replace")):
        if token[0].isdigit() or token[0] in "+-.":
            try:
                nums.append(float(token))
            except ValueError:
                nums = []
            continue
        op = token
        if op in _COLOR_OPS:
            last_fill = ",".join(f"{v:.2f}" for v in nums[-4:]) if nums else last_fill
        elif op == "re" and len(nums) >= 4:
            pending = nums[-4:]
        elif op in _FILL_OPS and len(pending) == 4:
            x, y, w, h = pending
            if w > 0 and h > 0:
                rects.append((x, page_height - y - h, x + w, page_height - y,
                              last_fill or "unknown"))
            pending = []
        nums = []
    return rects


def read_bars_from_rects(rects: Sequence[Tuple[float, float, float, float, str]],
                         tick_spans: Sequence[Tuple[float, str]], *,
                         label_spans: Optional[Sequence[Tuple[float, float, str]]] = None,
                         min_bar_w: float = 4.0,
                         max_bar_w: float = 80.0,
                         bottom_tol: float = 6.0) -> Dict[str, Any]:
    """矢量柱状图读数：填充矩形 + 刻度标签 -> 数据表。

    Args:
        rects: (x0, y0, x1, y1, color) 屏幕坐标，来自 _filled_rects
        tick_spans: [(y_center_pixel, text), ...] 刻度标签（数值文字）
        label_spans: [(x_center, y_center, text), ...] x 轴类目标签（"周一"之类）；
            不给也能读数，但柱体 label 会是 None —— 而**没有类目名，
            Agent 就答不出"周几最高"**，所以调用方尽量传

    与 read_bar_chart 的分工：本函数读**精确**几何（PDF 矢量），
    read_bar_chart 读**近似**像素（位图）。两者返回同一种契约，
    调用方不需要知道图表是矢量还是位图。
    """
    ticks: List[Tuple[float, float]] = []
    for y_px, text in tick_spans:
        try:
            ticks.append((float(y_px), float(str(text).strip())))
        except ValueError:
            continue      # 非数值标签（如 "0%"）跳过；少于两个刻度会降级
    if len(ticks) < 2:
        return {"bars": [], "degraded": True,
                "reason": f"可用数值刻度不足（{len(ticks)}/2）"}

    (py0, v0), (py1, v1) = min(ticks, key=lambda t: t[0]), max(ticks, key=lambda t: t[0])
    if py1 == py0:
        return {"bars": [], "degraded": True, "reason": "刻度位置重合"}
    slope = (v1 - v0) / (py1 - py0)

    def px2val(py: float) -> float:
        return v0 + (py - py0) * slope

    # 轴底线 = 刻度最低处的 y（柱体底边应贴在上面）
    axis_y = py1
    bars: List[Dict[str, Any]] = []
    for x0, y0, x1, y1, color in rects:
        w, h = x1 - x0, y1 - y0
        if not (min_bar_w <= w <= max_bar_w):
            continue                      # 太宽的是背景/表格线，太细的是噪点
        if abs(y1 - axis_y) > bottom_tol + h * 0.05:
            continue                      # 底边不贴轴：不是柱子（可能是装饰块）
        bars.append({"x_pixel": round((x0 + x1) / 2, 1),
                     "top_pixel": round(y0, 1),
                     "value": round(px2val(y0), 3),
                     "confidence": 1.0,    # 矢量几何，无测量误差
                     "color": color})
    if not bars:
        return {"bars": [], "degraded": True, "reason": "未找到贴轴的柱体矩形"}
    bars.sort(key=lambda b: b["x_pixel"])

    # x 轴类目标签：取轴下方、水平位置最近的那条文字
    if label_spans:
        for b in bars:
            cands = [(abs(xc - b["x_pixel"]), text)
                     for xc, yc, text in label_spans
                     if yc >= axis_y - bottom_tol]
            if cands:
                _, label = min(cands, key=lambda t: t[0])
                b["label"] = label
    for i, b in enumerate(bars):
        b["index"] = i
        b.setdefault("label", None)
    return {"bars": bars, "degraded": False, "reason": "",
            "source": "vector", "calibration": {"slope": round(slope, 4)}}
