# -*- coding: utf-8 -*-
"""core/observability.py —— span 树：把一次 Agent 运行变成可归因的耗时树。

★ 为什么单独一个模块：

    事件流（core/agent.py 的 13 类 EventType）回答的是"发生了什么、按什么顺序"，
    但回答不了"慢在哪、错在哪"。这两件事的数据形状不一样：

        - 事件是**扁平序列**，适合画时间线；
        - 耗时是**树形结构**（一次 run 包含若干节点，节点包含若干工具调用），
          适合画甘特条，且必须能算"自身耗时"才能定位瓶颈。

    把二者混进同一个结构，会得到既不扁平也不成树的四不像。所以 span 走
    **独立通道**：不进事件流（事件流的精确序列被 859 项测试钉死，
    塞 span 进去会全红），而是在引擎内部维护、结束时落盘 `<tid>.spans.jsonl`。

有了本模块才能说出这种话：

    「本次 4.3s，其中 tool:kb_search 独占 890ms（20.7%），
      错误节点是 run > node:tools > tool:send_mail（白名单拦截）」

零依赖：只用标准库。时钟与 sink 都是注入的，便于确定性测试。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

# ---------------------------------------------------------------- 常量
KIND_RUN = "run"        # 整次运行（根）
KIND_NODE = "node"      # 一个图节点
KIND_TOOL = "tool"      # 一次工具调用

STATUS_OK = "ok"
STATUS_ERROR = "error"        # 抛异常
STATUS_BLOCKED = "blocked"    # 被白名单拦下（不是 bug，是安全策略生效）
STATUS_LIMIT = "limit"        # 熔断（轮次 / 工具数 / 超时 / 死循环）

_BAD_STATUS = (STATUS_ERROR, STATUS_BLOCKED, STATUS_LIMIT)


class Span:
    """一个有时间跨度的执行片段。

    ★ `t1 is None` 表示"还没结束"（或异常路径下没来得及结束），
      此时 `duration_ms` 返回 0 而不是抛异常 —— 未闭合的 span 是**常见情况**
      （进程被杀、节点抛异常），观测代码自己先崩掉是最糟的结果。
    """

    __slots__ = ("span_id", "parent_id", "name", "kind", "t0", "t1",
                 "status", "error", "attrs")

    def __init__(
        self,
        span_id: str,
        parent_id: Optional[str],
        name: str,
        kind: str = KIND_NODE,
        t0: float = 0.0,
        t1: Optional[float] = None,
        status: str = STATUS_OK,
        error: str = "",
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.kind = kind
        self.t0 = t0
        self.t1 = t1
        self.status = status
        self.error = error
        self.attrs = dict(attrs or {})

    # ------------------------------------------------------------ 计算
    @property
    def duration_ms(self) -> float:
        if self.t1 is None:
            return 0.0
        return round((self.t1 - self.t0) * 1000.0, 3)

    @property
    def closed(self) -> bool:
        return self.t1 is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "t0": self.t0,
            "t1": self.t1,
            "status": self.status,
            "error": self.error,
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Span":
        """★ 缺字段一律用默认值：轨迹文件可能只写了一半（进程被杀），
        回放时宁可显示"信息不全"，也不能抛 KeyError 让面板白屏。"""
        return cls(
            span_id=str(d.get("span_id") or "?"),
            parent_id=d.get("parent_id") or None,
            name=str(d.get("name") or "unnamed"),
            kind=str(d.get("kind") or KIND_NODE),
            t0=float(d.get("t0") or 0.0),
            t1=d.get("t1"),
            status=str(d.get("status") or STATUS_OK),
            error=str(d.get("error") or ""),
            attrs=d.get("attrs") or {},
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return "<Span %s %s %.1fms %s>" % (self.name, self.kind,
                                           self.duration_ms, self.status)


class SpanTracer:
    """收集 span 并组装成树。

    用法（推荐 with，异常自动标 error 并继续抛）::

        tr = SpanTracer()
        with tr.span("run", KIND_RUN):
            with tr.span("node:tools", KIND_NODE):
                with tr.span("tool:kb_search", KIND_TOOL, top_k=3) as sp:
                    result = call()
                    tr.finish(sp, hits=len(result))
    """

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._clock: Callable[[], float] = clock or time.time
        self._sink = sink
        self._spans: List[Span] = []
        self._stack: List[Span] = []
        self._sunk: set = set()
        self._seq = 0

    # ------------------------------------------------------------ 生命周期
    def reset(self, sink: Optional[Callable[[Dict[str, Any]], None]] = None
              ) -> None:
        """清空，开始新一次运行。`sink` 换了就一起换（落盘目标是 per-thread 的）。"""
        self._spans = []
        self._stack = []
        self._sunk = set()
        self._seq = 0
        if sink is not None:
            self._sink = sink

    def start(self, name: str, kind: str = KIND_NODE,
              parent_id: Optional[str] = None,
              **attrs: Any) -> Span:
        """开一个 span。不指定 parent 就挂在当前栈顶（嵌套的由来）。"""
        self._seq += 1
        parent = self._stack[-1] if self._stack else None
        pid = parent_id if parent_id is not None else (
            parent.span_id if parent else None)
        span = Span(span_id="s%d" % self._seq, parent_id=pid, name=name,
                    kind=kind, t0=self._clock(), attrs=attrs)
        self._stack.append(span)
        # ★ 开**始**就进列表，不等 finish：未闭合的 span（进程被杀 / 节点抛异常）
        #   也应该在甘特条上出现（0ms，标 error），而不是凭空消失 ——
        #   消失会让人误判"这个节点没执行"。
        self._spans.append(span)
        return span

    def finish(self, span: Span, status: Optional[str] = None,
               error: Optional[str] = None, **attrs: Any) -> Span:
        """闭合 span、落盘（经 sink）、出栈。

        ★ 幂等：重复 finish 同一个 span 不会把时长清成 0，也不会重复落盘。
          工具节点里既有正常 finish 又有异常分支时很容易双写。
        """
        if span.t1 is None:
            span.t1 = self._clock()
        if status is not None:
            span.status = status
        if error is not None:
            span.error = error
        if attrs:
            span.attrs.update(attrs)
        if span in self._stack:
            idx = self._stack.index(span)
            # ★ 比它更深的未闭合 span 一起兜底闭合：只 `del self._stack[idx:]`
            #   会把它们悄悄丢出栈，之后 close_all() 就找不到了 ——
            #   甘特条上会留一根永远 0ms 且状态是 ok 的条，看着像"瞬间成功"。
            orphans = list(reversed(self._stack[idx + 1:]))
            del self._stack[idx:]
            for orphan in orphans:
                self.finish(orphan, status=STATUS_ERROR,
                            error="父 span 已结束，子 span 仍未闭合")
        if span not in self._spans:      # 兜底：外部手工构造的 span
            self._spans.append(span)
        # ★ 幂等：只落盘一次。工具节点既有正常 finish 又有异常分支时容易双写，
        #   双写会让回放的甘特条出现两根一样的条。
        if span.span_id not in self._sunk:
            self._sunk.add(span.span_id)
            self._emit_sink(span)
        return span

    @contextmanager
    def span(self, name: str, kind: str = KIND_NODE,
             **attrs: Any) -> Iterator[Span]:
        """上下文管理器：正常结束标 ok，抛异常标 error 后**继续抛**。

        ★ 必须继续抛：观测层绝不能吞掉业务异常，否则"加了监控之后
          熔断不生效了"这种故事就来了（真实发生过一次：见 agent_langgraph 的
          纯函数踩坑记录，那次是吞掉后状态不更新）。
        """
        sp = self.start(name, kind, **attrs)
        try:
            yield sp
        except Exception as exc:  # noqa: BLE001 - 记录后原样抛出
            self.finish(sp, status=STATUS_ERROR,
                        error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.finish(sp)

    def close_all(self, status: str = STATUS_ERROR,
                  error: str = "span 未正常闭合（运行被中断？）") -> None:
        """兜底闭合栈里所有未结束的 span（由深到浅）。

        什么时候需要：图执行抛异常被 `_run_steps` 的 except 吞掉时，
        节点 span 就没人闭合了 —— 不兜底的话甘特条上会挂一排 0ms 的条，
        看起来像"所有节点都是瞬间完成"，比不画还糟。
        """
        for sp in list(reversed(self._stack)):
            self.finish(sp, status=status, error=error)
        self._stack = []

    def _emit_sink(self, span: Span) -> None:
        if self._sink is None:
            return
        try:
            self._sink(span.to_dict())
        except Exception:  # noqa: BLE001 - 旁路：落盘失败不影响主流程
            pass

    # ------------------------------------------------------------ 访问
    @property
    def spans(self) -> List[Span]:
        """按开始时间排序（稳定：同一时刻保持产生顺序）—— 甘特条要的就是这个序。"""
        return sorted(self._spans, key=lambda s: s.t0)

    @property
    def current(self) -> Optional[Span]:
        return self._stack[-1] if self._stack else None

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._spans]

    def load_dicts(self, dicts: List[Dict[str, Any]]) -> None:
        """从落盘数据重建（回放用）。会清空当前内容。"""
        self._spans = [Span.from_dict(d) for d in dicts]
        self._stack = []
        # 回放进来的 span 视为"已落盘"，避免再写一遍出现重复条
        self._sunk = {s.span_id for s in self._spans}
        self._seq = len(self._spans)

    # ------------------------------------------------------------ 分析
    def total_ms(self) -> float:
        if not self._spans:
            return 0.0
        starts = [s.t0 for s in self._spans]
        ends = [s.t1 for s in self._spans if s.t1 is not None]
        if not ends:
            return 0.0
        return round((max(ends) - min(starts)) * 1000.0, 3)

    def children_of(self, span_id: Optional[str]) -> List[Span]:
        kids = [s for s in self._spans if s.parent_id == span_id]
        return sorted(kids, key=lambda s: s.t0)

    def self_ms(self, span: Span) -> float:
        """自身耗时 = 总耗时 - 所有**直接**子 span 耗时之和。

        ★ 为什么必须扣掉孩子：不扣的话父节点（node:tools）永远比子节点
          （tool:kb_search）"慢"，瓶颈榜第一名恒为 node:tools，归因等于没做。
          真实原因"是这次检索慢"反而被藏在第二行。
        """
        kids_ms = sum(k.duration_ms for k in self.children_of(span.span_id))
        return round(max(span.duration_ms - kids_ms, 0.0), 3)

    def path_of(self, span: Span) -> str:
        """从根到该 span 的完整路径：`run > node:tools > tool:send_mail`。"""
        by_id = {s.span_id: s for s in self._spans}
        chain: List[str] = []
        cur: Optional[Span] = span
        guard = 0
        while cur is not None and guard < 100:   # 防环（脏数据）
            chain.append(cur.name)
            pid = cur.parent_id
            cur = by_id.get(pid) if pid else None
            guard += 1
        return " > ".join(reversed(chain))

    def _node_dict(self, span: Span, total: float) -> Dict[str, Any]:
        d = span.to_dict()
        d["duration_ms"] = span.duration_ms
        d["self_ms"] = self.self_ms(span)
        d["pct"] = round(d["duration_ms"] / total * 100.0,
                         1) if total > 0 else 0.0
        return d

    def tree(self) -> List[Dict[str, Any]]:
        """嵌套树。每个节点额外带 duration_ms / self_ms / pct / children /
        has_error / depth，前端画甘特条与标红都靠它们。

        ★ 孤儿 span（父亲不在数据里，落盘写了一半的常见形态）
          会被当成根节点渲染，而不是丢弃或抛异常。
        """
        total = self.total_ms()
        nodes = {s.span_id: self._node_dict(s, total) for s in self._spans}
        for node in nodes.values():
            node["children"] = []
            node["has_error"] = node["status"] in _BAD_STATUS
            node["depth"] = 0

        roots: List[Dict[str, Any]] = []
        for s in self._spans:
            node = nodes[s.span_id]
            pid = s.parent_id
            if pid and pid in nodes and pid != s.span_id:
                nodes[pid]["children"].append(node)
            else:
                roots.append(node)

        def _decorate(node: Dict[str, Any], depth: int) -> None:
            node["depth"] = depth
            for kid in node["children"]:
                _decorate(kid, depth + 1)
                node["has_error"] = node["has_error"] or kid["has_error"]

        for r in roots:
            _decorate(r, 0)
        return roots

    def bottlenecks(self, n: int = 3) -> List[Dict[str, Any]]:
        """自身耗时 Top-N。**排除根 span**（理由见测试 O4）。"""
        total = self.total_ms()
        items = []
        for s in self._spans:
            if s.kind == KIND_RUN:
                continue
            d = self._node_dict(s, total)
            d["self_pct"] = round(d["self_ms"] / total * 100.0,
                                  1) if total > 0 else 0.0
            d["path"] = self.path_of(s)
            items.append(d)
        items.sort(key=lambda x: (-x["self_ms"], x["name"]))
        return items[:n]

    def errors(self) -> List[Dict[str, Any]]:
        """所有非 ok 的 span，带完整路径。面板"错误节点"列表直接用。"""
        out = []
        for s in sorted(self._spans, key=lambda x: x.t0):
            if s.status in _BAD_STATUS:
                d = self._node_dict(s, self.total_ms())
                d["path"] = self.path_of(s)
                out.append(d)
        return out

    # ------------------------------------------------------------ 渲染
    def render_gantt(self, width: int = 44) -> str:
        """ASCII 甘特条。给 demo / CI / 终端用（前端有自己的 SVG 版）。

        输出形态::

            name                |####      |  80.0ms ( 20.0%)
              tool:kb_search    |   ####   | 300.0ms ( 75.0%)
        """
        if not self._spans:
            return "(无 span)"
        t_start = min(s.t0 for s in self._spans)
        total = self.total_ms() or 1.0
        # ★ 宽度要把缩进算进去，否则深层 span（名字更长）会把条挤歪
        name_w = max(len(s.name) + 2 * self._depth_of(s) for s in self._spans)
        lines: List[str] = []
        for s in sorted(self._spans, key=lambda x: (x.t0, x.span_id)):
            depth = self._depth_of(s)
            indent = "  " * depth
            offset = int((s.t0 - t_start) / (total / 1000.0) * width
                         ) if total > 0 else 0
            bar_len = max(int(s.duration_ms / total * width), 1)
            offset = min(offset, max(width - 1, 0))
            bar_len = min(bar_len, max(width - offset, 1))
            bar = " " * offset + "#" * bar_len
            bar = (bar + " " * width)[:width]
            mark = {STATUS_ERROR: "!", STATUS_BLOCKED: "x",
                    STATUS_LIMIT: "~"}.get(s.status, " ")
            pct = s.duration_ms / total * 100.0
            lines.append("%s%-*s |%s|%s %7.1fms (%5.1f%%)" % (
                indent, name_w, s.name, bar, mark, s.duration_ms, pct))
        return "\n".join(lines)

    def _depth_of(self, span: Span) -> int:
        by_id = {s.span_id: s for s in self._spans}
        depth = 0
        cur = span
        guard = 0
        while cur.parent_id and guard < 100:
            nxt = by_id.get(cur.parent_id)
            if nxt is None:
                break
            cur = nxt
            depth += 1
            guard += 1
        return depth


def gantt_of(dicts: List[Dict[str, Any]], width: int = 44) -> str:
    """把落盘的 span dict 列表直接渲染成甘特条（回放时用，无需重建引擎）。"""
    tr = SpanTracer()
    tr.load_dicts(dicts)
    return tr.render_gantt(width=width)


def analyze(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """落盘数据 -> 面板要的一切：树 + 瓶颈 + 错误节点 + 总耗时。"""
    tr = SpanTracer()
    tr.load_dicts(dicts)
    total = tr.total_ms()
    return {
        "total_ms": total,
        "span_count": len(tr.spans),
        "tree": tr.tree(),
        "bottlenecks": tr.bottlenecks(3),
        "errors": tr.errors(),
        "gantt": tr.render_gantt(),
    }
