# -*- coding: utf-8 -*-
"""core/agent_langgraph.py —— 用真 LangGraph 实现的 Agent 引擎（与 AgentEngine 可互换）。

★ 为什么要有这个文件（而不是"干脆全换成 langgraph"）：

    简历上写的是「基于 LangGraph 的受控 Agent 工作流」。
    而 `core/agent.py` 是**自实现的等价状态机** —— 行为对齐 LangGraph 语义，
    但不是 LangGraph 本身。面试官问一句「你的 StateGraph 怎么编译的？
    Checkpointer 用 SqliteSaver 还是 MemorySaver？」就能戳穿。

    所以这里用**真 langgraph** 重写编排层，同时：

      ① 接口与 `AgentEngine` **完全一致**（同样的构造参数、同样的 run/resume 签名、
         同样的 AgentState / AgentRun / EventType 返回类型）
      ② 业务逻辑**零改动** —— 意图识别、计划、工具白名单、熔断、记忆全都不变，
         只是"谁来驱动这些步骤"从自写 while 循环换成 langgraph 的 StateGraph
      ③ 可用 `make_agent_engine(prefer_langgraph=True/False)` 切换，
         装不上 langgraph 或行为不一致时**一行回退**

★ 诚实说明（面试也应如实讲）：

    LangGraph 在这里的收益是**可观测性与可恢复性的标准化**：
      · StateGraph 把"节点 + 条件边"显式化，拓扑可画出来（`graph.get_graph()`）
      · Checkpointer 由框架统一管理，续跑语义有明确定义
    它的代价是**引入了依赖与抽象层**：调试时要穿透框架，
    而自实现版只有 376 行、每一步都可见。

    "为什么不用 LangGraph" 与 "为什么用 LangGraph" 都是合理的工程选择，
    关键是你能说清取舍。这个文件让你**两种都能演示**。
"""
from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .agent import (
    DEFAULT_MAX_ROUNDS, DEFAULT_MAX_TOOL_CALLS, DEFAULT_TOTAL_TIMEOUT_S,
    DEFAULT_TOOL_TIMEOUT_S, AgentRun, AgentState, Checkpointer, EventType,
    make_event,
)
from .mcp import ToolRegistry


# --------------------------------------------------------------------- 依赖探测
def langgraph_available() -> bool:
    """langgraph 是否可用（延迟探测，不 import 时不报错）。"""
    try:
        importlib.import_module("langgraph.graph")
        return True
    except ImportError:
        return False


def _load_stategraph() -> Any:
    """取到 langgraph 的 StateGraph 与 END 常量。"""
    mod = importlib.import_module("langgraph.graph")
    return mod.StateGraph, mod.END


# --------------------------------------------------------------------- 引擎
class LangGraphAgentEngine:
    """与 `AgentEngine` 接口完全一致的 LangGraph 版实现。

    节点划分（对应简历里的"受控 Agent 工作流"）：

        [intent]  意图识别 + 白名单解析
            |
        [plan]    生成子任务计划
            |
        [act] ←──────────┐   模型决策：调工具 或 直接给答案
            |             |
        [tools] ─────────┘   执行工具（含白名单硬校验）
            |
        [verify]  校验答案
            |
        [finalize] 收尾（熔断原因 / 事件）

    条件边：
        act --(有 final)--> verify
        act --(要调工具)--> tools
        act --(熔断)------> finalize
        verify --(通过)----> finalize
        verify --(不通过)--> act
    """

    # 节点名（同时作为 langgraph 的 node id）
    N_INTENT = "intent"
    N_PLAN = "plan"
    N_ACT = "act"
    N_TOOLS = "tools"
    N_VERIFY = "verify"
    N_FINALIZE = "finalize"

    def __init__(
        self,
        registry: ToolRegistry,
        intent_fn: Callable[[str], str],
        plan_fn: Callable[[str, str], List[str]],
        act_fn: Callable[[AgentState, List[str]], Dict[str, Any]],
        verify_fn: Optional[Callable[[AgentState, str], Tuple[bool, str]]] = None,
        route_table: Optional[Dict[str, Sequence[str]]] = None,
        checkpointer: Optional[Checkpointer] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        total_timeout_s: float = DEFAULT_TOTAL_TIMEOUT_S,
        tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
        clock: Callable[[], float] = time.time,
        graph_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.registry = registry
        self.intent_fn = intent_fn
        self.plan_fn = plan_fn
        self.act_fn = act_fn
        self.verify_fn = verify_fn or (lambda s, a: (bool(a.strip()), "非空回答"))
        self.route_table = dict(route_table or {})
        self.checkpointer = checkpointer
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.total_timeout_s = total_timeout_s
        self.tool_timeout_s = tool_timeout_s
        self._clock = clock
        # 允许注入自定义 StateGraph（测试用假实现；真机传 None 走 langgraph）
        self._graph_factory = graph_factory
        self._intent_calls = 0

        self._events: List[Dict[str, Any]] = []
        self._graph = self._build_graph()

    # ------------------------------------------------------------ 白名单
    def route(self, task_type: str) -> List[str]:
        """意图 -> 允许的工具集。语义与 AgentEngine.route 完全一致。"""
        return list(self.route_table.get(task_type, []))

    # ------------------------------------------------------------ 事件
    def _emit(self, etype: str, **data: Any) -> None:
        self._events.append(make_event(etype, **data))

    def _limit_reached(self, state: AgentState) -> Optional[str]:
        """三道熔断 + 超时。返回停止原因（None 表示还能继续）。"""
        if state.rounds >= self.max_rounds:
            return "max_rounds"
        if state.tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if state.elapsed_s(self._clock) > self.total_timeout_s:
            return "total_timeout"
        return None

    def detect_loop(self, state: AgentState, window: int = 3) -> bool:
        """死循环检测：最近 window 轮的工具调用签名完全相同则判定为循环。"""
        sigs = [s.get("sig") for s in state.scratchpad
                if s.get("type") == "tool_call"]
        sigs = [s for s in sigs if s]
        if len(sigs) < window:
            return False
        tail = sigs[-window:]
        return len(set(tail)) == 1

    # ------------------------------------------------------------ 建图
    def _build_graph(self) -> Any:
        """构建 StateGraph —— ★ 这是"基于 LangGraph"的实质所在。"""
        if self._graph_factory is not None:
            StateGraph, END = self._graph_factory(), None
            # 假实现自带 END 常量
            if StateGraph is not None and hasattr(StateGraph, "END"):
                END = StateGraph.END
        else:
            StateGraph, END = _load_stategraph()

        builder = StateGraph(dict)
        builder.add_node(self.N_INTENT, self._node_intent)
        builder.add_node(self.N_PLAN, self._node_plan)
        builder.add_node(self.N_ACT, self._node_act)
        builder.add_node(self.N_TOOLS, self._node_tools)
        builder.add_node(self.N_VERIFY, self._node_verify)
        builder.add_node(self.N_FINALIZE, self._node_finalize)

        builder.set_entry_point(self.N_INTENT)
        builder.add_edge(self.N_INTENT, self.N_PLAN)
        builder.add_edge(self.N_PLAN, self.N_ACT)

        # act 之后的分支：要答案就 verify，要工具就 tools，熔断就收尾
        builder.add_conditional_edges(
            self.N_ACT,
            self._route_after_act,
            {self.N_TOOLS: self.N_TOOLS,
             self.N_VERIFY: self.N_VERIFY,
             self.N_FINALIZE: self.N_FINALIZE},
        )
        builder.add_edge(self.N_TOOLS, self.N_ACT)

        # verify 之后：通过就收尾，不通过回到 act 再试
        builder.add_conditional_edges(
            self.N_VERIFY,
            self._route_after_verify,
            {self.N_FINALIZE: self.N_FINALIZE,
             self.N_ACT: self.N_ACT},
        )
        builder.add_edge(self.N_FINALIZE, END)
        return builder.compile()

    # ------------------------------------------------------------ 条件路由
    def _route_after_act(self, state: Dict[str, Any]) -> str:
        if state.get("stop_reason"):
            return self.N_FINALIZE
        if state.get("answer"):
            return self.N_VERIFY
        return self.N_TOOLS

    def _route_after_verify(self, state: Dict[str, Any]) -> str:
        if state.get("stop_reason"):
            return self.N_FINALIZE
        if state.get("verified") or state.get("answer"):
            return self.N_FINALIZE
        return self.N_ACT

    # ------------------------------------------------------------ 节点实现
    def _node_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._intent_calls += 1
        query = state["query"]
        ttype = self.intent_fn(query)
        allowed = self.route(ttype)
        self._emit(EventType.INTENT, task_type=ttype, allowed_tools=allowed)
        return {"task_type": ttype, "allowed_tools": allowed}

    def _node_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = self.plan_fn(state["query"], state.get("task_type", "general"))
        self._emit(EventType.PLAN, steps=plan)
        return {"plan": list(plan)}

    def _node_act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """模型决策节点：调工具 / 给答案 / 触发熔断。

        ★ 节点必须是**纯函数**：只读 state，只返回"更新"，绝不原地修改传入的 dict。
          踩坑记录：初版这里写了 `state["rounds"] = state.get(...) + 1`，
          既原地改了输入、又从来不在返回里带上 rounds —— 于是轮次永远停在 0，
           `max_rounds` 熔断、`loop_detected`、时间超时**全部失效**，
           `_pending` 也因为只写在原地而丢失，导致 tools 节点什么都取不到。
          18 个测试同时红就是这个原因。
          教训：图的节点契约是「in -> updates」，不是「in-place mutation」。
          真实 langgraph 同样会把节点返回值当作状态更新合并 —— 原地修改要么被丢、
          要么在并行分支下产生难以复现的竞态。
        """
        if state.get("finished"):
            return {}

        cur = self._as_state(state)

        limit = self._limit_reached(cur)
        if limit:
            self._emit(EventType.LIMIT, reason=limit, rounds=cur.rounds,
                       tool_calls=cur.tool_calls)
            return {"stop_reason": limit}

        if self.detect_loop(cur):
            self._emit(EventType.LIMIT, reason="loop_detected", rounds=cur.rounds)
            return {"stop_reason": "loop_detected"}

        decision = self.act_fn(cur, list(state.get("allowed_tools", [])))
        decision = decision or {}
        rounds = cur.rounds + 1

        if decision.get("final") is not None:
            answer = str(decision.get("final") or "")
            cites = list(decision.get("citations") or [])
            self._emit(EventType.ANSWER, text=answer[:200], citations=cites)
            return {"answer": answer, "citations": cites, "rounds": rounds,
                    "_pending": None}

        if decision.get("error"):
            self._emit(EventType.ERROR, message=str(decision["error"]))
            return {"answer": str(decision.get("final") or ""), "rounds": rounds,
                    "stop_reason": "act_error", "_pending": None}

        self._emit(EventType.TOOL_CALL, tool=decision.get("tool"),
                   args=decision.get("args") or {})
        return {"rounds": rounds, "_pending": decision}

    def _node_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """工具执行节点：白名单硬校验 + 调用 + 记录。纯函数，只返回更新。"""
        decision = state.get("_pending") or {}
        tool = decision.get("tool")
        args = decision.get("args") or {}
        allowed = list(state.get("allowed_tools", []))
        scratchpad = list(state.get("scratchpad", []))
        tool_calls = int(state.get("tool_calls", 0))

        if tool not in allowed:
            # ★ 白名单是硬约束：不在清单里直接拒绝，且**不消耗**工具预算
            scratchpad.append({"type": "tool_call", "tool": tool, "args": args,
                               "blocked": True, "ok": False,
                               "sig": f"{tool}:blocked"})
            self._emit(EventType.TOOL_ERROR, tool=tool, blocked=True,
                       message=f"工具 {tool} 不在当前意图的允许清单 {allowed} 内")
            return {"scratchpad": scratchpad, "_pending": None}

        if tool_calls >= self.max_tool_calls:
            self._emit(EventType.LIMIT, reason="max_tool_calls")
            return {"stop_reason": "max_tool_calls", "_pending": None}

        try:
            result = self.registry.call(tool, args)
            scratchpad.append({
                "type": "tool_call", "tool": tool, "args": args, "ok": True,
                "result": result,
                # 死循环检测用的签名：同样的工具 + 同样的参数 = 一样的 sig
                "sig": f"{tool}:{sorted((str(k), str(v)) for k, v in args.items())}",
            })
            self._emit(EventType.TOOL_RESULT, tool=tool, ok=True)
            return {"scratchpad": scratchpad, "_pending": None,
                    "tool_calls": tool_calls + 1, "last_observation": result}
        except Exception as exc:  # noqa: BLE001 - 工具异常不该让整轮崩掉
            scratchpad.append({
                "type": "tool_call", "tool": tool, "args": args, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "sig": f"{tool}:error",
            })
            self._emit(EventType.TOOL_ERROR, tool=tool, blocked=False,
                       message=str(exc))
            return {"scratchpad": scratchpad, "_pending": None,
                    "tool_calls": tool_calls + 1}

    def _node_verify(self, state: Dict[str, Any]) -> Dict[str, Any]:
        answer = state.get("answer") or ""
        ok, reason = self.verify_fn(self._as_state(state), answer)
        self._emit(EventType.VERIFY, ok=ok, reason=reason)
        updates: Dict[str, Any] = {"verified": bool(ok), "verify_reason": reason}
        if not ok and int(state.get("rounds", 0)) < self.max_rounds:
            # 不通过则清空答案回到 act（让模型再试一次）
            updates["answer"] = ""
        return updates

    def _node_finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        answer = state.get("answer") or ""
        if not answer and state.get("stop_reason"):
            answer = f"（已达到停止条件：{state['stop_reason']}）"
        self._emit(EventType.DONE, reason=state.get("stop_reason") or "verified",
                   rounds=state.get("rounds", 0),
                   tool_calls=state.get("tool_calls", 0))
        return {"finished": True, "answer": answer}

    # ------------------------------------------------------------ 辅助
    def _as_state(self, d: Dict[str, Any]) -> AgentState:
        """把 langgraph 的 dict state 还原成 AgentState（供注入函数使用）。

        ★ 为什么需要这一层：`act_fn` / `verify_fn` 的签名是 `(AgentState, ...)`，
          这是与自实现版保持接口一致的代价。用轻量适配而不是改注入函数签名，
          好处是**同一套业务函数在两版引擎下都能跑**。
        """
        s = AgentState(
            query=d.get("query", ""),
            thread_id=d.get("thread_id", ""),
            task_type=d.get("task_type", "general"),
            allowed_tools=list(d.get("allowed_tools", [])),
            plan=list(d.get("plan", [])),
            rounds=int(d.get("rounds", 0)),
            tool_calls=int(d.get("tool_calls", 0)),
            started_at=float(d.get("started_at", self._clock())),
            answer=d.get("answer", "") or "",
            citations=list(d.get("citations", [])),
            verified=bool(d.get("verified")),
            verify_reason=d.get("verify_reason", "") or "",
            finished=bool(d.get("finished")),
            stop_reason=d.get("stop_reason", "") or "",
            last_observation=d.get("last_observation"),
        )
        s.scratchpad = list(d.get("scratchpad", []))
        return s

    # ------------------------------------------------------------ 主流程
    def run(
        self,
        query: str,
        thread_id: Optional[str] = None,
        resume: bool = False,
    ) -> AgentRun:
        tid = thread_id or f"t{int(self._clock() * 1000)}"
        self._events = []
        self._emit(EventType.START, query=query, thread_id=tid,
                   engine="langgraph")

        init: Dict[str, Any] = {
            "query": query,
            "thread_id": tid,
            "task_type": "general",
            "allowed_tools": [],
            "plan": [],
            "scratchpad": [],
            "rounds": 0,
            "tool_calls": 0,
            "started_at": self._clock(),
            "answer": "",
            "citations": [],
            "verified": False,
            "verify_reason": "",
            "refused": False,
            "finished": False,
            "stop_reason": "",
            "last_observation": None,
            "_pending": None,
        }

        # 续跑：把检查点状态灌回初始状态
        if resume and self.checkpointer is not None:
            saved = self.checkpointer.load(tid)
            if saved is not None:
                init.update(saved.to_dict())
                self._emit(EventType.RESUME, thread_id=tid, rounds=saved.rounds,
                           tool_calls=saved.tool_calls, finished=saved.finished)
                if saved.finished:
                    self._emit(EventType.DONE, reason="已完成的检查点，无需续跑")
                    return AgentRun(state=saved, events=self._events)

        final: Dict[str, Any] = dict(init)
        try:
            if hasattr(self._graph, "stream"):
                # ★ 用 stream 而不是 invoke：能在每个节点后落 checkpoint，
                #   这才是"可恢复"的前提（invoke 只在结束时返回）。
                for chunk in self._graph.stream(init):
                    for _node, updates in (chunk or {}).items():
                        if isinstance(updates, dict):
                            final.update(updates)
                    self._save(self._as_state(final))
            else:
                final.update(self._graph.invoke(init) or {})
        except Exception as exc:  # noqa: BLE001
            self._emit(EventType.ERROR, message=f"图执行失败: {exc}")
            final["stop_reason"] = "graph_error"

        state = self._as_state(final)
        self._save(state)
        if not state.finished:
            state.finished = True
        return AgentRun(state=state, events=self._events)

    def resume(self, thread_id: str) -> AgentRun:
        return self.run("", thread_id=thread_id, resume=True)

    def _save(self, state: AgentState) -> None:
        if self.checkpointer is not None:
            self.checkpointer.save(state.thread_id, state)

    # ------------------------------------------------------------ 自省
    def graph_nodes(self) -> List[str]:
        """列出图里的节点（面试时可以直接演示拓扑）。"""
        try:
            g = self._graph.get_graph()
            return sorted(n.id for n in g.nodes.values()) if hasattr(g, "nodes") else []
        except Exception:  # noqa: BLE001
            return [self.N_INTENT, self.N_PLAN, self.N_ACT,
                    self.N_TOOLS, self.N_VERIFY, self.N_FINALIZE]

    def describe(self) -> Dict[str, Any]:
        """返回引擎自述（用于 health 接口与面试演示）。"""
        return {
            "engine": "langgraph",
            "langgraph_available": langgraph_available(),
            "nodes": [self.N_INTENT, self.N_PLAN, self.N_ACT,
                      self.N_TOOLS, self.N_VERIFY, self.N_FINALIZE],
            "conditional_edges": [
                f"{self.N_ACT} -> ({self.N_TOOLS} | {self.N_VERIFY} | {self.N_FINALIZE})",
                f"{self.N_VERIFY} -> ({self.N_FINALIZE} | {self.N_ACT})",
            ],
            "fuses": {"max_rounds": self.max_rounds,
                      "max_tool_calls": self.max_tool_calls,
                      "total_timeout_s": self.total_timeout_s},
            "route_table": {k: list(v) for k, v in self.route_table.items()},
        }


# --------------------------------------------------------------------- 工厂
def make_agent_engine(
    prefer_langgraph: bool = True,
    **kwargs: Any,
) -> Any:
    """按环境自动选择实现，**保证调用方代码完全相同**。

    Args:
        prefer_langgraph: True 时若有 langgraph 就用 LangGraph 版，否则回退自实现版

    用法（业务代码不需要知道用的是哪一版）：

        engine = make_agent_engine(registry=..., intent_fn=..., ...)
        run = engine.run("问题")

    ★ 这个工厂的价值：它让"用不用框架"变成一个**运行期开关**，
      而不是一次架构决策。面试时你能同时演示两种实现并讲清取舍。
    """
    if prefer_langgraph and langgraph_available():
        return LangGraphAgentEngine(**kwargs)
    from .agent import AgentEngine
    return AgentEngine(**kwargs)
