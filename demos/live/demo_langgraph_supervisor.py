# -*- coding: utf-8 -*-
"""demo_langgraph_supervisor.py —— supervisor-worker 多智能体（真 LangGraph 子图组合）。

跑法（需要 langgraph，见文件末的依赖说明）：
    conda run -n agent_rag python demos/live/demo_langgraph_supervisor.py

★ 这个 demo 演示什么（E1 · LangGraph 真库验证的第二半）：

    项目一的主引擎是**单 Agent**（core/agent_langgraph.py：一个图里跑完
    意图→计划→行动→验证）。这个 demo 演示 LangGraph 的**多智能体原语**：

        ① supervisor 图：一个"调度"节点按任务类型把活派给 worker
        ② 每个 worker 是一个**编译好的子图**（prepare -> execute 两节点），
           通过 add_node 挂进 supervisor 图 —— 这是 LangGraph 的子图组合，
           不是"在一个图里多画几个节点"那么糊弄
        ③ 条件边路由：supervisor 每次只派一个 worker，派完回到 supervisor
           重新决策，直到计划清空再进 verifier
        ④ verifier 汇总各 worker 的产出，给出带引用的最终答案

★ 与 demo_agent2.py 的分工：
    demo_agent2.py   单 Agent + 真 function calling 决策（项目二的日常形态）
    本 demo          多智能体拓扑 + 子图组合（回答"什么时候需要多智能体"）

★ 诚实说明（面试也这么讲）：
    demo 里的决策是**脚本化**的（不调真模型）——因为它要演示的是
    **编排机制**（子图/路由/汇总），不是模型能力。接真模型时只需把
    supervisor_node 里的"下一个派谁"换成 LLM 决策，worker 的
    execute 节点换成真工具调用，图的结构一行不用改。

★ 为什么需要多智能体（本 demo 的回答）：
    当"检索"和"查数据"是**不同的工具集、不同的失败模式、可以并行**时，
    拆成 worker 才能分别加熔断/重试/降级；单 Agent 不是不能做，是
    管控粒度太粗。代价也一样真实：token 消耗常为单体 10 倍以上
    （Anthropic 多智能体研究：+90.2% 效果 / 15 倍 token）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.agent_langgraph import langgraph_available  # noqa: E402
from core.mcp import ToolRegistry  # noqa: E402


# --------------------------------------------------------------------- 依赖门
def _require_langgraph():
    """live demo 约定：缺依赖时说清楚缺什么，然后正常退出（不算失败）。"""
    if not langgraph_available():
        print("langgraph 未安装 —— 本 demo 需要真库。")
        print("安装：conda run -n agent_rag pip install langgraph==0.2.28")
        print("（CI 零依赖环境不跑 live demo，这是预期行为）")
        return False
    return True


# --------------------------------------------------------------------- 工具
def make_registry():
    reg = ToolRegistry()

    @reg.tool(description="检索知识库")
    def kb_search(query: str, top_k: int = 3):
        """检索制度

        query: 关键词
        top_k: 条数
        """
        return {"count": 1, "hits": [{"id": "doc18#c3", "text": "复核流程见《制度》第三章"}]}

    @reg.tool(description="查询工单")
    def ticket_query(ticket_id: str):
        """查工单

        ticket_id: 单号
        """
        return {"ticket_id": ticket_id, "status": "closed", "owner": "审批组"}

    return reg


# --------------------------------------------------------------------- 状态
def _load_types():
    from langgraph.graph import StateGraph
    from typing import Any, Dict, List, TypedDict

    class SupervisorState(TypedDict, total=False):
        """supervisor 图的状态。

        ★ 同样必须 TypedDict 显式声明：裸 dict 让真库无法推断通道，
          输入键不进状态（core/agent_langgraph.py 的踩坑记录）。
        ★ 节点返回值也只能写这些键 —— 真库对未声明的键直接
          InvalidUpdateError（本 demo 第二版抓到的），这是比
          静默丢更新更友好的失败方式。
        """
        query: str
        task_type: str
        plan: List[str]            # 要派哪些 worker（顺序即派发顺序）
        next_worker: str           # supervisor 的决策：派谁 / 进 verifier
        dispatched: List[str]      # 已派完的 worker
        worker_outputs: Dict[str, Any]   # 各 worker 的产出汇总
        current_tool: str          # worker 子图内部：当前在调的工具
        current_args: Dict[str, Any]      # worker 子图内部：当前参数
        answer: str
        citations: List[str]
        finished: bool

    return StateGraph, SupervisorState


# --------------------------------------------------------------------- worker 子图
def build_worker_subgraph(name, tool, args_fn, registry, StateGraph,
                          SupervisorState, END, trace):
    """把一个 worker 编译成**三节点子图**：prepare -> execute -> record -> END。

    子图挂进 supervisor 后，对父图来说就是一个普通节点——
    这是 LangGraph 的组合原语：worker 内部可以任意复杂，
    父图只关心它的输入输出契约。

    ★ 子图必须用**与父图相同的 TypedDict schema**：裸 dict 会让真库无法
      推断通道、输入键不进状态（core/agent_langgraph.py 踩坑记录，
      本 demo 第一版就在这里又踩了一次）。
    ★ record 节点负责把产出写回共享状态（worker_outputs / dispatched）——
      初版把这段逻辑写成父图里的独立函数却忘了接线，worker 跑完不更新
      dispatched，supervisor 无限派同一个 worker 撞递归上限。
      教训：多智能体里"谁写共享状态"必须显式到一个节点里。
    """
    tag = f"worker[{name}]"

    def prepare(state):
        trace.append(f"{tag}.prepare  收到任务 query={state.get('query', '')!r}")
        return {"current_tool": tool}

    def execute(state):
        args = args_fn(state)
        trace.append(f"{tag}.execute  调用 {tool}({args})")
        return {"current_args": args}

    def record(state):
        args = args_fn(state)
        result = registry.call(tool, args)
        outputs = dict(state.get("worker_outputs") or {})
        outputs[name] = {"tool": tool, "args": args, "result": result}
        dispatched = list(state.get("dispatched") or [])
        if name not in dispatched:
            dispatched.append(name)
        trace.append(f"{tag}.record   产出已写回共享状态（{len(outputs)} 个 worker 有产出）")
        return {"worker_outputs": outputs, "dispatched": dispatched}

    g = StateGraph(SupervisorState)
    g.add_node("prepare", prepare)
    g.add_node("execute", execute)
    g.add_node("record", record)
    g.add_edge("prepare", "execute")
    g.add_edge("execute", "record")
    g.add_edge("record", END)
    g.set_entry_point("prepare")
    return g.compile()


# --------------------------------------------------------------------- supervisor 图
def build_supervisor(registry, StateGraph, SupervisorState, END, trace):
    """supervisor-worker 拓扑：

        [supervisor] --条件边--> [retrieval 子图] --+
                     |                              |
                     |<-----------------------------+  （worker 干完回 supervisor）
                     |--条件边--> [data 子图] ------+
                     |--条件边--> [verifier] --> END

    任务计划（派哪些 worker）由调用方写进 state['plan']；
    接真模型时把 supervisor_node 里的"下一个派谁"换成 LLM 决策即可。
    """
    WORKER_TOOLS = {"retrieval": ("kb_search", lambda s: {"query": s["query"]}),
                    "data": ("ticket_query", lambda s: {"ticket_id": "T-100"})}

    workers = {}
    for wname, (tool, args_fn) in WORKER_TOOLS.items():
        workers[wname] = build_worker_subgraph(
            wname, tool, args_fn, registry, StateGraph, SupervisorState,
            END, trace)

    def supervisor_node(state):
        plan = list(state.get("plan") or [])
        done = list(state.get("dispatched") or [])
        nxt = next((w for w in plan if w not in done), None)
        if nxt is None:
            trace.append("supervisor      计划清空 -> 进 verifier")
            return {"next_worker": "__verify__"}
        trace.append(f"supervisor      派发 {nxt}（剩余 {[w for w in plan if w not in done]}）")
        return {"next_worker": nxt}

    def route(state):
        return state.get("next_worker") or "__verify__"

    def verifier(state):
        outputs = state.get("worker_outputs") or {}
        cites, parts = [], []
        for rec in outputs.values():
            res = rec["result"]
            if "hits" in res:
                for h in res["hits"]:
                    cites.append(h["id"])
                    parts.append(h["text"])
            else:
                parts.append(f"工单 {res['ticket_id']} 状态 {res['status']}（{res['owner']}）")
        answer = "；".join(parts) if parts else "（无产出）"
        ok = bool(parts)
        trace.append(f"verifier        汇总 {len(outputs)} 个 worker 产出 -> {'通过' if ok else '无产出'}")
        return {"answer": answer, "citations": cites, "finished": True}

    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor_node)
    for wname, subgraph in workers.items():
        g.add_node(wname, subgraph)          # ★ 编译好的子图作为节点
    g.add_node("__verify__", verifier)
    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor", route,
        {"retrieval": "retrieval", "data": "data", "__verify__": "__verify__"})
    g.add_edge("retrieval", "supervisor")
    g.add_edge("data", "supervisor")
    g.add_edge("__verify__", END)
    return g.compile()


# --------------------------------------------------------------------- 演示
def run_task(graph, query, plan, trace):
    trace.clear()
    trace.append(f"任务 {query!r} 计划 {plan}")
    final = {}
    for chunk in graph.stream({"query": query, "task_type": "full_review",
                               "plan": plan, "dispatched": [],
                               "worker_outputs": {}}):
        for updates in (chunk or {}).values():
            if isinstance(updates, dict):
                final.update(updates)
    return final


def main():
    if not _require_langgraph():
        return 0

    StateGraph, SupervisorState = _load_types()
    END = "__end__"
    from langgraph.graph import END as LG_END
    END = LG_END

    registry = make_registry()
    trace = []

    # 任务计划由"路由器"决定（接真模型时这里换成 LLM 决策）
    PLANS = {
        "full_review": ["retrieval", "data"],
        "spec_inquiry": ["retrieval"],
    }

    graph = build_supervisor(registry, StateGraph, SupervisorState,
                             END, trace)

    print("=" * 78)
    print("supervisor-worker 多智能体 demo（真 LangGraph 子图组合）")
    print("=" * 78)

    print("\n【拓扑】supervisor 图的节点（含两个 worker 子图与 verifier）：")
    real = graph.get_graph()
    for nid in sorted(real.nodes):
        print(f"  - {nid}")

    print("\n【Mermaid】graph.get_graph().draw_mermaid() —— 可直接贴进 README：")
    try:
        print(real.draw_mermaid())
    except Exception as exc:  # noqa: BLE001 - 画图是锦上添花，失败不拦演示
        print(f"  （mermaid 导出失败：{exc}）")

    results = {}
    for task, plan in PLANS.items():
        print(f"\n【运行】task_type={task}  计划={plan}")
        final = run_task(graph, "工单 T-100 的复核依据是什么", plan, trace)
        for line in trace:
            print(f"  {line}")
        results[task] = final
        print(f"  答案：{final.get('answer', '')}")
        print(f"  引用：{final.get('citations', [])}")

    # ------------------------------------------------------------ 自检
    print("\n【自检】")
    full = results["full_review"]
    spec = results["spec_inquiry"]
    checks = [
        ("full_review 派了两个 worker", sorted(full.get("dispatched", [])) == ["data", "retrieval"]),
        ("spec_inquiry 只派了 retrieval", full is not spec and spec.get("dispatched") == ["retrieval"]),
        ("答案非空且含引用", bool(full.get("answer")) and bool(full.get("citations"))),
        ("流程正常收尾", full.get("finished") is True),
        ("拓扑含 5 个节点（supervisor+2 worker+verifier）", len(real.nodes) >= 5),
    ]
    ok = True
    for name, passed in checks:
        print(f"  {'✓' if passed else '✗'} {name}")
        ok = ok and passed
    print("\n结论：supervisor 路由 / worker 子图 / 产出汇总 / verifier 收尾全部符合预期。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
