# -*- coding: utf-8 -*-
"""demos/demo_agent_trace.py —— Agent 推理轨迹：实时流 / 落盘 / 回放 / SSE 帧化。

对应简历那句「SSE 实时输出 Agent 推理轨迹，支撑前端可视化面板快速定位错误节点」。
本 demo 把这句话拆成可验证的五个环节，每个都跑真代码、真断言：

    ① 轨迹逐事件产出      run_stream() 与 run() 事件序列完全一致
    ② 错误节点可观测      越权工具 -> tool_error(blocked=True)
    ③ 熔断可观测          max_rounds -> limit
    ④ 轨迹能落盘能回放    checkpointer 的 events.jsonl 往返一致
    ⑤ 事件能变成 SSE 帧   core/sse.stream_agent_events 的帧格式正确
    ⑥ 耗时归因            span 树 -> 甘特条 / 瓶颈 / 错误节点路径

★ 零依赖可跑：没有 langgraph 时 make_agent_engine 会自动回退到自实现版
  AgentEngine，两版 run_stream 的**事件内容与顺序一致**（差别只在到达时机：
  LangGraph 版节点级实时，回退版跑完再吐）。所以本 demo 在 CI 里也能跑。

用法：
    python demos/demo_agent_trace.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import (  # noqa: E402
    EventType, InMemoryCheckpointer, JsonFileCheckpointer,
)
from core.agent_langgraph import make_agent_engine  # noqa: E402
from core.mcp import ToolRegistry  # noqa: E402
from core.observability import analyze  # noqa: E402
from core.sse import parse_sse, stream_agent_events  # noqa: E402

CHECKS: list = []


def ok(name: str, cond: bool, extra: str = "") -> None:
    CHECKS.append((name, bool(cond)))
    print(("  [%s] %s" % ("OK" if cond else "!!", name)) + (
        "  " + extra if extra else ""))


def section(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(description="检索知识库")
    def kb_search(query: str, top_k: int = 3):
        """检索

        query: 关键词
        top_k: 条数
        """
        # ★ 演示用：真实检索（BGE-M3 嵌入 + 精排）确实要花上百毫秒，
        #   这里用 sleep 模拟，好让 ⑥ 的甘特条看得出"谁占了大头"。
        #   没有耗时的甘特条是一排等宽的条，什么都说明不了。
        time.sleep(0.12)
        return {"count": 1, "hits": [{"id": "制度#p1#c0", "text": "复核要求"}]}

    @reg.tool(description="查询工单")
    def ticket_query(ticket_id: str):
        """查工单

        ticket_id: 单号
        """
        return {"ticket_id": ticket_id, "status": "closed"}

    @reg.tool(description="发送邮件（高危，默认不在白名单）")
    def send_mail(to: str, subject: str = ""):
        """发邮件

        to: 收件人
        subject: 主题
        """
        return {"sent": True}

    return reg


ROUTE = {"spec_inquiry": ("kb_search",),
         "record_lookup": ("kb_search", "ticket_query"),
         "general": ()}


def make_engine(decisions, **kw):
    """脚本化决策的引擎：不接 LLM，决策完全可预测。"""
    cursor = {"i": 0}

    def act(state, allowed):
        d = decisions[min(cursor["i"], len(decisions) - 1)]
        cursor["i"] += 1
        return d(state, allowed) if callable(d) else d

    return make_agent_engine(
        registry=build_registry(),
        intent_fn=lambda q: "spec_inquiry" if "制度" in q else "general",
        plan_fn=lambda q, t: ["解析意图", "检索", "回答"],
        act_fn=act,
        route_table=ROUTE,
        **kw,
    )


def types(events):
    return [e["type"] for e in events]


def main() -> int:
    print("Agent 推理轨迹 —— 实时流 / 落盘 / 回放 / SSE 帧化")

    # ---------------------------------------------------------------- ①
    section("① 轨迹逐事件产出（run_stream 与 run 必须一致）")
    engine = make_engine([{"tool": "kb_search", "args": {"query": "复核"}},
                          {"final": "答案：须经二级复核", "citations": ["c1"]}])
    print("引擎实现：%s" % type(engine).__name__)
    streamed = list(engine.run_stream("制度里的复核要求", thread_id="T-DEMO"))
    sync = make_engine([{"tool": "kb_search", "args": {"query": "复核"}},
                        {"final": "答案：须经二级复核", "citations": ["c1"]}]
                       ).run("制度里的复核要求", thread_id="T-DEMO")
    print("  事件序列：" + " → ".join(types(streamed)))
    for i, ev in enumerate(streamed):
        payload = ev.get("data") or {}
        brief = ", ".join("%s=%s" % (k, str(v)[:40])
                          for k, v in list(payload.items())[:3])
        print("    %2d %-12s %s" % (i + 1, ev["type"], brief))
    ok("run_stream 与 run 事件序列一致",
       types(streamed) == types(sync.events))
    ok("轨迹覆盖 plan / tool_call / tool_result / answer / verify / done",
       all(t in types(streamed) for t in
           (EventType.PLAN, EventType.TOOL_CALL, EventType.TOOL_RESULT,
            EventType.ANSWER, EventType.VERIFY, EventType.DONE)))
    ok("last_state 可用（API 层推完事件要拿最终答案）",
       engine.last_state is not None and "复核" in engine.last_state.answer)

    # ---------------------------------------------------------------- ②
    section("② 错误节点可观测（越权工具被白名单拦截）")
    e2 = make_engine([{"tool": "send_mail", "args": {"to": "a@b.com"}},
                      {"final": "权限不足，无法执行"}])
    evs2 = list(e2.run_stream("发个邮件给老板", thread_id="T-ERR"))
    errs = [e for e in evs2 if e["type"] == EventType.TOOL_ERROR]
    print("  事件序列：" + " → ".join(types(evs2)))
    for e in errs:
        print("    错误节点：%s" % (e.get("data") or {}))
    ok("越权调用产生 tool_error", bool(errs))
    ok("错误节点标记 blocked=True（可与工具异常区分）",
       bool(errs) and errs[0]["data"].get("blocked"))

    # ---------------------------------------------------------------- ③
    section("③ 熔断可观测（max_rounds）")
    e3 = make_engine([{"tool": "kb_search", "args": {"query": "x"}}],
                     max_rounds=1)
    evs3 = list(e3.run_stream("制度要求是什么", thread_id="T-LIMIT"))
    limits = [e for e in evs3 if e["type"] == EventType.LIMIT]
    print("  事件序列：" + " → ".join(types(evs3)))
    for e in limits:
        print("    熔断节点：%s" % (e.get("data") or {}))
    ok("触发熔断并产生 limit 事件", bool(limits))

    # ---------------------------------------------------------------- ④
    section("④ 轨迹落盘与回放（面板「历史回放」的数据来源）")
    tmp = tempfile.mkdtemp(prefix="trace_demo_")
    cp = JsonFileCheckpointer(path=tmp)
    e4 = make_engine([{"tool": "kb_search", "args": {"query": "复核"}},
                      {"final": "答案 A"}], checkpointer=cp)
    ran = list(e4.run_stream("制度里的复核要求", thread_id="T-STORE"))
    stored = cp.load_events("T-STORE")
    print("  落盘文件：%s" % os.path.join(tmp, "T-STORE.events.jsonl"))
    print("  落盘 %d 条 / 实时流 %d 条" % (len(stored), len(ran)))
    ok("落盘条数与实时流一致", len(stored) == len(ran))
    ok("回放的事件类型序列一致", types(stored) == types(ran))
    ok("list_threads 不被轨迹文件污染",
       cp.list_threads() == ["T-STORE"], str(cp.list_threads()))
    e4b = make_engine([{"final": "续跑答案"}], checkpointer=cp)
    resumed = list(e4b.run_stream("制度里的复核要求", thread_id="T-STORE",
                                  resume=True))
    print("  续跑后共 %d 条（首次 %d + 续跑 %d）"
          % (len(cp.load_events("T-STORE")), len(ran), len(resumed)))
    ok("续跑是追加轨迹而非覆盖",
       len(cp.load_events("T-STORE")) >= len(ran))

    mem = InMemoryCheckpointer()
    e5 = make_engine([{"final": "内存版"}], checkpointer=mem)
    list(e5.run_stream("制度要求", thread_id="T-MEM"))
    ok("内存版 checkpointer 同样记录轨迹",
       len(mem.load_events("T-MEM")) > 0)

    # ---------------------------------------------------------------- ⑤
    section("⑤ 事件 -> SSE 帧（前端能直接消费）")
    frames = "".join(stream_agent_events(ran))
    parsed = parse_sse(frames)
    names = [p.get("event") for p in parsed]
    print("  帧数 %d，事件名：" % len(parsed))
    print("    " + " → ".join(str(n) for n in names))
    sample = parsed[0] if parsed else {}
    print("  首帧：event=%s  id=%s" % (sample.get("event"), sample.get("id")))
    ok("每个事件都有一帧 SSE", len(parsed) == len(ran))
    ok("帧的 event 名与事件类型一致", names == types(ran))
    ok("帧带递增 id（断线重连用）",
       [p.get("id") for p in parsed] == [str(i + 1) for i in range(len(parsed))])

    # ---------------------------------------------------------------- ⑥
    section("⑥ 耗时归因：span 树 -> 甘特条 / 瓶颈 / 错误节点")
    e6 = make_engine([{"tool": "kb_search", "args": {"query": "复核"}},
                      {"final": "答案 B"}], checkpointer=cp)
    events6 = list(e6.run_stream("制度里的复核要求", thread_id="T-SPAN"))
    info6 = analyze(e6.last_spans)
    print("  引擎：%s" % type(e6).__name__)
    print("  本次 %.1fms · %d 个 span"
          % (info6["total_ms"], info6["span_count"]))
    print()
    print(info6["gantt"])
    print()
    ok("拿到 span 树", info6["span_count"] > 0)
    ok("根 span 是 run（甘特条总时长就是它）",
       bool(info6["tree"]) and info6["tree"][0]["name"] == "run")
    tools = [s for s in e6.last_spans if s.get("kind") == "tool"]
    ok("工具调用有独立 span", len(tools) >= 1)
    ok("工具 span 有父节点（是嵌套树，不是平铺列表）",
       all(s.get("parent_id") for s in tools))
    top = info6["bottlenecks"][0] if info6["bottlenecks"] else None
    ok("最慢的是工具调用（归因有效，不是摆设）",
       top is not None and top["kind"] == "tool",
       ("%s %sms" % (top["name"], top["self_ms"])) if top else "")
    ok("span 不进事件流（事件序列仍然是纯事件）",
       all("span" not in str(e["type"]) for e in events6))
    ok("落盘回放的树与实时一致",
       analyze(cp.load_spans("T-SPAN"))["tree"] == info6["tree"])

    e6b = make_engine([{"tool": "send_mail", "args": {"to": "ops@x.com"}},
                       {"final": "兜底答案"}], checkpointer=cp)
    list(e6b.run_stream("制度要求", thread_id="T-BLOCK"))
    errs = analyze(e6b.last_spans)["errors"]
    print("  错误节点：" + ("；".join("%s（%s）" % (e["path"], e["status"])
                                     for e in errs) or "无"))
    ok("越权工具留下 blocked span",
       any(e["status"] == "blocked" for e in errs))
    ok("错误节点能给出完整路径（run > … > tool:xxx）",
       bool(errs) and all(e["path"].startswith("run") for e in errs))

    # ---------------------------------------------------------------- 汇总
    section("自检汇总")
    bad = [n for n, c in CHECKS if not c]
    print("  %d/%d 通过" % (len(CHECKS) - len(bad), len(CHECKS)))
    if bad:
        for n in bad:
            print("  未通过：" + n)
        return 1
    print("\n轨迹链路自证完毕：事件能实时产出、错误与熔断可观测、")
    print("能落盘能回放、能直接序列化成 SSE 帧推给前端面板。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
