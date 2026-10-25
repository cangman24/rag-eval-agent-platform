# -*- coding: utf-8 -*-
"""demo_agent.py —— Agent 四大工程机制演示（无需网络 / GPU / API Key）。

跑法：python demo_agent.py
输出：demo_agent_output.txt

演示五个场景：
  1. 多步任务全流程 + SSE 事件流（轨迹可观测）
  2. 工具白名单拦截越权调用
  3. Checkpoint 断点续跑（中断后已完成节点不重跑）
  4. 超时熔断（最大轮次 / 工具次数 / 死循环检测）
  5. 三级上下文压缩（阈值触发 + 工具结果截断）

★ 这里的"模型"是脚本化决策队列（act_fn），只用于验证**编排逻辑**。
  真机接 DeepSeek function calling 时，只需把 act_fn 换成"调 LLM 让它选工具"，
  引擎、白名单、checkpoint、熔断全部不用改。
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import (
    AgentEngine, EventType, JsonFileCheckpointer,
)
from core.bm25 import BM25Index
from core.mcp import ToolError, ToolRegistry
from core.memory import ContextCompressor, LayeredMemory, Message
from core.retrieval import recursive_chunk

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = []


def log(line=""):
    OUT.append(line)
    print(line)


def banner(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


# ================================================================== 语料与检索工具
CORPUS = """第一章 复核制度
第一条 高风险工单必须经人工复核后方可关闭。
第二条 复核人需具备三级及以上权限。
第二章 工单处理
第三条 工单关闭前需核对处置结果与证据材料。
第三章 邮件通知
第四条 涉及外部单位的通知邮件必须经主管审批后发送。
"""

chunks = recursive_chunk(CORPUS, chunk_size=64, overlap=8)
DOCS = {f"c{i}": f"{c['heading_path']} {c['text']}" for i, c in enumerate(chunks)}
bm25 = BM25Index()
for did, text in DOCS.items():
    bm25.add(did, text)
bm25.finalize()

TICKETS = {
    "T-100": {"ticket_id": "T-100", "level": "高风险", "status": "closed",
              "reviewed_by": None},
    "T-200": {"ticket_id": "T-200", "level": "低风险", "status": "closed",
              "reviewed_by": "张三"},
}
MAIL_LOG = []


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(description="检索知识库，返回与查询最相关的制度条款")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 检索关键词
        top_k: 返回条数
        """
        hits = bm25.search(query, top_k=top_k, min_score=0.5)
        return {
            "count": len(hits),
            "hits": [{"id": d, "heading_path": chunks[int(d[1:])]["heading_path"],
                      "text": chunks[int(d[1:])]["text"], "score": round(s, 4)}
                     for d, s in hits],
        }

    @reg.tool(description="按工单号查询工单状态与复核信息")
    def ticket_query(ticket_id: str):
        """查询工单

        ticket_id: 工单号，如 T-100
        """
        if ticket_id not in TICKETS:
            raise ToolError(f"工单不存在: {ticket_id}")
        return TICKETS[ticket_id]

    @reg.tool(description="发送通知邮件（外部副作用，需人工确认）", sensitive=True)
    def send_mail(to: str, subject: str = "通知"):
        """发送邮件

        to: 收件人
        subject: 主题
        """
        MAIL_LOG.append({"to": to, "subject": subject})
        return {"sent": True, "to": to}

    return reg


ROUTE_TABLE = {
    "policy_inquiry": ["kb_search"],                       # 只查制度
    "ticket_review": ["kb_search", "ticket_query"],         # 查制度 + 查工单
    "notify": ["kb_search", "send_mail"],                   # 允许发邮件
    "general": [],                                          # 未识别 -> 空白名单
}


def make_engine(decisions, checkpointer=None, **kw):
    """decisions: 脚本化的模型决策队列。"""
    reg = build_registry()
    cursor = {"i": 0}

    def act_fn(state, allowed):
        i = min(cursor["i"], len(decisions) - 1)
        cursor["i"] += 1
        return decisions[i]

    return AgentEngine(
        registry=reg,
        intent_fn=kw.pop("intent_fn", lambda q: "ticket_review"),
        plan_fn=lambda q, t: ["检索制度条款", "查询工单状态", "汇总结论"],
        act_fn=act_fn,
        route_table=ROUTE_TABLE,
        checkpointer=checkpointer,
        **kw,
    )


# ================================================================== 1 全流程
banner("1. 多步任务全流程 + SSE 事件流")

engine = make_engine([
    {"tool": "kb_search", "args": {"query": "高风险工单 人工复核", "top_k": 2}},
    {"tool": "ticket_query", "args": {"ticket_id": "T-100"}},
    {"final": "高风险工单必须人工复核后方可关闭 [1]；但 T-100 已关闭且无复核记录，"
              "存在流程违规风险 [2]",
     "citations": ["c0", "T-100"]},
])
run = engine.run("工单 T-100 的关闭流程是否合规？", thread_id="demo-1")

log("任务: 工单 T-100 的关闭流程是否合规？")
log(f"意图识别 -> task_type={run.state.task_type}")
log(f"白名单   -> {run.state.allowed_tools}")
log(f"执行计划 -> {run.state.plan}")
log()
log("事件流（前端轨迹面板按这些渲染）：")
for e in run.events:
    d = e["data"]
    if e["type"] == EventType.TOOL_CALL:
        log(f"  [tool_call]   {d['tool']}({d['args']})")
    elif e["type"] == EventType.TOOL_RESULT:
        res = d["result"] if isinstance(d["result"], dict) else {}
        brief = (f"{res.get('count')} 条" if "count" in res
                 else f"状态={res.get('status')}" if "status" in res
                 else "ok")
        log(f"  [tool_result] {d['tool']}  -> {brief}")
    elif e["type"] == EventType.INTENT:
        log(f"  [intent]      {d['task_type']}  允许工具={d['allowed_tools']}")
    elif e["type"] == EventType.VERIFY:
        log(f"  [verify]      ok={d['ok']}  {d['reason']}")
    elif e["type"] == EventType.ANSWER:
        log(f"  [answer]      {d['text'][:60]}...")
        log(f"                引用={d['citations']}")
    elif e["type"] in (EventType.START, EventType.PLAN, EventType.DONE):
        log(f"  [{e['type']}]      {d}")
log()
log(f"轮次={run.state.rounds}  工具调用={run.state.tool_calls}  "
    f"完成={run.state.finished}  耗时={run.state.elapsed_s():.3f}s")

# ================================================================== 2 白名单
banner("2. 工具白名单拦截越权调用")
log("场景：意图识别为 policy_inquiry（只允许 kb_search），模型却试图发邮件。")
log()

engine2 = make_engine([
    {"tool": "send_mail", "args": {"to": "external@x.com", "subject": "通知"}},
    {"tool": "kb_search", "args": {"query": "邮件 审批"}},
    {"final": "制度要求外部邮件须经主管审批 [1]；我无权限直接发送，已转人工。",
     "citations": ["c3"]},
], intent_fn=lambda q: "policy_inquiry")

run2 = engine2.run("帮我发一封通知邮件给外部单位", thread_id="demo-2")
for e in run2.events:
    if e["type"] == EventType.TOOL_ERROR:
        log(f"  [tool_error]  {e['data']['tool']} 被拦截: {e['data']['error']}")
    elif e["type"] == EventType.TOOL_CALL:
        log(f"  [tool_call]   {e['data']['tool']}({e['data']['args']})")
log()
log(f"结果：MAIL_LOG = {MAIL_LOG}  -> 邮件从未真的发出")
log(f"      被拦截的调用**不计入**工具预算："
    f"tool_calls={run2.state.tool_calls}（只有 kb_search 那次被计入）")
log(f"      最终回答：{run2.state.answer}")

# ================================================================== 3 Checkpoint
banner("3. Checkpoint 断点续跑（已完成节点不重跑）")

import shutil
ckpt_dir = os.path.join(ROOT, "data", "demo_checkpoints")
shutil.rmtree(ckpt_dir, ignore_errors=True)

intent_counter = {"n": 0}


def counting_intent(q):
    intent_counter["n"] += 1
    return "ticket_review"


log("第一段：max_rounds=1，跑到第 1 轮就被熔断（模拟进程中断/超时）")
eng_a = make_engine(
    [{"tool": "kb_search", "args": {"query": "复核制度"}}],
    checkpointer=JsonFileCheckpointer(ckpt_dir), max_rounds=1,
    intent_fn=counting_intent,
)
run_a = eng_a.run("工单 T-100 是否合规？", thread_id="T-RESUME")
log(f"  rounds={run_a.state.rounds}  stop_reason={run_a.state.stop_reason}  "
    f"intent_fn 被调用 {intent_counter['n']} 次")
log(f"  检查点文件: {os.listdir(ckpt_dir)}")
log()

log("第二段：同一检查点，放宽 max_rounds 后 resume()")
eng_b = make_engine(
    [{"tool": "kb_search", "args": {"query": "重复调用会被跳过"}},
     {"final": "续跑得到的最终结论：T-100 缺少复核记录，属流程违规。"}],
    checkpointer=JsonFileCheckpointer(ckpt_dir), max_rounds=5,
    intent_fn=counting_intent,
)
run_b = eng_b.resume("T-RESUME")
log(f"  rounds={run_b.state.rounds}（从 1 继续累加，不是从 0 重来）")
log(f"  intent_fn 累计调用 {intent_counter['n']} 次（续跑没有重新识别意图）")
log(f"  finished={run_b.state.finished}")
log(f"  answer={run_b.state.answer}")
log()
log("★ 面试点：Checkpoint 的价值在于「长任务中断后已完成节点不重跑」，")
log("  否则一次超时就得从头再来，成本和体验都不可接受。")

# ================================================================== 4 熔断
banner("4. 超时熔断（三道闸 + 死循环检测）")

engine3 = make_engine(
    [{"tool": "kb_search", "args": {"query": "循环调用"}}],
    max_rounds=10, max_tool_calls=3,
)
run3 = engine3.run("查制度", thread_id="demo-4")
log(f"  工具次数闸门: tool_calls={run3.state.tool_calls}  "
    f"stop_reason={run3.state.stop_reason}")

engine4 = make_engine([{"tool": "kb_search", "args": {"query": "x"}}], max_rounds=6)
run4 = engine4.run("查制度", thread_id="demo-5")
log(f"  轮次闸门:     rounds={run4.state.rounds}  "
    f"stop_reason={run4.state.stop_reason}")

engine5 = make_engine([{"tool": "kb_search", "args": {"query": "x"}}], max_rounds=8)
run5 = engine5.run("查制度", thread_id="demo-6")
log(f"  死循环检测:   最近 2 轮工具序列是否重复 = "
    f"{engine5.detect_loop(run5.state, window=2)}")
log()
log("★ 三道闸各自独立：轮次限制防「想太多」，工具次数限制防「调太多」，")
log("  总耗时限制防「外部依赖太慢」，死循环检测给模型一个换策略的信号。")

# ================================================================== 5 压缩
banner("5. 三级上下文压缩（阈值触发 + 工具结果截断）")

compressor = ContextCompressor(
    token_budget=400, tier1=0.60, tier2=0.85, tier3=0.95,
    keep_recent=(12, 8, 6), max_tool_chars=200,
    summarize_fn=lambda t: f"（摘要）涉及 {len(t)} 字符的历史对话，核心是复核制度与工单状态",
)

for n, chars, label in [(2, 20, "轻载"), (8, 30, "中等"), (20, 40, "重载")]:
    history = [Message("user" if i % 2 == 0 else "assistant",
                       f"第{i}轮：" + "内容" * chars) for i in range(n)]
    history.append(Message("tool", "Z" * 800))   # 一个超长工具返回
    r = compressor.compress(history)
    log(f"  {label:<4} 消息数={n + 1:<3} 压缩档位={r.level:<20} "
        f"token {r.tokens_before:>5} -> {r.tokens_after:<5} "
        f"节省 {r.saved_ratio * 100:5.1f}%")

log()
log("★ 注意：工具结果截断是**无条件**的第一步（不受阈值影响）。")
log("  一次召回 20 条 chunk 的长 JSON 常占上下文大头，而模型真正需要的只是前几条 ——")
log("  这一步零成本、立竿见影，是性价比最高的压缩手段。")

banner("6. 三层记忆协同")
mem = LayeredMemory()
mem.working.add("工单 T-100 是否合规？", "存在违规风险")
mem.working.add("复核人需要什么权限？", "三级及以上")
mem.episodic.add("用户在排查 2026 年 8 月的高风险工单流程合规问题")
mem.profile.update_many({"角色": "风控专员", "常查类型": "高风险工单"})
ctx = mem.build_context("高风险工单 复核权限", recent_n=2, recall_k=1)
log(f"  用户画像：{ctx['profile']}")
log(f"  长期召回：{[e['summary'] for e in ctx['episodic']]}")
log(f"  短期窗口：{[m['content'][:18] for m in ctx['recent']]}")
log()
log("★ 分工：短期=最近几轮逐字保留（模型直接看）；")
log("        长期=很久以前的摘要按相关度召回（模型按需取）；")
log("        画像=稳定属性覆盖式更新。")
log("  面试常问「为什么不全部塞进上下文」：Token 成本与注意力稀释。")

# ================================================================== 写文件
io.open(os.path.join(ROOT, "demo_agent_output.txt"), "w",
        encoding="utf-8").write("\n".join(OUT))
print("\n已写入 demo_agent_output.txt")
