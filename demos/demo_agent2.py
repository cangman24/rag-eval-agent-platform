# -*- coding: utf-8 -*-
"""demo_agent2.py —— 项目二端到端演示：真 function calling 决策循环。

跑法：python demo_agent2.py
输出：demo_agent2_output.txt

★ 与 demo_agent.py 的区别：
    demo_agent.py   用"脚本化决策队列"验证**引擎逻辑**（白名单/续跑/熔断）
    demo_agent2.py  用"模拟 OpenAI function calling 响应"验证**接真模型这条路径**
                     （决策解析 / 多轮 messages 组装 / 上下文预算 / 兜底）

演示六个场景：
  1. 多步任务：检索制度 → 查工单 → 汇总结论（真实决策格式）
  2. 幻觉工具调用被拦截，下一轮模型改口
  3. 越权调用被白名单拦下
  4. 模型输出无法解析 → 兜底回答（带上已获得的工具结果）
  5. 上下文预算：长轨迹截断后 prompt 规模有上界
  6. 断点续跑（真决策器版本）
"""
import io
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import EventType
from core.bm25 import BM25Index
from core.mcp import ToolRegistry
from core.memory import ContextCompressor, LayeredMemory
from core.retrieval import recursive_chunk
from rag2_kb_agent.app import (
    ConversationMemory, build_engine, rule_based_intent, rule_based_plan,
)
from rag2_kb_agent.decisions import ToolCallingDecider, extract_text
from rag2_kb_agent.prompts import MessageBuilder, messages_token_estimate

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


# ================================================================== 模拟 LLM
def resp_tool(name, args, content=None, call_id="call_1"):
    """构造 OpenAI function calling 响应。"""
    return {"choices": [{"message": {
        "role": "assistant", "content": content,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name,
                                     "arguments": json.dumps(args, ensure_ascii=False)}}],
    }}]}


def resp_text(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class ScriptedLLM:
    """按脚本返回 function calling 响应，记录每轮收到的 messages/tools。

    这比"真调 DeepSeek"更适合演示 —— 它把**决策循环**的每一步都摊开，
    而且不用联网、不花钱、结果可复现。真实接入只需把本类换成 LLMClient。
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.seen_tools = []

    def chat(self, messages, temperature=0.0, tools=None, **kw):
        self.calls.append(messages)
        self.seen_tools.append(tools)
        raw = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]

        class R:
            pass
        r = R()
        r.raw = raw
        r.text = extract_text(raw)
        return r


# ================================================================== 语料与工具
CORPUS = """第一章 复核制度
第一条 高风险工单必须经人工复核后方可关闭。
第二条 复核人需具备三级及以上权限，并在系统内留痕。
第二章 工单处理
第三条 工单关闭前需核对处置结果与证据材料是否一致。
第四条 低风险工单可由系统自动关闭，无需人工复核。
第三章 指标口径
第五条 违规率 = 未经复核关闭的高风险工单数 / 高风险工单总数。
"""

chunks = recursive_chunk(CORPUS, chunk_size=64, overlap=8)
DOCS = {f"c{i}": f"{c['heading_path']} {c['text']}" for i, c in enumerate(chunks)}
bm25 = BM25Index()
for did, text in DOCS.items():
    bm25.add(did, text)
bm25.finalize()

TICKETS = {
    "T-100": {"ticket_id": "T-100", "level": "高风险", "status": "closed",
              "reviewed_by": None, "evidence": "缺失"},
    "T-200": {"ticket_id": "T-200", "level": "高风险", "status": "closed",
              "reviewed_by": "张三", "evidence": "完整"},
}


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(description="检索知识库，返回与查询最相关的制度条款片段")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 检索关键词
        top_k: 返回条数
        """
        hits = bm25.search(query, top_k=top_k, min_score=0.3)
        if not hits:
            hits = bm25.search(query, top_k=top_k, min_score=0.0)
        return {"count": len(hits),
                "hits": [{"id": d, "text": DOCS[d]} for d, _ in hits]}

    @reg.tool(description="按工单号查询工单状态与处理记录")
    def ticket_query(ticket_id: str):
        """查询工单

        ticket_id: 工单号，如 T-100
        """
        if ticket_id in TICKETS:
            return TICKETS[ticket_id]
        return {"ticket_id": ticket_id, "status": "not_found"}

    @reg.tool(description="查询业务指标统计值")
    def metric_query(metric: str, period: str = "近30天"):
        """查询指标

        metric: 指标名
        period: 统计周期
        """
        return {"metric": metric, "period": period, "value": "12.5%"}

    @reg.tool(description="发送通知邮件（外部副作用，需人工确认）", sensitive=True)
    def send_mail(to: str):
        """发送邮件

        to: 收件人
        """
        return {"sent": True, "to": to}

    return reg


def make_engine(responses, checkpointer=None, **kw):
    reg = build_registry()
    llm = ScriptedLLM(responses)
    memory = ConversationMemory(
        memory=LayeredMemory(),
        compressor=ContextCompressor(token_budget=3000, keep_recent=(12, 8, 6)),
    )
    builder = MessageBuilder(registry=reg, max_tool_result_chars=400,
                             recent_rounds=3)

    def factory(registry, memory_provider):
        builder.memory_provider = memory_provider
        return ToolCallingDecider(llm=llm, registry=registry,
                                  build_messages=builder)

    engine = build_engine(reg, factory, memory=memory,
                          checkpointer=checkpointer, **kw)
    return engine, llm, reg, memory


def dump_events(run, limit=None):
    for e in run.events:
        d = e["data"]
        t = e["type"]
        if t == EventType.INTENT:
            log(f"  [intent]       {d['task_type']}  允许工具={d['allowed_tools']}")
        elif t == EventType.TOOL_CALL:
            log(f"  [tool_call]    {d['tool']}({json.dumps(d['args'], ensure_ascii=False)})")
        elif t == EventType.TOOL_RESULT:
            r = d["result"] if isinstance(d["result"], dict) else {}
            brief = (f"{r.get('count')} 条" if "count" in r
                     else f"status={r.get('status')}" if "status" in r
                     else "ok")
            log(f"  [tool_result]  {d['tool']} -> {brief}")
        elif t == EventType.TOOL_ERROR:
            tag = "被拦截" if d.get("blocked") else "执行失败"
            log(f"  [tool_error]   {d['tool']} {tag}: {d.get('error')}")
        elif t == EventType.VERIFY:
            log(f"  [verify]       ok={d['ok']}")
        elif t == EventType.ANSWER:
            log(f"  [answer]       {d['text'][:70]}")
        elif t == EventType.LIMIT:
            log(f"  [limit]        {d}")
        elif t == EventType.DONE:
            log(f"  [done]         rounds={d['rounds']} tool_calls={d['tool_calls']}")


# ================================================================== 1 多步任务
banner("1. 多步任务：检索制度 → 查工单 → 汇总结论（真实 function calling 格式）")
engine, llm, reg, memory = make_engine([
    resp_tool("kb_search", {"query": "高风险工单 人工复核"}, content="我先查一下复核制度。"),
    resp_tool("ticket_query", {"ticket_id": "T-100"}),
    resp_text("高风险工单必须人工复核并留痕 [1]；T-100 已关闭但无复核记录、"
              "证据缺失，属流程违规 [2]。"),
])

log("意图与计划（规则识别，不消耗模型调用）：")
q = "工单 T-100 的关闭流程是否合规？"
log(f"  query = {q}")
log(f"  task_type = {rule_based_intent(q)}")
log(f"  plan      = {rule_based_plan(q, rule_based_intent(q))}")
log()
run = engine.run(q, thread_id="A2-1")
log("事件流：")
dump_events(run)
log()
log(f"最终回答：{run.state.answer}")
log(f"引用：{run.state.citations}")
log(f"轮次={run.state.rounds}  工具调用={run.state.tool_calls}  finished={run.state.finished}")
log()
log("★ 注意第 1 轮：模型边写「我先查一下复核制度」边调工具 ——")
log("  决策解析必须优先信 tool_calls，否则那句话会被当最终答案、任务直接断掉。")

# ================================================================== 2 幻觉工具
banner("2. 幻觉工具调用被拦截，下一轮模型改口")
engine2, llm2, _, _ = make_engine([
    resp_tool("web_search", {"query": "外部资料"}),   # 未注册的工具
    resp_text("我无法使用外部搜索，只能基于本地知识库作答："
              "高风险工单必须人工复核 [1]。"),
])
run2 = engine2.run("查一下高风险工单的复核要求", thread_id="A2-2")
dump_events(run2)
log()
log(f"回答：{run2.state.answer[:60]}")
log("★ 幻觉工具没有被静默丢弃，而是产生了一条可观测的 tool_error 事件。")

# ================================================================== 3 越权
banner("3. 越权调用被白名单拦下")
engine3, llm3, _, _ = make_engine([
    resp_tool("send_mail", {"to": "external@x.com"}),
    resp_text("外部邮件需经主管审批，我无权限直接发送，已转人工。"),
], intent_fn=lambda q: "spec_inquiry")   # 只允许 kb_search
run3 = engine3.run("帮我发一封通知邮件给外部单位", thread_id="A2-3")
dump_events(run3)
log()
log(f"被拦截的调用不计入工具预算：tool_calls={run3.state.tool_calls}")
log("★ 白名单在 system prompt（只列允许的工具）和引擎两层同时生效。")

# ================================================================== 4 兜底
banner("4. 模型输出无法解析 → 兜底回答（带上已获得的结果）")
engine4, _, _, _ = make_engine([
    resp_tool("kb_search", {"query": "复核制度"}),
    {"choices": [{"message": {"content": None, "tool_calls": []}}]},   # 空回复
])
run4 = engine4.run("复核制度要求是什么", thread_id="A2-4")
dump_events(run4)
log()
log(f"兜底回答前 80 字：{run4.state.answer[:80]}...")
log(f"引用：{run4.state.citations}")
log("★ 兜底策略：把已拿到的工具结果组织成回答，而不是让整轮白跑。")

# ================================================================== 5 预算
banner("5. 上下文预算：长轨迹截断后 prompt 规模有上界")
reg5 = build_registry()
builder5 = MessageBuilder(registry=reg5, max_tool_result_chars=200,
                          recent_rounds=2)


class StateStub:
    def __init__(self, n):
        self.query = "查询制度"
        self.plan = ["a", "b"]
        self.rounds = n
        self.scratchpad = [
            {"type": "tool_call", "tool": "kb_search", "ok": True,
             "result": {"count": 20, "hits": [{"id": f"c{i}",
                                              "text": "长文本" * 300}]}}
            for i in range(n)
        ]


log(f"{'轮数':<6}{'messages':<10}{'估算 token':<12}{'工具字符上限':<12}")
for n in (1, 3, 10, 50):
    msgs = builder5.build(StateStub(n), ["kb_search"])
    log(f"{n:<6}{len(msgs):<10}{messages_token_estimate(msgs):<12}{200:<12}")
log()
log("★ 关键：只保留最近 2 轮明细、单条结果截断到 200 字符、")
log("  更早的压成一行摘要 —— 所以轮数增长不会让 prompt 无界膨胀。")
log("  这是「不超窗、成本可控」的结构性保证，而不是等超窗再补救。")

# ================================================================== 6 续跑
banner("6. 断点续跑（真决策器版本）")
ckpt_dir = os.path.join(ROOT, "data", "demo_agent2_ckpt")
shutil.rmtree(ckpt_dir, ignore_errors=True)
from core.agent import JsonFileCheckpointer

cp = JsonFileCheckpointer(ckpt_dir)
intent_counter = {"n": 0}


def counting_intent(q):
    intent_counter["n"] += 1
    return "full_review"


eng_a, _, _, _ = make_engine(
    [resp_tool("kb_search", {"query": "复核制度"})],
    checkpointer=cp, intent_fn=counting_intent, max_rounds=1)
a = eng_a.run("审查 T-100 是否违规", thread_id="A2-RESUME")
log(f"第一段：rounds={a.state.rounds} finished={a.state.finished} "
    f"stop_reason={a.state.stop_reason}  intent_fn 调用 {intent_counter['n']} 次")

eng_b, _, _, _ = make_engine(
    [resp_text("续跑结论：T-100 缺少复核记录，判定为流程违规。")],
    checkpointer=JsonFileCheckpointer(ckpt_dir), intent_fn=counting_intent,
    max_rounds=5)
b = eng_b.resume("A2-RESUME")
log(f"第二段：rounds={b.state.rounds}（从 1 继续累加）finished={b.state.finished}")
log(f"        intent_fn 累计调用 {intent_counter['n']} 次（续跑没重新识别意图）")
log(f"        answer={b.state.answer}")
shutil.rmtree(ckpt_dir, ignore_errors=True)

# ================================================================== 收尾
banner("★ 接入真 DeepSeek 只需换一行")
log("""
把 ScriptedLLM 换成 core.llm.LLMClient 即可：

    from core.llm import LLMClient, DiskCache
    llm = LLMClient(api_key=os.environ["DEEPSEEK_API_KEY"],
                    model="deepseek-chat",
                    cache=DiskCache("data/cache/llm"))
    engine = build_tool_calling_engine(llm, registry, checkpointer=...)

改进点（ToolCallingDecider 已用 inspect 自动探测，无需你传开关）：
  · 客户端支持 tools 参数 -> 走原生 function calling
  · 不支持（或白名单为空）-> 自动回退到纯文本解析

本文件里的引擎、白名单、Checkpoint、熔断、记忆、压缩、决策解析
全部**不需要改动**。
""")

io.open(os.path.join(ROOT, "demo_agent2_output.txt"), "w",
        encoding="utf-8").write("\n".join(OUT))
print("\n已写入 demo_agent2_output.txt")
