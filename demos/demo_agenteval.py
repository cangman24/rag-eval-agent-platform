# -*- coding: utf-8 -*-
"""demo_agenteval.py —— 项目二的 Agent 层消融评测（六类指标）。

跑法：python demo_agenteval.py
产物：demo_agenteval_output.txt + reports/agent_demo_ablation.md

★ 与 demo_agent2.py 的区别：
    demo_agent2.py     验证"真 function calling 决策循环"能不能跑通
    demo_agenteval.py  验证"四档消融能不能量化出各能力模块的贡献"

四档配置（差异落在能力开关上，不是只改名字）：
    A_baseline_no_tools   空白名单 -> 模型拿不到工具，只能直答
    B_plus_tools          给工具，无记忆、无压缩
    C_plus_memory         加三层记忆
    D_full_harness        再加三级上下文压缩（= Full Harness）

六类指标：成功率 / 平均轮次 / 平均工具调用 / 平均工具失败 / 平均Token / P95延迟
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent_eval import (
    AgentTask, agent_ablation_configs, render_agent_ablation,
    render_agent_type_table, render_agent_failures, run_agent_eval,
    save_agent_tasks, validate_agent_tasks, write_agent_report,
)
from core.bm25 import BM25Index
from core.mcp import ToolRegistry
from core.retrieval import recursive_chunk
from rag2_kb_agent.app import build_ablation_engine
from rag2_kb_agent.decisions import extract_text

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

    @reg.tool(description="按工单号查询工单状态与复核记录")
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

    return reg


# ================================================================== 模拟 LLM
class PolicyLLM:
    """模拟"守规矩的模型"：按任务计划逐个调用工具，然后作答。

    ★ 这个假模型的行为对四档消融是**一致的** —— 它总是尝试调工具。
      差异完全来自引擎的能力开关：
        A 档（空白名单）-> 调用被硬拦 -> 工具序列为空 -> 任务判失败
        B/C/D 档        -> 调用成功 -> 任务成功
      这正是"消融要把能力模块拆开"的正确做法：模型不变，只变能力。

    ★ 已执行工具的判断方式：读 prompt 里"已执行的工具调用"段落中出现的
      工具名次数。初版我数 "返回：" 字符串，结果模型永远以为没执行过、
      无限重复同一个工具直到 max_rounds —— demo 输出里出现
      "kb_search,kb_search,kb_search..." 就是这个 bug。
    """

    def __init__(self, planner):
        self.planner = planner      # (query) -> [(tool, args), ...]
        self.calls = 0

    def _executed_count(self, user_msg: str, plan) -> int:
        """数 prompt 里已经出现过多少次计划中的工具调用。"""
        marker = "已执行的工具调用："
        if marker not in user_msg:
            return 0
        history = user_msg.split(marker, 1)[1]
        return sum(history.count(f" {tool} 返回") + history.count(f" {tool} 失败")
                   + history.count(f" {tool} 被拒绝")
                   for tool, _ in plan)

    def chat(self, messages, temperature=0.0, tools=None, **kw):
        self.calls += 1
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        query = user_msg.split("任务：")[-1].split("\n")[0].strip()
        plan = self.planner(query)
        done = self._executed_count(user_msg, plan)

        if done < len(plan):
            tool, args = plan[done]
            raw = {"choices": [{"message": {
                "role": "assistant",
                "content": f"我先调用 {tool}。",
                "tool_calls": [{"id": f"c{done}", "type": "function",
                                "function": {"name": tool,
                                             "arguments": json.dumps(args, ensure_ascii=False)}}],
            }}]}
        else:
            raw = {"choices": [{"message": {"role": "assistant",
                                            "content": self.answer(query)}}]}

        class R:
            pass
        r = R()
        r.raw = raw
        r.text = extract_text(raw)
        return r

    @staticmethod
    def answer(query: str) -> str:
        if "T-100" in query:
            return ("高风险工单必须人工复核并留痕 [1]；"
                    "T-100 已关闭但无复核记录、证据缺失，属流程违规 [2]。")
        if "T-200" in query:
            return "T-200 由张三复核且证据完整，符合制度要求 [1][2]。"
        if "违规率" in query or "指标" in query or "口径" in query:
            return "违规率口径 = 未经复核关闭的高风险工单数 / 高风险工单总数 [1]。"
        if "低风险" in query:
            return "低风险工单可由系统自动关闭，无需人工复核 [1]。"
        if "权限" in query:
            return "复核人需具备三级及以上权限，并在系统内留痕 [1]。"
        return "高风险工单必须经人工复核后方可关闭，且需具备三级以上权限 [1]。"


def plan_for(query: str):
    """按意图给出工具调用计划（模拟模型的规划能力）。"""
    if "T-100" in query or "T-200" in query:
        tid = "T-100" if "T-100" in query else "T-200"
        return [("kb_search", {"query": "高风险工单 人工复核"}),
                ("ticket_query", {"ticket_id": tid})]
    if "违规率" in query:
        return [("kb_search", {"query": "违规率 口径"}),
                ("metric_query", {"metric": "违规率"})]
    return [("kb_search", {"query": "复核制度 权限"})]


# ================================================================== 任务集
banner("0. 任务集（多步任务，比检索评测更强调闭环）")
TASKS = [
    AgentTask("t01", "工单 T-100 的关闭流程是否合规？",
              expect_tools=["kb_search", "ticket_query"],
              must_contain=["复核"], task_type="full_review"),
    AgentTask("t02", "工单 T-200 的关闭流程是否合规？",
              expect_tools=["kb_search", "ticket_query"],
              must_contain=["张三"], task_type="full_review"),
    AgentTask("t03", "违规率的口径是怎么定义的？",
              expect_tools=["kb_search", "metric_query"],
              must_contain=["违规率"], task_type="metric_report"),
    AgentTask("t04", "高风险工单关闭需要什么权限？",
              expect_tools=["kb_search"],
              must_contain=["三级"], task_type="spec_inquiry"),
    AgentTask("t05", "低风险工单需要人工复核吗？",
              expect_tools=["kb_search"],
              must_contain=["自动关闭"], task_type="spec_inquiry"),
]
health = validate_agent_tasks(TASKS)
log(f"任务数={health['total']}  类型分布={health['by_type']}  "
    f"体检={'通过' if health['ok'] else health['problems']}")
TASK_FILE = os.path.join(ROOT, "data", "eval", "demo_agent_tasks.jsonl")
save_agent_tasks(TASKS, TASK_FILE)
log(f"已写入 {os.path.relpath(TASK_FILE, ROOT)}")

# ================================================================== 跑四档
banner("1. 四档消融（模型不变，只切能力开关）")


def make_factory(llm):
    """构造 decider_factory。

    ★ 契约必须对齐：`build_engine` 调的是 `decider_factory(registry,
      memory_provider)` —— 只有两个参数。所以 llm 要用闭包捕获，
      不能把它当成第三个位置参数（否则报 missing argument）。
    """
    def factory(registry, memory_provider):
        from rag2_kb_agent.decisions import ToolCallingDecider
        from rag2_kb_agent.prompts import MessageBuilder
        builder = MessageBuilder(registry=registry)
        builder.memory_provider = memory_provider
        return ToolCallingDecider(llm=llm, registry=registry,
                                  build_messages=builder)
    return factory


log(f"{'配置':<26}{'成功率':>9}{'平均轮次':>10}{'工具调用':>10}"
    f"{'工具失败':>10}{'Token':>9}{'P95(ms)':>10}")
log("-" * 86)

reports = []
for cfg in agent_ablation_configs():
    registry = build_registry()
    # ★ 必须在这里 new 一个 LLM。
    #   踩坑：初版写成 `make_factory(PolicyLLM(plan_for))` ——
    #   LLM 在构造 factory 时就被求值了，四档共享同一个实例，
    #   于是每个 engine 里都绑着最后一个 LLM（闭包变量被反复覆盖）。
    #   正确做法是把「创建 LLM」放进循环体。
    llm = PolicyLLM(plan_for)
    engine = build_ablation_engine(
        cfg,
        decider_factory=make_factory(llm),
        registry=registry,
        max_rounds=cfg.max_rounds,
    )
    rep = run_agent_eval(TASKS, lambda q, e=engine: e.run(q),
                         config_name=f"{cfg.name} ({cfg.label()})")
    reports.append(rep)
    m = rep.metrics
    log(f"{cfg.name + ' (' + cfg.label() + ')':<26}"
        f"{m['success_rate']:>9.4f}{m['avg_rounds']:>10.2f}"
        f"{m['avg_tool_calls']:>10.2f}{m['avg_tool_errors']:>10.2f}"
        f"{m['avg_tokens']:>9.1f}{m['p95_latency_ms']:>10.1f}")


# ================================================================== 解读
banner("2. 消融结果解读")
base = reports[0].metrics
full = reports[-1].metrics
log(f"  成功率：{base['success_rate']:.2f} -> {full['success_rate']:.2f} "
    f"（提升 {(full['success_rate'] - base['success_rate']) * 100:.1f} 个百分点）")
log(f"  平均工具调用：{base['avg_tool_calls']:.2f} -> {full['avg_tool_calls']:.2f}")
log(f"  平均 Token：{base['avg_tokens']:.0f} -> {full['avg_tokens']:.0f}")
log()
log("★ 关键观察：")
log("  · A 档（空白名单）成功率接近 0 —— 因为任务要求'必须检索过'，")
log("    而模型调工具被引擎硬拦，工具序列为空 -> 判定失败。")
log("    这说明**空白名单是真的禁掉了工具**，不是靠提示词请求模型别调。")
log("  · B 档拿到工具后成功率跳升 —— 这是工具模块的贡献。")
log("  · C/D 档成功率与 B 相同时，要看**平均 Token**：")
log("    记忆与压缩的价值不在于提高成功率，而在于**用更少的上下文达成同样效果**。")
log("    只看成功率会误判'记忆没用'，这就是为什么要同时报六类指标。")

banner("3. 分任务类型（哪个类型最弱）")
log(render_agent_type_table(reports[-1]))

banner("4. 失败任务清单（Badcase 回流）")
log(render_agent_failures(reports[0], limit=5))

# ================================================================== 报告
banner("5. 落盘报告")
REPORT = os.path.join(ROOT, "reports", "agent_demo_ablation.md")
write_agent_report(
    reports[-1], REPORT,
    extra_sections={"四档消融对比（六类指标）": render_agent_ablation(reports)},
)
log(f"已写入 {os.path.relpath(REPORT, ROOT)}")
log()
log("报告内容预览：")
content = open(REPORT, encoding="utf-8").read().splitlines()
for line in content[:26]:
    log("  " + line)
log(f"  ...（共 {len(content)} 行）")

# ================================================================== 收尾
banner("★ 接入真模型后要改的只有一处")
log("""
把 PolicyLLM 换成 core.llm.LLMClient：

    from core.llm import LLMClient, DiskCache
    llm = LLMClient(api_key=os.environ["DEEPSEEK_API_KEY"],
                    cache=DiskCache("data/cache/llm"))
    engine = build_tool_calling_ablation_engine(llm, registry, cfg)

四档消融、六类指标、报告渲染**全部不需要改动**。

★ 一个真实限制要如实说：
  当前 avg_tokens 是**按工具结果文本量估算**的（因为假模型没有真实
  prompt_tokens）。真机接 LLMClient 后，应该把每次调用的
  LLMResult.prompt_tokens 累加进事件流，改用真口径。
  面试被问到时如实说明这一点，比含糊过去好。
""")

io.open(os.path.join(ROOT, "demo_agenteval_output.txt"), "w",
        encoding="utf-8").write("\n".join(OUT))
print("\n已写入 demo_agenteval_output.txt")
