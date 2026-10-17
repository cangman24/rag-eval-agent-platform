# -*- coding: utf-8 -*-
"""rag2_kb_agent/app.py —— 项目二装配：把工具、引擎、决策器、记忆接起来。

对应简历里的「基于 LangGraph 的受控 Agent 工作流 + MCP 工具体系与权限收敛
+ 分层记忆与上下文压缩 + 评测体系」。

四个机制在这里合流：
    core/mcp.py       工具注册中心 + 白名单（意图 -> task_type -> allowed_tools）
    core/agent.py     状态机 + Checkpoint 续跑 + 三道熔断 + 事件流
    core/memory.py    三层记忆 + 三级上下文压缩
    rag2/decisions.py 真模型 function calling 决策解析
    rag2/prompts.py   system_prompt 按白名单 + 记忆组装，工具结果按预算截断

★ 本文件**不依赖 langgraph 库**：core/agent.py 是自实现的等价状态机，
  Checkpoint / 白名单 / 熔断的逻辑已被 558 项单测覆盖，可在无网络环境验证。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.agent import AgentEngine, Checkpointer, JsonFileCheckpointer
from core.memory import ContextCompressor, LayeredMemory, Message
from core.mcp import ToolRegistry
from rag2_kb_agent.decisions import ToolCallingDecider
from rag2_kb_agent.prompts import MessageBuilder, messages_token_estimate

# --------------------------------------------------------------------- 路由表
# ★ 意图 -> 工具白名单。这是"权限收敛"的唯一定义处，不在别处硬编码工具名。
DEFAULT_ROUTE_TABLE: Dict[str, Sequence[str]] = {
    "spec_inquiry": ("kb_search",),                              # 只查制度
    "record_lookup": ("kb_search", "ticket_query"),               # 查制度 + 查记录
    "metric_report": ("kb_search", "metric_query"),               # 查制度 + 查指标
    "full_review": ("kb_search", "ticket_query", "metric_query"),  # 综合审查
    "general": (),                                                # 未识别 -> 空白名单
}

# 意图识别用的关键词（真机可换成语义/LLM 分类，这里保留规则兜底）
#
# ★ 顺序即优先级：命中即返回，所以更"专"的意图要排在前面。
#   踩坑记录：初版的 metric_report 只匹配「指标/统计/数量/多少起」，
#   而真实提问是「违规率的口径是怎么定义的？」—— 一个都不命中，
#   于是落到 general（空白名单），模型想调工具被拦、任务判失败。
#   demo_agenteval 的输出把这个缺口暴露得很清楚（metric_report 成功率 0）。
#   教训：规则意图必须用**真实提问语料**回归，不能凭想象写关键词。
INTENT_RULES: List[tuple] = [
    ("full_review", ("合规", "审查", "是否违规", "风险")),
    ("record_lookup", ("T-", "单号", "处理记录", "查询工单", "工单号")),
    ("metric_report", ("指标", "统计", "数量", "多少起", "违规率", "口径", "占比", "百分比", "工单量", "是多少")),
    ("spec_inquiry", ("制度", "规范", "要求", "标准", "流程", "权限",
                      "复核", "关闭")),
]


def rule_based_intent(query: str) -> str:
    """规则意图识别。

    ★ 为什么保留规则而不是全交给 LLM：
      ① 零成本、零延迟；② 可预测（决定工具权限，不能有不确定性）；
      ③ 命中不了再退到 general（空白名单），这是最保守的默认。
      真机可以叠加 LLM 分类，但**权限决策不应完全交给模型**。
    """
    q = query or ""
    for task_type, keywords in INTENT_RULES:
        if any(k in q for k in keywords):
            return task_type
    return "general"


def rule_based_plan(query: str, task_type: str) -> List[str]:
    """按意图给一个粗粒度计划（模型会在 act 阶段细化）。"""
    plans = {
        "spec_inquiry": ["检索相关制度条款", "汇总并标注来源"],
        "record_lookup": ["检索制度依据", "查询具体记录", "对照制度给出结论"],
        "metric_report": ["检索指标口径", "查询统计数值", "汇总成结论"],
        "full_review": ["检索制度依据", "查询记录明细", "查询统计指标", "综合判断合规性"],
        "general": ["判断是否需要工具", "回答或说明信息不足"],
    }
    return plans.get(task_type, plans["general"])


# --------------------------------------------------------------------- 记忆
class ConversationMemory:
    """把 LayeredMemory 包成"可被提示词层调用"的 provider。

    同时负责：每轮结束后把工具结果压缩成情景记忆（避免上下文持续膨胀）。
    """

    def __init__(self, memory: Optional[LayeredMemory] = None,
                 compressor: Optional[ContextCompressor] = None) -> None:
        self.memory = memory or LayeredMemory()
        self.compressor = compressor or ContextCompressor(token_budget=6000)

    def context_provider(self, query: str) -> Dict[str, Any]:
        return self.memory.build_context(query, recent_n=6, recall_k=3)

    def record_turn(self, user: str, assistant: str) -> None:
        self.memory.working.add(user, assistant)

    def archive_observations(self, state: Any) -> None:
        """把本轮的工具观察压成一条摘要进长期记忆。

        ★ 这是"中间结果写入、按 token 预算注入"的落地：
          工具原始返回不进长期记忆（太长、噪声大），只进摘要。
        """
        hits: List[str] = []
        for item in getattr(state, "scratchpad", []) or []:
            if item.get("type") == "tool_call" and item.get("ok"):
                tool = item.get("tool", "tool")
                result = item.get("result")
                if isinstance(result, dict) and result.get("count"):
                    hits.append(f"{tool} 命中 {result['count']} 条")
                else:
                    hits.append(tool)
        if not hits:
            return
        query = getattr(state, "query", "")
        answer = getattr(state, "answer", "")
        summary = f"问「{query[:60]}」→ 调用 {', '.join(hits)}；结论：{answer[:120]}"
        self.memory.episodic.add(summary, meta={"task_type": state.task_type})

    def compress(self, messages: Sequence[Message]) -> Any:
        return self.compressor.compress(messages)


# --------------------------------------------------------------------- 装配
def build_engine(
    registry: ToolRegistry,
    decider_factory: Callable[[ToolRegistry, Any], Any],
    memory: Optional[ConversationMemory] = None,
    checkpointer: Optional[Checkpointer] = None,
    route_table: Optional[Dict[str, Sequence[str]]] = None,
    intent_fn: Optional[Callable[[str], str]] = None,
    plan_fn: Optional[Callable[[str, str], List[str]]] = None,
    max_rounds: int = 6,
    max_tool_calls: int = 12,
    total_timeout_s: float = 120.0,
) -> AgentEngine:
    """组装受控 Agent 引擎。

    Args:
        decider_factory: `(registry, memory_provider) -> act_fn`
            测试时注入脚本化决策器；生产时注入 `ToolCallingDecider`。
            这样"引擎装配"与"模型接入"解耦，引擎逻辑可离线验证。
    """
    memory = memory or ConversationMemory()

    def _plan(q: str, t: str) -> List[str]:
        return (plan_fn or rule_based_plan)(q, t)

    intent_calls: Dict[str, int] = {"n": 0}

    def _intent(q: str) -> str:
        intent_calls["n"] += 1
        return (intent_fn or rule_based_intent)(q)

    engine = AgentEngine(
        registry=registry,
        intent_fn=_intent,
        plan_fn=_plan,
        act_fn=decider_factory(registry, memory.context_provider),
        route_table=dict(route_table or DEFAULT_ROUTE_TABLE),
        checkpointer=checkpointer,
        max_rounds=max_rounds,
        max_tool_calls=max_tool_calls,
        total_timeout_s=total_timeout_s,
    )
    engine._intent_calls = intent_calls  # 便于测试断言"续跑没重新识别意图"
    return engine


def build_ablation_engine(
    cfg: Any,
    decider_factory: Callable[[ToolRegistry, Any], Any],
    registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kw: Any,
) -> AgentEngine:
    """按一档消融配置装配引擎（★ 让四档差异真正落在能力开关上）。

    ★ 关键设计：`use_tools=False` 的实现方式是**把路由表整体清空**，
      而不是"提示模型不要调工具"。原因：
        ① 提示词约束不可靠（模型仍可能调）；
        ② 清空白名单后，任何工具调用都会被引擎硬拦截，
           baseline 就成了干净的"直答"对照组。

      `use_memory=False` 用"返回空上下文的 provider"，而不是改引擎 ——
      这样记忆层的有无是唯一变量，其它链路完全一致（消融的前提）。
    """
    use_tools = getattr(cfg, "use_tools", True)
    use_memory = getattr(cfg, "use_memory", False)
    use_compress = getattr(cfg, "use_compress", False)

    # 空白名单 = 禁掉所有工具
    if use_tools:
        route_table = kw.pop("route_table", None) or DEFAULT_ROUTE_TABLE
    else:
        route_table = {k: () for k in DEFAULT_ROUTE_TABLE}

    # 记忆开关：关掉时注入一个"永远返回空"的 provider
    memory = kw.pop("memory", None)
    if memory is None:
        if use_compress:
            from core.memory import ContextCompressor
            memory = ConversationMemory(
                compressor=ContextCompressor(token_budget=3000))
        else:
            memory = ConversationMemory()
    if not use_memory:
        memory = ConversationMemory(memory=memory.memory)
        memory.context_provider = lambda q: {"profile": "", "episodic": [],
                                             "recent": []}

    # ★ max_rounds 的来源有三处，必须只取一次，否则会 "got multiple values"：
    #     ① cfg.max_rounds（消融配置，优先级最高）
    #     ② kw 里显式传的
    #     ③ build_engine 的默认值
    #   这里统一 pop 出来再显式传下去。
    kw_rounds = kw.pop("max_rounds", None)
    max_rounds = (getattr(cfg, "max_rounds", None) or kw_rounds or 6)

    return build_engine(
        registry if registry is not None else ToolRegistry(),
        decider_factory,
        memory=memory,
        checkpointer=checkpointer,
        route_table=route_table,
        max_rounds=int(max_rounds),
        **kw,
    )


def build_tool_calling_engine(
    llm: Any,
    registry: ToolRegistry,
    memory: Optional[ConversationMemory] = None,
    checkpointer: Optional[Checkpointer] = None,
    **kw: Any,
) -> AgentEngine:
    """生产装配：用真 LLM 的 function calling 做决策。"""
    memory = memory or ConversationMemory()
    builder = MessageBuilder(registry=registry)

    def factory(reg: ToolRegistry, memory_provider: Any) -> Any:
        builder.memory_provider = memory_provider
        return ToolCallingDecider(llm=llm, registry=reg, build_messages=builder)

    return build_engine(registry, factory, memory=memory,
                        checkpointer=checkpointer, **kw)


def build_tool_calling_ablation_engine(
    llm: Any,
    registry: ToolRegistry,
    cfg: Any,
    checkpointer: Optional[Checkpointer] = None,
    **kw: Any,
) -> AgentEngine:
    """消融专用的生产装配：同样的 LLM，只切换能力开关。"""
    builder = MessageBuilder(registry=registry)

    def factory(reg: ToolRegistry, memory_provider: Any) -> Any:
        builder.memory_provider = memory_provider
        return ToolCallingDecider(llm=llm, registry=reg, build_messages=builder)

    return build_ablation_engine(cfg, factory, registry=registry,
                                 checkpointer=checkpointer, **kw)


def build_registry_from_index(index: Any, metrics_fn: Optional[Callable] = None
                              ) -> ToolRegistry:
    """把检索索引封装成 3 个 MCP 工具。

    ★ 三个工具对应白名单的三个粒度：
        kb_search     —— 所有意图都允许（检索制度）
        ticket_query  —— 仅 record_lookup / full_review
        metric_query  —— 仅 metric_report / full_review
      这样"意图决定能调什么"就是结构性保证，而不是靠提示词约束。
    """
    from core.bm25 import BM25Index

    docs: Dict[str, str] = {}
    lookup: Callable[[str], str] = lambda _c: ""  # noqa: E731
    if hasattr(index, "child_text"):
        docs = {c.chunk_id: c.text for c in index.chunks}
        lookup = index.doc_text
    elif hasattr(index, "chunks"):
        docs = {c.chunk_id: c.text for c in index.chunks}

    reg = ToolRegistry()

    @reg.tool(description="检索知识库，返回与查询最相关的制度条款片段")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 检索关键词
        top_k: 返回条数（建议 3~5）
        """
        if hasattr(index, "bm25"):
            hits = index.bm25.search(query, top_k=top_k, min_score=0.3)
            if not hits:
                hits = index.bm25.search(query, top_k=top_k, min_score=0.0)
            return {"count": len(hits),
                    "hits": [{"id": d, "text": lookup(d)} for d, _ in hits]}
        return {"count": 0, "hits": [], "note": "索引未就绪"}

    @reg.tool(description="按工单号查询工单状态与处理记录")
    def ticket_query(ticket_id: str):
        """查询工单

        ticket_id: 工单号，如 T-100
        """
        store = getattr(index, "tickets", None) or {}
        if ticket_id in store:
            return store[ticket_id]
        return {"ticket_id": ticket_id, "status": "not_found",
                "note": "未找到该工单，请确认单号"}

    @reg.tool(description="查询业务指标统计值（数量、占比、趋势）")
    def metric_query(metric: str, period: str = "近30天"):
        """查询指标

        metric: 指标名，如 工单量 / 违规率
        period: 统计周期
        """
        provider = metrics_fn or (lambda m, p: {"metric": m, "period": p,
                                                "value": None,
                                                "note": "指标服务未接入"})
        return provider(metric, period)

    return reg


def build_default_checkpointer(path: str = "data/checkpoints") -> Checkpointer:
    """默认用 JSON 文件落盘 —— 零依赖、可读、便于调试。

    真机要并发写再换 SqliteSaver，接口不变。
    """
    try:
        return JsonFileCheckpointer(path)
    except OSError:
        from core.agent import InMemoryCheckpointer
        return InMemoryCheckpointer()
