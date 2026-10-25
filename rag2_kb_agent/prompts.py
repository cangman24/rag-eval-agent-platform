# -*- coding: utf-8 -*-
"""rag2_kb_agent/prompts.py —— 组装发给模型的 messages。

对应简历里的「system_prompt 按当前启用的工具列表、跨会话持久化记忆、
Sub-agent 并发规则组装」以及「中间结果写入、按 token 预算裁剪注入」。

★ 这里的核心工程问题是 **上下文预算**：Agent 跑 5 轮，每轮工具结果可能
  上千字符，如果全量回灌，第 4 轮就爆窗或成本失控。所以：

    1. 工具结果**进 messages 前先截断**（不是等超窗再压缩）
    2. 只保留最近 N 轮的工具调用明细，更早的压成一行摘要
    3. 工具列表只注入**白名单内**的（这直接降低模型幻觉调用的概率）
    4. 记忆分三层注入：画像（稳定）、长期召回（按需）、短期窗口（最近）

  `build_messages` 是纯函数（输入 state -> 输出 list[dict]），所以可以
  在不联网的情况下断言"prompt 里确实只有白名单工具""超长结果被截断"。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from core.memory import truncate_tool_result

# --------------------------------------------------------------------- 模板
SYSTEM_TEMPLATE = """你是企业知识库助手，通过调用工具完成任务。

可用工具（只能调用这些，不要臆造其他工具）：
{tool_list}

工作规则：
1. 需要事实依据时必须调用工具检索，不要凭常识作答。
2. 多步任务先检索、再查询具体记录，最后汇总；每步只调一个最合适的工具。
3. 最终回答必须标注来源，格式如 [1]、[2]，与检索结果一一对应。
4. 信息不足时明确说"知识库中未找到相关内容"，禁止编造。
5. 工具报错时可以调整参数重试一次；仍失败则如实说明并给出已获得的部分结论。
{memory_block}"""

MEMORY_TEMPLATE = """
已知背景（供参考，不要直接当作回答）：
{lines}"""


def format_tool_list(tools: Sequence[Dict[str, Any]]) -> str:
    """把 OpenAI 格式的工具列表渲染成人类可读清单。

    ★ 除了 name/description，还要把**参数名**列出来。
      实测经验：只给描述不给参数名，模型更容易漏填必填参数。
    """
    if not tools:
        return "（当前任务无可用工具，请直接依据已有信息作答；若信息不足则说明未找到）"
    lines = []
    for t in tools:
        fn = t.get("function") or t
        name = fn.get("name", "?")
        desc = (fn.get("description") or "").strip().split("\n")[0]
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])
        if props:
            sig = ", ".join(
                f"{p}{'*' if p in required else ''}" for p in props
            )
        else:
            sig = ""
        lines.append(f"- {name}({sig})：{desc}")
    lines.append("（带 * 的参数为必填）")
    return "\n".join(lines)


def format_memory_block(context: Optional[Dict[str, Any]],
                        max_episodic: int = 3) -> str:
    """渲染三层记忆。没有任何记忆时返回空串（不注入空标题浪费 token）。"""
    if not context:
        return ""
    lines: List[str] = []
    profile = (context.get("profile") or "").strip()
    if profile:
        lines.append(f"- 用户画像：{profile}")
    episodic = context.get("episodic") or []
    for i, e in enumerate(episodic[:max_episodic], start=1):
        summary = (e.get("summary") or "").strip()
        if summary:
            lines.append(f"- 历史相关（{i}）：{summary}")
    if not lines:
        return ""
    return MEMORY_TEMPLATE.format(lines="\n".join(lines))


def format_scratchpad(state: Any, recent_rounds: int = 3,
                      max_chars_per_result: int = 600) -> str:
    """把工具调用轨迹压缩成一段文本注入。

    ★ 只保留最近 N 轮明细，更早的压成计数摘要 —— 这是"中间结果写入文件系统，
      按需注入"思想的轻量版：全量回灌会迅速吃掉预算。
    """
    items = [s for s in (getattr(state, "scratchpad", []) or [])
             if s.get("type") == "tool_call"]
    if not items:
        return ""
    recent = items[-recent_rounds:]
    older = len(items) - len(recent)
    lines: List[str] = []
    if older > 0:
        lines.append(f"（更早的 {older} 次工具调用已省略，结论已体现在后续步骤中）")
    for i, item in enumerate(recent, start=1):
        tool = item.get("tool", "tool")
        if item.get("blocked"):
            lines.append(f"{i}. {tool} 被拒绝：{item.get('error', '不在白名单')}")
            continue
        if not item.get("ok"):
            lines.append(f"{i}. {tool} 失败：{str(item.get('error'))[:200]}")
            continue
        result = item.get("result")
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        lines.append(f"{i}. {tool} 返回：{truncate_tool_result(text, max_chars_per_result)}")
    return "\n".join(lines)


class MessageBuilder:
    """把 AgentState 组装成 LLM messages。

    Args:
        registry: 工具注册中心（用于取白名单内的工具描述）
        memory_provider: `(query) -> {"profile":str, "episodic":[...], "recent":[...]}`
            通常是 `core.memory.LayeredMemory.build_context`
        max_tool_result_chars: 单条工具结果注入上限
        recent_rounds: 轨迹里保留明细的轮数
        system_extra: 追加到 system prompt 的项目特定指令
    """

    def __init__(
        self,
        registry: Any,
        memory_provider: Optional[Any] = None,
        max_tool_result_chars: int = 600,
        recent_rounds: int = 3,
        system_extra: str = "",
    ) -> None:
        self.registry = registry
        self.memory_provider = memory_provider
        self.max_tool_result_chars = max_tool_result_chars
        self.recent_rounds = recent_rounds
        self.system_extra = system_extra

    def build(self, state: Any, allowed_tools: Sequence[str]) -> List[Dict[str, str]]:
        tools = self.registry.openai_tools(allowed=allowed_tools)
        memory_block = ""
        if self.memory_provider is not None:
            try:
                ctx = self.memory_provider(getattr(state, "query", ""))
            except Exception:  # noqa: BLE001 - 记忆层坏了不该让整轮挂掉
                ctx = None
            memory_block = format_memory_block(ctx)

        system = SYSTEM_TEMPLATE.format(
            tool_list=format_tool_list(tools), memory_block=memory_block)
        if self.system_extra:
            system = f"{system}\n{self.system_extra}"

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

        plan = getattr(state, "plan", None) or []
        query = getattr(state, "query", "")
        user_parts = [f"任务：{query}"]
        if plan:
            user_parts.append("建议步骤：\n" + "\n".join(
                f"{i}. {s}" for i, s in enumerate(plan, start=1)))
        scratch = format_scratchpad(state, self.recent_rounds,
                                    self.max_tool_result_chars)
        if scratch:
            user_parts.append("已执行的工具调用：\n" + scratch)
            user_parts.append("请基于以上信息继续：若已能作答，直接给出带引用的最终回答；"
                              "若还需信息，调用下一个工具。")
        else:
            user_parts.append("请开始执行：若需要事实依据，先调用检索工具。")
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        # 短期记忆作为历史对话追加在最后（在任务描述之后，避免模型误把它当任务）
        if self.memory_provider is not None:
            try:
                ctx = self.memory_provider(query) or {}
            except Exception:  # noqa: BLE001
                ctx = {}
            for m in (ctx.get("recent") or [])[-6:]:
                role = m.get("role")
                content = (m.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        return messages

    # 让实例可直接当 build_messages 用
    def __call__(self, state: Any, allowed_tools: Sequence[str]) -> List[Dict[str, str]]:
        return self.build(state, allowed_tools)


def messages_token_estimate(messages: Sequence[Dict[str, str]]) -> int:
    """估算 messages 的 token 数（本地近似，用于预算断言）。"""
    from core.retrieval import count_tokens_approx
    return sum(count_tokens_approx(m.get("content", "")) for m in messages)
