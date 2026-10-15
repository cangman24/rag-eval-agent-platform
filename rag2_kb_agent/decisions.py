# -*- coding: utf-8 -*-
"""rag2_kb_agent/decisions.py —— 把"模型输出"解析成 Agent 引擎的决策格式。

★ 这一层存在的理由（也是它必须独立单测的原因）：

  `core/agent.py` 的 `act_fn` 契约只有两个形状：
      {"tool": "kb_search", "args": {...}}      -> 去调工具
      {"final": "答案", "citations": [...]}      -> 收尾作答

  但真实模型通过 function calling 返回的是 OpenAI 结构：
      {"choices":[{"message":{"tool_calls":[{"function":{"name":"kb_search",
                    "arguments":"{\\"query\\": \\"x\\"}"}}]}}]}

  中间这段"翻译"是**最容易出错、也最该被测试**的地方，因为它要处理：
    ① arguments 是**字符串**（不是 dict），且模型可能给非法 JSON
    ② 模型可能一次给多个 tool_calls（这里只取第一个，其余丢弃并记日志）
    ③ 模型可能调一个**不存在**或**不在白名单**的工具（幻觉调用）
    ④ 模型可能既不调工具也不给 final（空回复 / 只输出思考过程）
    ⑤ 模型可能把答案塞进 content 同时带 tool_calls（要优先信 tool_calls）

  把这些分支都测一遍，才能真正说"Agent 接上真模型是可用的"。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

# 模型在 content 里表达"我要收尾了"的常见前缀，用于兜底解析
_FINAL_JSON_KEYS = ("final", "answer", "response")


class DecisionError(Exception):
    """无法从模型输出里解析出任何可用决策。"""


def parse_arguments(raw: Any) -> Dict[str, Any]:
    """解析 tool_call 的 arguments。

    模型可能给：dict（少见）、合法 JSON 字符串（常见）、非法 JSON、
    空字符串、None。全部要能安全处理 —— 解析失败返回空 dict，
    让参数校验层去报"缺少必填参数"，而不是在这里抛异常打断整轮。
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        return {"value": list(raw)}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 模型偶尔会写成单引号或带尾逗号：给它一次宽松修正的机会
            repaired = _repair_json(text)
            if repaired is None:
                return {}
            parsed = repaired
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {"value": raw}


def _repair_json(text: str) -> Optional[Any]:
    """对常见 JSON 瑕疵做一次修正：单引号、尾逗号、```json 包裹。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    t = t.replace("'", '"')
    # 去掉对象/数组结尾的多余逗号
    import re
    t = re.sub(r",\s*([}\]])", r"\1", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """从 OpenAI 兼容响应里取出 tool_calls（已归一化）。

    返回 [{"name": str, "args": dict, "id": str}]。
    兼容四种输入：
      ① 完整响应 {"choices":[{"message":{"tool_calls":[...]}}]}
      ② message dict {"tool_calls":[...]}
      ③ 已是 tool_calls 列表 [{"function":{...}}]
      ④ 带 to_dict() 的对象（如 openai SDK 的响应对象）

    ★ 踩坑记录：初版只处理 ① 和 ③，把 ② 误当成 ① 处理 ——
      `response.get("choices")` 返回 None，于是 message 仍是整个 dict，
      再取 `message.get("tool_calls")` 其实能拿到，但当时我写成了
      「有 choices 才更新 message」，导致 ② 走不到更新分支而返回空。
      简化成"先判断有没有 choices，有就下钻，没有就当成 message"就没这问题。
    """
    if response is None:
        return []

    # 已是 tool_calls 列表
    if isinstance(response, list):
        calls = response
    else:
        if not isinstance(response, dict):
            to_dict = getattr(response, "to_dict", None)
            if callable(to_dict):
                response = to_dict()
            elif hasattr(response, "model_dump"):
                response = response.model_dump()
            else:
                response = getattr(response, "__dict__", {}) or {}

        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            message = first.get("message") or first.get("delta") or {}
        elif "tool_calls" not in response and isinstance(response.get("message"), dict):
            # 少一层的变体 {"message": {...}}（有些封装库会这样返回）
            message = response["message"]
        else:
            # 本身就是 message（或含 tool_calls 的容器）
            message = response
        calls = message.get("tool_calls") or []

    out: List[Dict[str, Any]] = []
    for c in calls or []:
        if not isinstance(c, dict):
            continue
        fn = c.get("function") or {}
        name = fn.get("name") or c.get("name")
        if not name:
            continue
        out.append({
            "name": str(name),
            "args": parse_arguments(fn.get("arguments", c.get("arguments"))),
            "id": str(c.get("id") or ""),
        })
    return out


def extract_text(response: Any) -> str:
    """从响应里取出纯文本 content。"""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        response = getattr(response, "__dict__", {}) or {}
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] or {}
        message = first.get("message") or first.get("delta") or {}
        return str(message.get("content") or "")
    return str(response.get("content") or "")


def parse_final_from_text(text: str) -> Optional[Dict[str, Any]]:
    """当模型没走 function calling、而是把答案写在 content 里时的兜底。

    支持两种形态：
      ① content 是 JSON：{"final": "...", "citations": [...]}
      ② content 是普通文本：整段作为 final
    """
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("{"):
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            obj = _repair_json(t)
        if isinstance(obj, dict):
            for key in _FINAL_JSON_KEYS:
                if key in obj:
                    return {"final": str(obj[key]),
                            "citations": list(obj.get("citations") or [])}
    return {"final": t, "citations": []}


def parse_decision(
    response: Any,
    allowed_tools: Sequence[str],
    known_tools: Sequence[str],
    round_index: int = 1,
    max_tool_calls: int = 1,
) -> Dict[str, Any]:
    """把模型响应转成 `core/agent.py` 的 act_fn 决策格式。

    优先级：**tool_calls > content**。
    理由：模型经常在调工具的同时写一句"我来查一下"——如果优先信 content，
    就会把这句话当成最终答案，任务直接断在这一轮。

    Args:
        allowed_tools: 当前任务白名单（越权调用会被这里提前拦掉，
            让 agent 引擎把它记成 blocked 事件而不是真的执行）
        max_tool_calls: 单轮最多执行几个工具调用（多余丢弃并记录）
    """
    allowed = list(allowed_tools or [])
    known = set(known_tools or [])
    text = extract_text(response)
    calls = extract_tool_calls(response)

    # ---------- 分支 1：模型调了工具 ----------
    if calls:
        chosen = calls[0]
        extra = calls[1:]
        name = chosen["name"]
        decision: Dict[str, Any] = {
            "tool": name,
            "args": chosen["args"],
            "_round": round_index,
        }
        # 幻觉工具：不在注册表里。仍然交给引擎 -> 会被白名单/ToolNotFound 拦下，
        # 并产生一条可观测的 tool_error 事件（比在这里静默丢弃更有价值）
        if name not in known:
            decision["_note"] = f"模型调用了未注册的工具 {name!r}"
        elif name not in allowed:
            decision["_note"] = f"工具 {name!r} 不在当前任务白名单内"
        if extra:
            decision["_dropped_calls"] = [c["name"] for c in extra]
        return decision

    # ---------- 分支 2：模型给了文本 ----------
    final = parse_final_from_text(text)
    if final is not None:
        final["_round"] = round_index
        return final

    # ---------- 分支 3：什么都没有 ----------
    raise DecisionError(f"模型既未调用工具也未给出回答（round={round_index}）")


# --------------------------------------------------------------------- 引用解析
_CITATION_RE = None


def _citation_pattern():
    global _CITATION_RE
    if _CITATION_RE is None:
        import re
        _CITATION_RE = re.compile(r"\[(\d{1,2})\]")
    return _CITATION_RE


def collect_hit_ids(state: Any) -> List[str]:
    """按工具调用顺序收集所有 hits 的 id（去掉重复）。

    ★ 这个顺序就是模型看到的 [1][2][3] 的顺序 —— 因为 prompt 里注入工具结果时
      也是按同一顺序排列的（`format_scratchpad` 逐条列）。所以下标可以直接对应。
    """
    ids: List[str] = []
    for item in getattr(state, "scratchpad", []) or []:
        if not (item.get("type") == "tool_call" and item.get("ok")):
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        for hit in result.get("hits") or []:
            if isinstance(hit, dict):
                hid = hit.get("id")
                if hid and str(hid) not in ids:
                    ids.append(str(hid))
    return ids


def resolve_citations(answer: str, state: Any) -> List[str]:
    """把回答里的 `[1] [2]` 解析成真实 chunk_id。

    ★ 为什么必须做这一步（被 demo_agent2.py 抓到）：
      模型在纯文本回答里写 `[1] [2]`，但它不会告诉你 1 和 2 分别是哪个 chunk。
      如果不解析，`state.citations` 就是空的 —— 前端渲染不出可点击来源，
      "引用溯源"这个卖点等于没实现。

    下标规则：`[n]` 对应"按工具调用顺序拼接后的第 n 个 hit"。
    越界或没有引用时返回空列表（宁可不给引用，也不给错引用）。
    """
    if not answer:
        return []
    hit_ids = collect_hit_ids(state)
    if not hit_ids:
        return []
    cited: List[str] = []
    for m in _citation_pattern().finditer(answer):
        idx = int(m.group(1))
        if 1 <= idx <= len(hit_ids):
            cid = hit_ids[idx - 1]
            if cid not in cited:
                cited.append(cid)
    return cited


# --------------------------------------------------------------------- 便捷封装
class ToolCallingDecider:
    """把 LLMClient + 解析器组合成一个可直接当 act_fn 用的可调用对象。

    用法：
        decider = ToolCallingDecider(llm=llm_client, registry=registry,
                                     build_messages=my_builder)
        engine = AgentEngine(..., act_fn=decider)

    ★ 关键：`build_messages` 由你注入（含记忆、检索上下文、工具列表），
      这样"怎么组装 prompt"与"怎么解析决策"解耦，两者都能独立测。
    """

    def __init__(
        self,
        llm: Any,
        registry: Any,
        build_messages,
        max_tool_calls_per_round: int = 1,
        temperature: float = 0.0,
        fallback_on_error: bool = True,
        supports_tools: Optional[bool] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.build_messages = build_messages
        self.max_tool_calls_per_round = max_tool_calls_per_round
        self.temperature = temperature
        self.fallback_on_error = fallback_on_error
        # ★ 只在初始化时探测一次"客户端是否支持 tools 参数"，
        #   不要写成 `try: chat(tools=...) except TypeError: chat(...)`——
        #   那个写法会把**函数体内部**抛出的任何 TypeError 也误判成
        #   "不支持 tools"，然后带着错误的假设重试一次，掩盖真实 bug。
        #   这个坑是被 test_llm_without_tools_param_still_works 连带暴露的。
        self.supports_tools = (supports_tools if supports_tools is not None
                               else self._detect_tools_support())
        self.stats = {"rounds": 0, "tool_decisions": 0, "final_decisions": 0,
                      "errors": 0, "multi_call_rounds": 0}

    def _detect_tools_support(self) -> bool:
        """用 inspect 判断 chat/transport 是否接受 tools 关键字。"""
        import inspect

        target = self.llm
        fn = getattr(target, "chat", None) or target
        transport = getattr(target, "_transport", None)
        if transport is not None and callable(transport):
            fn = transport
        if not callable(fn):
            return False
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        for param in sig.parameters.values():
            if param.name == "tools":
                return True
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    def __call__(self, state: Any, allowed_tools: List[str]) -> Dict[str, Any]:
        self.stats["rounds"] += 1
        messages = self.build_messages(state, allowed_tools)
        tools = self.registry.openai_tools(allowed=allowed_tools) or None

        kwargs: Dict[str, Any] = {"temperature": self.temperature}
        if self.supports_tools and tools:
            kwargs["tools"] = tools
        response = self.llm.chat(messages, **kwargs)

        raw = getattr(response, "raw", None) or {}
        # LLMResult.raw 里是原始响应；没有就用文本兜底（例如注入的假客户端）
        if not raw:
            raw = {"choices": [{"message": {"content": getattr(response, "text", "")}}]}

        try:
            decision = parse_decision(
                raw,
                allowed_tools=allowed_tools,
                known_tools=self.registry.names(),
                round_index=state.rounds,
                max_tool_calls=self.max_tool_calls_per_round,
            )
        except DecisionError as exc:
            self.stats["errors"] += 1
            if not self.fallback_on_error:
                raise
            # 兜底：把已收集到的工具观察结果拼成回答，避免整轮白跑
            decision = self._fallback_final(state, str(exc))

        # ★ 补齐引用：模型只写 [1][2]，需要反查成真实 chunk_id
        if "final" in decision and not decision.get("citations"):
            decision["citations"] = resolve_citations(decision["final"], state)

        if decision.get("_dropped_calls"):
            self.stats["multi_call_rounds"] += 1
        if decision.get("tool"):
            self.stats["tool_decisions"] += 1
        else:
            self.stats["final_decisions"] += 1
        return decision

    @staticmethod
    def _fallback_final(state: Any, reason: str) -> Dict[str, Any]:
        """解析失败时的兜底回答：基于已有工具结果拼一段，并标注原因。"""
        observations: List[str] = []
        citations: List[str] = []
        for item in getattr(state, "scratchpad", []) or []:
            if item.get("type") == "tool_call" and item.get("ok"):
                tool = item.get("tool", "tool")
                result = item.get("result")
                text = result if isinstance(result, str) else json.dumps(
                    result, ensure_ascii=False)
                observations.append(f"[{tool}] {text[:400]}")
                for hit in (result or {}).get("hits", []) if isinstance(result, dict) else []:
                    if isinstance(hit, dict) and hit.get("id"):
                        citations.append(str(hit["id"]))
        body = "\n".join(observations) if observations else "（本轮未获得有效工具结果）"
        return {
            "final": f"模型输出无法解析（{reason}），以下为已检索到的原始结果：\n{body}",
            "citations": citations,
        }
