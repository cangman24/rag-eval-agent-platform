# -*- coding: utf-8 -*-
"""tests/test_decisions.py —— 真模型决策解析 + messages 组装单测。全部离线。

重点验证"接真模型"这条路径上最容易出错的边界：
  ① arguments 是字符串且可能是非法 JSON
  ② 模型一次给多个 tool_calls
  ③ 模型调了不存在 / 不在白名单的工具（幻觉调用）
  ④ 模型只输出文本不调工具
  ⑤ 模型既写话又调工具（必须优先信 tool_calls）
  ⑥ 解析彻底失败时的兜底回答
  ⑦ prompt 里只出现白名单工具、超长工具结果被截断
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import AgentEngine, AgentState, EventType  # noqa: E402
from core.mcp import ToolRegistry  # noqa: E402
from rag2_kb_agent.decisions import (  # noqa: E402
    DecisionError, ToolCallingDecider, collect_hit_ids, extract_text,
    extract_tool_calls, parse_arguments, parse_decision, parse_final_from_text,
    resolve_citations,
)
from rag2_kb_agent.prompts import (  # noqa: E402
    MessageBuilder, format_memory_block, format_scratchpad, format_tool_list,
    messages_token_estimate,
)


# ================================================================== 构造响应
def resp_tool_call(name: str, arguments, call_id: str = "call_1", content=None):
    """构造 OpenAI 兼容的 function calling 响应。"""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str)
                        else json.dumps(arguments, ensure_ascii=False),
                    },
                }],
            }
        }]
    }


def resp_text(text: str):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def resp_multi(calls):
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": f"c{i}", "type": "function",
                     "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                    for i, (n, a) in enumerate(calls)
                ],
            }
        }]
    }


def build_registry():
    reg = ToolRegistry()

    @reg.tool(description="检索知识库")
    def kb_search(query: str, top_k: int = 3):
        """检索知识库

        query: 检索词
        top_k: 返回条数
        """
        return {"count": 1, "hits": [{"id": "c1", "text": "精度要求 0.90"}]}

    @reg.tool(description="查询工单")
    def ticket_query(ticket_id: str):
        """查询工单

        ticket_id: 工单号
        """
        return {"ticket_id": ticket_id, "status": "closed"}

    return reg


KNOWN = ["kb_search", "ticket_query"]


# ================================================================== arguments
class TestParseArguments(unittest.TestCase):
    def test_json_string(self):
        self.assertEqual(parse_arguments('{"query": "x"}'), {"query": "x"})

    def test_already_dict(self):
        self.assertEqual(parse_arguments({"query": "x"}), {"query": "x"})

    def test_none_and_empty(self):
        self.assertEqual(parse_arguments(None), {})
        self.assertEqual(parse_arguments(""), {})
        self.assertEqual(parse_arguments("   "), {})

    def test_invalid_json_returns_empty(self):
        """★ 非法 JSON 不能抛异常打断整轮，交给参数校验层报错。"""
        self.assertEqual(parse_arguments("{not json"), {})

    def test_single_quotes_repaired(self):
        self.assertEqual(parse_arguments("{'query': 'x'}"), {"query": "x"})

    def test_trailing_comma_repaired(self):
        self.assertEqual(parse_arguments('{"query": "x",}'), {"query": "x"})

    def test_fenced_json_repaired(self):
        self.assertEqual(parse_arguments('```json\n{"query": "x"}\n```'),
                         {"query": "x"})

    def test_non_dict_json_wrapped(self):
        self.assertEqual(parse_arguments("[1,2]"), {"value": [1, 2]})

    def test_non_string_passthrough(self):
        self.assertEqual(parse_arguments(42), {"value": 42})


# ================================================================== tool_calls
class TestExtractToolCalls(unittest.TestCase):
    def test_full_response(self):
        calls = extract_tool_calls(resp_tool_call("kb_search", {"query": "x"}))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "kb_search")
        self.assertEqual(calls[0]["args"], {"query": "x"})

    def test_bare_message_dict(self):
        """message 本体（无 choices 包裹）也要能解析。"""
        msg = {"role": "assistant",
               "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]}
        self.assertEqual(extract_tool_calls(msg)[0]["name"], "t")

    def test_wrapped_message_dict(self):
        """{"message": {...}} 这种少一层的变体也应容错。"""
        wrapped = {"message": {"tool_calls": [
            {"function": {"name": "t", "arguments": "{}"}}]}}
        self.assertEqual(extract_tool_calls(wrapped)[0]["name"], "t")

    def test_object_with_to_dict(self):
        class Resp:
            def to_dict(self):
                return resp_tool_call("kb_search", {"query": "x"})
        self.assertEqual(extract_tool_calls(Resp())[0]["name"], "kb_search")

    def test_bare_tool_calls_list(self):
        calls = [{"function": {"name": "t", "arguments": '{"a":1}'}}]
        self.assertEqual(extract_tool_calls(calls)[0]["args"], {"a": 1})

    def test_no_tool_calls(self):
        self.assertEqual(extract_tool_calls(resp_text("普通回答")), [])
        self.assertEqual(extract_tool_calls(None), [])
        self.assertEqual(extract_tool_calls({}), [])

    def test_call_without_name_skipped(self):
        calls = [{"function": {"arguments": "{}"}}]
        self.assertEqual(extract_tool_calls(calls), [])

    def test_multiple(self):
        calls = extract_tool_calls(resp_multi([("a", {}), ("b", {})]))
        self.assertEqual([c["name"] for c in calls], ["a", "b"])


class TestExtractText(unittest.TestCase):
    def test_from_response(self):
        self.assertEqual(extract_text(resp_text("你好")), "你好")

    def test_from_string(self):
        self.assertEqual(extract_text("直传"), "直传")

    def test_empty(self):
        self.assertEqual(extract_text(None), "")
        self.assertEqual(extract_text({}), "")

    def test_none_content(self):
        self.assertEqual(extract_text({"choices": [{"message": {"content": None}}]}), "")


# ================================================================== final 文本
class TestParseFinalFromText(unittest.TestCase):
    def test_plain_text(self):
        out = parse_final_from_text("这是回答")
        self.assertEqual(out["final"], "这是回答")

    def test_json_with_final(self):
        out = parse_final_from_text('{"final": "答案", "citations": ["c1"]}')
        self.assertEqual(out["final"], "答案")
        self.assertEqual(out["citations"], ["c1"])

    def test_json_with_answer_key(self):
        out = parse_final_from_text('{"answer": "甲"}')
        self.assertEqual(out["final"], "甲")

    def test_empty_returns_none(self):
        self.assertIsNone(parse_final_from_text(""))
        self.assertIsNone(parse_final_from_text("   "))

    def test_json_without_final_key_falls_back_to_text(self):
        out = parse_final_from_text('{"foo": 1}')
        self.assertIn("foo", out["final"])


# ================================================================== parse_decision
class TestParseDecision(unittest.TestCase):
    def test_tool_call(self):
        d = parse_decision(resp_tool_call("kb_search", {"query": "x"}),
                           allowed_tools=["kb_search"], known_tools=KNOWN)
        self.assertEqual(d["tool"], "kb_search")
        self.assertEqual(d["args"], {"query": "x"})
        self.assertNotIn("_note", d)

    def test_final_text(self):
        d = parse_decision(resp_text("答案 [1]"), allowed_tools=[], known_tools=KNOWN)
        self.assertEqual(d["final"], "答案 [1]")

    def test_final_json(self):
        d = parse_decision(resp_text('{"final": "答案", "citations": ["c1"]}'),
                           allowed_tools=[], known_tools=KNOWN)
        self.assertEqual(d["citations"], ["c1"])

    def test_tool_calls_win_over_content(self):
        """★ 关键：模型边说"我来查一下"边调工具时，必须优先信 tool_calls。

        否则那句"我来查一下"会被当成最终答案，任务直接断在这一轮。
        """
        d = parse_decision(
            resp_tool_call("kb_search", {"query": "x"}, content="我来查一下相关资料。"),
            allowed_tools=["kb_search"], known_tools=KNOWN)
        self.assertIn("tool", d)
        self.assertNotIn("final", d)

    def test_hallucinated_tool_flagged_not_dropped(self):
        """★ 调了未注册的工具 -> 打标记但仍交给引擎（产生可观测的拦截事件）。"""
        d = parse_decision(resp_tool_call("web_search", {"q": "x"}),
                           allowed_tools=[], known_tools=KNOWN)
        self.assertEqual(d["tool"], "web_search")
        self.assertIn("未注册", d["_note"])

    def test_out_of_whitelist_flagged(self):
        d = parse_decision(resp_tool_call("ticket_query", {"ticket_id": "T"}),
                           allowed_tools=["kb_search"], known_tools=KNOWN)
        self.assertEqual(d["tool"], "ticket_query")
        self.assertIn("白名单", d["_note"])

    def test_multiple_calls_takes_first_and_records_rest(self):
        d = parse_decision(resp_multi([("kb_search", {"query": "x"}),
                                       ("ticket_query", {"ticket_id": "T"})]),
                           allowed_tools=KNOWN, known_tools=KNOWN)
        self.assertEqual(d["tool"], "kb_search")
        self.assertEqual(d["_dropped_calls"], ["ticket_query"])

    def test_empty_response_raises(self):
        with self.assertRaises(DecisionError):
            parse_decision({}, allowed_tools=[], known_tools=KNOWN)

    def test_none_content_and_no_calls_raises(self):
        with self.assertRaises(DecisionError):
            parse_decision({"choices": [{"message": {"content": None,
                                                     "tool_calls": []}}]},
                           allowed_tools=[], known_tools=KNOWN)

    def test_round_index_attached(self):
        d = parse_decision(resp_text("答案"), allowed_tools=[], known_tools=KNOWN,
                           round_index=3)
        self.assertEqual(d["_round"], 3)


# ================================================================== decider
class FakeLLMResponse:
    def __init__(self, raw, text=""):
        self.raw = raw
        self.text = text


class FakeLLM:
    """按脚本逐个返回响应。

    support_tools=False 时**签名里根本没有 tools 参数** —— 用来验证
    `ToolCallingDecider` 是靠 inspect 探测而不是靠 except TypeError 兜底。
    """

    def __init__(self, responses, support_tools=True):
        self.responses = list(responses)
        self.support_tools = support_tools
        self.calls = []
        self.seen_tools = []

    def chat(self, messages, temperature=0.0, **kw):
        if self.support_tools:
            return self._respond(messages, kw.get("tools"))
        if "tools" in kw:
            # 不支持 tools 的客户端收到 tools 就该报错；
            # decider 不该把这个错吞掉。
            raise TypeError("unexpected keyword argument 'tools'")
        return self._respond(messages, None)

    def _respond(self, messages, tools):
        self.calls.append(messages)
        self.seen_tools.append(tools)
        raw = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        return FakeLLMResponse(raw, extract_text(raw))


class TestToolCallingDecider(unittest.TestCase):
    def _decider(self, responses, support_tools=True, **kw):
        reg = build_registry()
        llm = FakeLLM(responses, support_tools)
        builder = lambda state, allowed: [{"role": "user", "content": state.query}]
        return ToolCallingDecider(llm=llm, registry=reg, build_messages=builder, **kw), llm

    def test_returns_tool_decision(self):
        dec, llm = self._decider([resp_tool_call("kb_search", {"query": "x"})])
        state = AgentState(query="查精度", rounds=1)
        d = dec(state, ["kb_search"])
        self.assertEqual(d["tool"], "kb_search")
        self.assertEqual(dec.stats["tool_decisions"], 1)

    def test_returns_final_decision(self):
        dec, _ = self._decider([resp_text("答案 [1]")])
        d = dec(AgentState(query="q", rounds=1), [])
        self.assertEqual(d["final"], "答案 [1]")
        self.assertEqual(dec.stats["final_decisions"], 1)

    def test_passes_whitelisted_tools_only(self):
        """★ 注入给模型的 tools 必须只是白名单子集。"""
        dec, llm = self._decider([resp_text("ok")])
        dec(AgentState(query="q", rounds=1), ["kb_search"])
        names = [t["function"]["name"] for t in (llm.seen_tools[0] or [])]
        self.assertEqual(names, ["kb_search"])

    def test_no_tools_when_whitelist_empty(self):
        dec, llm = self._decider([resp_text("ok")])
        dec(AgentState(query="q", rounds=1), [])
        # 空白名单 -> 不传 tools（None 或空列表都算对）
        self.assertFalse(llm.seen_tools[0])

    def test_llm_without_tools_param_still_works(self):
        """★ 兼容不支持 tools 的客户端：回退到纯文本解析。"""
        dec, _ = self._decider([resp_text("兜底答案")], support_tools=False)
        d = dec(AgentState(query="q", rounds=1), [])
        self.assertEqual(d["final"], "兜底答案")

    def test_unparseable_output_falls_back(self):
        dec, _ = self._decider([{"choices": [{"message": {"content": None}}]}])
        d = dec(AgentState(query="q", rounds=1), [])
        self.assertIn("final", d)
        self.assertIn("无法解析", d["final"])
        self.assertEqual(dec.stats["errors"], 1)

    def test_fallback_includes_tool_observations(self):
        """★ 兜底回答要带上已获得的工具结果，避免整轮白跑。"""
        dec, _ = self._decider([{"choices": [{"message": {"content": None}}]}])
        state = AgentState(query="q", rounds=2)
        state.scratchpad = [{"type": "tool_call", "tool": "kb_search", "ok": True,
                             "result": {"count": 1,
                                        "hits": [{"id": "c1", "text": "精度 0.90"}]}}]
        d = dec(state, [])
        self.assertIn("kb_search", d["final"])
        self.assertIn("c1", d["citations"])

    def test_fallback_can_be_disabled(self):
        dec, _ = self._decider([{}], fallback_on_error=False)
        with self.assertRaises(DecisionError):
            dec(AgentState(query="q", rounds=1), [])

    def test_stats_accumulate(self):
        dec, _ = self._decider([resp_tool_call("kb_search", {"query": "x"}),
                                resp_text("答案")])
        state = AgentState(query="q", rounds=1)
        state.rounds = 1
        dec(state, ["kb_search"])
        state.rounds = 2
        dec(state, ["kb_search"])
        self.assertEqual(dec.stats["rounds"], 2)
        self.assertEqual(dec.stats["tool_decisions"], 1)
        self.assertEqual(dec.stats["final_decisions"], 1)


# ================================================================== 端到端：真决策接引擎
class TestEngineWithRealDecider(unittest.TestCase):
    def test_full_agent_loop_with_function_calling(self):
        """★ 端到端：AgentEngine + ToolCallingDecider，走完整工具调用循环。"""
        reg = build_registry()
        llm = FakeLLM([
            resp_tool_call("kb_search", {"query": "精度要求"}),
            resp_text("精度要求 mAP 不低于 0.90 [1]"),
        ])
        builder = lambda state, allowed: [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": state.query},
        ]
        decider = ToolCallingDecider(llm=llm, registry=reg, build_messages=builder)
        engine = AgentEngine(
            registry=reg,
            intent_fn=lambda q: "spec",
            plan_fn=lambda q, t: ["检索", "汇总"],
            act_fn=decider,
            route_table={"spec": ["kb_search"]},
            max_rounds=4,
        )
        run = engine.run("精度要求是多少")
        self.assertTrue(run.ok, run.error)
        self.assertEqual(run.state.tool_calls, 1)
        self.assertIn("0.90", run.state.answer)
        types = [e["type"] for e in run.events]
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.ANSWER, types)

    def test_hallucinated_tool_gets_blocked_and_loop_continues(self):
        """★ 模型调了白名单外的工具 -> 引擎拦截 -> 下一轮模型改口作答。"""
        reg = build_registry()
        llm = FakeLLM([
            resp_tool_call("ticket_query", {"ticket_id": "T-1"}),   # 越权
            resp_text("我无权查询工单，仅能回答制度问题。"),
        ])
        builder = lambda state, allowed: [{"role": "user", "content": state.query}]
        decider = ToolCallingDecider(llm=llm, registry=reg, build_messages=builder)
        engine = AgentEngine(
            registry=reg, intent_fn=lambda q: "spec",
            plan_fn=lambda q, t: ["检索"], act_fn=decider,
            route_table={"spec": ["kb_search"]}, max_rounds=4,
        )
        run = engine.run("查一下工单 T-1")
        blocked = [e for e in run.events
                   if e["type"] == EventType.TOOL_ERROR and e["data"].get("blocked")]
        self.assertEqual(len(blocked), 1)
        self.assertEqual(run.state.tool_calls, 0)
        self.assertTrue(run.state.finished)


class TestCitationResolution(unittest.TestCase):
    """★ 引用解析：模型只写 [1][2]，必须反查成真实 chunk_id。

    （demo_agent2.py 曾暴露此问题：回答里明明有 [1][2]，但 citations 是空的，
     前端渲染不出可点击来源，"引用溯源"等于没实现。）
    """

    def _state(self, hits_per_call):
        state = AgentState(query="q")
        state.scratchpad = [
            {"type": "tool_call", "tool": "kb_search", "ok": True,
             "result": {"count": len(hits), "hits": hits}}
            for hits in hits_per_call
        ]
        return state

    def test_collect_hit_ids_in_call_order(self):
        state = self._state([
            [{"id": "c1"}, {"id": "c2"}],
            [{"id": "c3"}],
        ])
        self.assertEqual(collect_hit_ids(state), ["c1", "c2", "c3"])

    def test_collect_dedups(self):
        state = self._state([[{"id": "c1"}], [{"id": "c1"}]])
        self.assertEqual(collect_hit_ids(state), ["c1"])

    def test_collect_ignores_failed_calls(self):
        state = AgentState()
        state.scratchpad = [
            {"type": "tool_call", "tool": "t", "ok": False, "error": "x"},
            {"type": "tool_call", "tool": "t", "ok": True,
             "result": {"hits": [{"id": "c9"}]}},
        ]
        self.assertEqual(collect_hit_ids(state), ["c9"])

    def test_resolve_maps_indices(self):
        state = self._state([[{"id": "c1"}, {"id": "c2"}], [{"id": "c3"}]])
        self.assertEqual(resolve_citations("结论甲 [1]，结论乙 [3]", state),
                         ["c1", "c3"])

    def test_resolve_ignores_out_of_range(self):
        state = self._state([[{"id": "c1"}]])
        self.assertEqual(resolve_citations("引用 [5]", state), [])

    def test_resolve_no_citations_in_answer(self):
        state = self._state([[{"id": "c1"}]])
        self.assertEqual(resolve_citations("没有任何标注", state), [])

    def test_resolve_no_hits(self):
        self.assertEqual(resolve_citations("答案 [1]", AgentState()), [])

    def test_resolve_empty_answer(self):
        state = self._state([[{"id": "c1"}]])
        self.assertEqual(resolve_citations("", state), [])

    def test_resolve_dedups_repeated_refs(self):
        state = self._state([[{"id": "c1"}]])
        self.assertEqual(resolve_citations("甲 [1] 乙 [1]", state), ["c1"])

    def test_decider_fills_citations(self):
        """★ 端到端：decider 在给 final 时必须自动补上 citations。"""
        reg = build_registry()
        llm = FakeLLM([
            resp_tool_call("kb_search", {"query": "精度"}),
            resp_text("精度要求 0.90 [1]"),
        ])
        builder = lambda state, allowed: [{"role": "user", "content": state.query}]
        decider = ToolCallingDecider(llm=llm, registry=reg, build_messages=builder)
        engine = AgentEngine(
            registry=reg, intent_fn=lambda q: "spec",
            plan_fn=lambda q, t: ["检索"], act_fn=decider,
            route_table={"spec": ["kb_search"]}, max_rounds=4,
        )
        run = engine.run("精度要求")
        self.assertTrue(run.state.citations, "应自动解析出引用")
        self.assertIn("c1", run.state.citations)

    def test_decider_keeps_model_supplied_citations(self):
        """模型若显式给了 citations（JSON 形式），不要被覆盖。"""
        reg = build_registry()
        llm = FakeLLM([resp_text('{"final": "答案", "citations": ["custom"]}')])
        builder = lambda state, allowed: [{"role": "user", "content": state.query}]
        decider = ToolCallingDecider(llm=llm, registry=reg, build_messages=builder)
        d = decider(AgentState(query="q", rounds=1), [])
        self.assertEqual(d["citations"], ["custom"])


# ================================================================== prompts
class TestFormatToolList(unittest.TestCase):
    def test_lists_name_params_and_required(self):
        reg = build_registry()
        text = format_tool_list(reg.openai_tools())
        self.assertIn("kb_search(query*, top_k)", text)
        self.assertIn("必填", text)

    def test_empty_tools_message(self):
        self.assertIn("无可用工具", format_tool_list([]))


class TestFormatMemoryBlock(unittest.TestCase):
    def test_full_context(self):
        block = format_memory_block({
            "profile": "角色=风控",
            "episodic": [{"summary": "用户在查工单流程"}],
        })
        self.assertIn("角色=风控", block)
        self.assertIn("工单流程", block)

    def test_empty_returns_blank(self):
        self.assertEqual(format_memory_block(None), "")
        self.assertEqual(format_memory_block({}), "")
        self.assertEqual(format_memory_block({"profile": "", "episodic": []}), "")

    def test_caps_episodic(self):
        block = format_memory_block(
            {"episodic": [{"summary": f"s{i}"} for i in range(10)]}, max_episodic=2)
        self.assertIn("s0", block)
        self.assertIn("s1", block)
        self.assertNotIn("s2", block)


class TestFormatScratchpad(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(format_scratchpad(AgentState()), "")

    def test_recent_detail_older_summarized(self):
        state = AgentState()
        state.scratchpad = [
            {"type": "tool_call", "tool": f"t{i}", "ok": True,
             "result": {"hits": [{"id": f"c{i}"}]}}
            for i in range(6)
        ]
        text = format_scratchpad(state, recent_rounds=2)
        self.assertIn("更早的 4 次", text)
        self.assertIn("t4", text)
        self.assertIn("t5", text)
        self.assertNotIn("t0", text)

    def test_result_truncated(self):
        state = AgentState()
        state.scratchpad = [{"type": "tool_call", "tool": "t", "ok": True,
                             "result": "x" * 5000}]
        text = format_scratchpad(state, max_chars_per_result=100)
        self.assertIn("已截断", text)
        self.assertLess(len(text), 1000)

    def test_failed_call_shown(self):
        state = AgentState()
        state.scratchpad = [{"type": "tool_call", "tool": "t", "ok": False,
                             "error": "超时"}]
        self.assertIn("失败", format_scratchpad(state))

    def test_blocked_call_shown(self):
        state = AgentState()
        state.scratchpad = [{"type": "tool_call", "tool": "t", "blocked": True,
                             "error": "不在白名单"}]
        self.assertIn("被拒绝", format_scratchpad(state))


class TestMessageBuilder(unittest.TestCase):
    def _builder(self, **kw):
        return MessageBuilder(registry=build_registry(), **kw)

    def test_only_whitelisted_tools_in_prompt(self):
        """★ 安全要求：system prompt 里不能出现白名单外的工具。"""
        msgs = self._builder().build(AgentState(query="q"), ["kb_search"])
        system = msgs[0]["content"]
        self.assertIn("kb_search", system)
        self.assertNotIn("ticket_query", system)

    def test_task_and_plan_in_user_message(self):
        state = AgentState(query="查精度", plan=["检索", "汇总"])
        msgs = self._builder().build(state, ["kb_search"])
        user = msgs[-1]["content"]
        self.assertIn("查精度", user)
        self.assertIn("检索", user)

    def test_scratchpad_injected_when_present(self):
        state = AgentState(query="q")
        state.scratchpad = [{"type": "tool_call", "tool": "kb_search", "ok": True,
                             "result": {"hits": [{"id": "c1"}]}}]
        user = self._builder().build(state, ["kb_search"])[-1]["content"]
        self.assertIn("已执行的工具调用", user)
        self.assertIn("kb_search", user)

    def test_memory_block_injected(self):
        builder = self._builder(memory_provider=lambda q: {
            "profile": "角色=工程师",
            "episodic": [{"summary": "历史摘要"}],
            "recent": [{"role": "user", "content": "之前问过精度"}],
        })
        msgs = builder.build(AgentState(query="q"), ["kb_search"])
        self.assertIn("角色=工程师", msgs[0]["content"])
        self.assertTrue(any(m["role"] == "user" and "之前问过精度" in m["content"]
                            for m in msgs))

    def test_broken_memory_provider_does_not_crash(self):
        def boom(q):
            raise RuntimeError("记忆层挂了")
        builder = self._builder(memory_provider=boom)
        msgs = builder.build(AgentState(query="q"), ["kb_search"])
        self.assertTrue(msgs)

    def test_no_memory_provider(self):
        msgs = self._builder().build(AgentState(query="q"), ["kb_search"])
        self.assertNotIn("已知背景", msgs[0]["content"])

    def test_system_extra_appended(self):
        msgs = self._builder(system_extra="额外要求：必须中文回答").build(
            AgentState(query="q"), [])
        self.assertIn("必须中文回答", msgs[0]["content"])

    def test_token_estimate_bounded(self):
        """★ 长轨迹经过截断后，prompt 规模必须有上界。"""
        state = AgentState(query="q")
        state.scratchpad = [
            {"type": "tool_call", "tool": "kb_search", "ok": True,
             "result": "x" * 3000} for _ in range(20)
        ]
        msgs = self._builder(max_tool_result_chars=200, recent_rounds=2).build(
            state, ["kb_search"])
        self.assertLess(messages_token_estimate(msgs), 2000)

    def test_callable(self):
        builder = self._builder()
        msgs = builder(AgentState(query="q"), [])
        self.assertEqual(msgs[0]["role"], "system")


class TestPromptsWithRealEngine(unittest.TestCase):
    def test_builder_used_by_decider_end_to_end(self):
        reg = build_registry()
        llm = FakeLLM([resp_tool_call("kb_search", {"query": "精度"}),
                       resp_text("答案 [1]")])
        builder = MessageBuilder(registry=reg)
        decider = ToolCallingDecider(llm=llm, registry=reg, build_messages=builder)
        engine = AgentEngine(registry=reg, intent_fn=lambda q: "spec",
                             plan_fn=lambda q, t: ["检索"], act_fn=decider,
                             route_table={"spec": ["kb_search"]}, max_rounds=4)
        run = engine.run("精度要求")
        self.assertTrue(run.ok, run.error)
        # 第二轮 prompt 里应带上第一轮的工具结果
        second_prompt = llm.calls[1][-1]["content"]
        self.assertIn("已执行的工具调用", second_prompt)
        self.assertIn("kb_search", second_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
