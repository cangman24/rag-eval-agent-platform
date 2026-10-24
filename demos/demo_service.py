# -*- coding: utf-8 -*-
"""demo_service.py —— 服务层演示：CLI 输出 + SSE 协议 + Agent 轨迹。

跑法：python demo_service.py
输出：demo_service_output.txt

演示内容：
  1. 服务层问答（含引用、拒答、无 LLM 降级、错误降级）
  2. SSE 帧协议（原文 + 解析回来）
  3. CLI 三条命令的实际输出（与服务层同一条代码路径）
  4. Agent 服务：执行 / 轨迹 / 工具清单 / 断点续跑
  5. 统一 dispatch 入口（同一份逻辑可供 HTTP / CLI / 消息队列复用）
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import AgentEngine, InMemoryCheckpointer
from core.mcp import ToolRegistry
from core.service import (
    AgentService, RagService, ServiceConfig, dispatch, new_request_id,
)
from core.sse import parse_sse
from rag1_knowledge_qa.cli import cmd_ask, cmd_health, cmd_retrieve

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


def capture(fn, *a):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*a)
    return buf.getvalue()


# ================================================================== 假依赖
DOCS = {
    "c1": "父块一：检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。",
    "c2": "父块二：训练集、验证集、测试集按 7:1:2 划分。",
    "c3": "父块三：模型应导出 ONNX 并量化，端侧时延不超过 200 毫秒。",
}
METAS = {
    "c1": {"heading_path": "第二章 技术要求 > 第一节 精度要求"},
    "c2": {"heading_path": "第二章 技术要求 > 第二节 数据集要求"},
    "c3": {"heading_path": "第三章 部署要求 > 第一节 端侧推理"},
}
KEYWORDS = {"精度": ["c1"], "mAP": ["c1"], "数据集": ["c2"], "划分": ["c2"],
            "ONNX": ["c3"], "时延": ["c3"], "部署": ["c3"]}


class FakeRetriever:
    def __init__(self, scores=None, raise_on=None):
        self.scores = scores or {}
        self.raise_on = raise_on

    def retrieve(self, query, cfg=None):
        if self.raise_on and self.raise_on in query:
            raise RuntimeError("Milvus 连接超时")
        ids = []
        for k, v in KEYWORDS.items():
            if k in query:
                ids.extend(i for i in v if i not in ids)
        return {"doc_ids": ids, "scores": [self.scores.get(i, 0.9) for i in ids]}


class FakeLLM:
    def __init__(self, text="mAP 不低于 0.90，F1 不低于 0.88 [1]", fail=False):
        self.text = text
        self.fail = fail

    def chat(self, messages, **kw):
        if self.fail:
            raise RuntimeError("DeepSeek 503 Service Unavailable")
        class R:
            pass
        r = R()
        r.text = self.text
        return r


def make_service(llm=None, scores=None, raise_on=None, **cfg):
    return RagService(
        retriever=FakeRetriever(scores, raise_on),
        doc_lookup=lambda c: DOCS.get(c, ""),
        llm=llm,
        meta_lookup=lambda c: METAS.get(c, {}),
        config=ServiceConfig(**cfg),
    )


# ================================================================== 1 问答
banner("1. 服务层问答：四种路径")
svc = make_service(llm=FakeLLM())

log("① 正常作答（带引用）")
r = svc.answer("检测精度要求是多少")
log(f"   回答: {r.answer}")
log(f"   引用: {[(c.index, c.chunk_id, c.heading_path) for c in r.citations]}")
log(f"   refused={r.refused}  耗时={r.latency_ms:.1f}ms")

log()
log("② 低置信拒答（检索分数 0.05 < 阈值 0.3）")
r2 = make_service(llm=FakeLLM("不该被调用"), scores={"c1": 0.05}).answer("精度要求")
log(f"   回答: {r2.answer}")
log(f"   refused={r2.refused}  引用数={len(r2.citations)}  ← 拒答时清空引用")

log()
log("③ 无 LLM 降级（没配 API Key 也能本地冒烟）")
r3 = make_service(llm=None).answer("精度要求")
log(f"   回答前 60 字: {r3.answer[:60]}...")

log()
log("④ 检索失败降级（不返回 500，给可读提示）")
r4 = make_service(llm=FakeLLM(), raise_on="触发").answer("触发异常")
log(f"   回答: {r4.answer}")
log(f"   error: {r4.error}")

log()
log("⑤ LLM 失败降级（返回检索片段而不是崩溃）")
r5 = make_service(llm=FakeLLM(fail=True)).answer("精度")
log(f"   error: {r5.error}")
log(f"   回答前 60 字: {r5.answer[:60]}...")

# ================================================================== 2 SSE
banner("2. SSE 流式协议")
frames = "".join(svc.answer_stream("部署时延要求"))
parsed = [e for e in parse_sse(frames) if "event" in e]
log(f"生成 {len(parsed)} 条事件帧")
log()
log("帧原文（前 2 条）:")
for f in frames.split("\n\n")[:2]:
    if f.strip():
        log("  " + f.replace("\n", "\n  "))
        log("  " + "-" * 40)
log()
log(f"事件类型序列: {[e['event'] for e in parsed]}")
log(f"每条带自增 id: {[e['id'] for e in parsed]}")
log()
log("★ 引用只在确定作答后才发 —— 拒答场景下前端不会渲染出自相矛盾的来源。")
refuse_frames = "".join(make_service(llm=FakeLLM(), scores={"c1": 0.01})
                        .answer_stream("精度"))
refuse_types = [e["event"] for e in parse_sse(refuse_frames) if "event" in e]
log(f"   拒答时的事件序列: {refuse_types}   ← 无 citation")

# ================================================================== 3 CLI
banner("3. CLI 输出（与服务层同一条代码路径）")
log("$ python -m rag1_knowledge_qa.cli ask \"检测精度要求是多少\"")
log("-" * 78)
log(capture(cmd_ask, svc, type("A", (), {"query": "检测精度要求是多少",
                                         "stream": False, "json": False})()).rstrip())
log()
log("$ python -m rag1_knowledge_qa.cli retrieve \"ONNX 时延\" --top-k 2")
log("-" * 78)
log(capture(cmd_retrieve, svc, type("A", (), {"query": "ONNX 时延",
                                              "top_k": 2, "json": False})()).rstrip())
log()
log("$ python -m rag1_knowledge_qa.cli health")
log("-" * 78)
log(capture(cmd_health, svc, type("A", (), {})()).rstrip())

# ================================================================== 4 Agent
banner("4. Agent 服务：执行 / 轨迹 / 工具清单")
registry = ToolRegistry()


@registry.tool(description="检索知识库，返回相关制度片段")
def kb_search(query: str, top_k: int = 3):
    """检索知识库

    query: 检索关键词
    top_k: 返回条数
    """
    return {"count": 1, "hits": [{"id": "c1", "text": DOCS["c1"]}]}


@registry.tool(description="查询工单状态")
def ticket_query(ticket_id: str):
    """查询工单

    ticket_id: 工单号
    """
    return {"ticket_id": ticket_id, "status": "closed"}


@registry.tool(description="发送通知邮件（外部副作用，需人工确认）", sensitive=True)
def send_mail(to: str):
    """发送邮件

    to: 收件人
    """
    return {"sent": True}


decisions = [
    {"tool": "kb_search", "args": {"query": "精度要求"}},
    {"tool": "ticket_query", "args": {"ticket_id": "T-100"}},
    {"final": "精度要求 mAP 不低于 0.90 [1]；工单 T-100 已关闭 [2]。",
     "citations": ["c1", "T-100"]},
]
cursor = {"i": 0}


def act_fn(state, allowed):
    d = decisions[min(cursor["i"], len(decisions) - 1)]
    cursor["i"] += 1
    return d


checkpointer = InMemoryCheckpointer()
engine = AgentEngine(
    registry=registry,
    intent_fn=lambda q: "review" if "工单" in q else "spec",
    plan_fn=lambda q, t: ["检索制度", "查询工单", "汇总结论"],
    act_fn=act_fn, route_table={"spec": ["kb_search"],
                                "review": ["kb_search", "ticket_query"],
                                "general": []},
    checkpointer=checkpointer, max_rounds=6, max_tool_calls=6,
)
agent_svc = AgentService(engine)

out = agent_svc.run("工单 T-100 的关闭是否合规", thread_id="T-DEMO")
log(f"意图       = {out['task_type']}")
log(f"白名单     = {out['allowed_tools']}")
log(f"执行计划   = {out['plan']}")
log(f"轮次/工具  = {out['rounds']} / {out['tool_calls']}")
log(f"verified   = {out['verified']}   finished = {out['finished']}")
log(f"回答       = {out['answer']}")
log()
log("工具清单（MCP 格式）与路由表:")
tools = agent_svc.tools()
for t in tools["tools"]:
    flag = "  [敏感]" if t["name"] == "send_mail" else ""
    log(f"  - {t['name']:<14} {t['description']}{flag}")
log(f"  routes = {tools['routes']}")
log()
trace = agent_svc.trace("T-DEMO")
log(f"轨迹事件 {len(trace['events'])} 条，scratchpad {len(trace['scratchpad'])} 项:")
for s in trace["scratchpad"]:
    if s.get("tool"):
        log(f"  [{s['type']}] {s['tool']} ok={s.get('ok')}")

# ================================================================== 5 dispatch
banner("5. 统一 dispatch 入口（HTTP / CLI / MQ 共用同一份逻辑）")
log("dispatch(rag_service, 'answer', {'query': '精度要求'})")
d1 = dispatch(svc, "answer", {"query": "精度要求"})
log(f"  -> answer={d1['answer'][:40]!r}  citations={len(d1['citations'])}")
log()
log("dispatch(agent_service, 'tools', {})")
d2 = dispatch(agent_svc, "tools", {})
log(f"  -> tools={[t['name'] for t in d2['tools']]}")
log()
log("dispatch(agent_service, 'trace', {'thread_id': 'T-DEMO'})")
d3 = dispatch(agent_svc, "trace", {"thread_id": "T-DEMO"})
log(f"  -> found={d3['found']} rounds={d3['rounds']}")
log()
log("★ 这一层让 HTTP / CLI / 消息队列 / 单元测试复用同一份业务逻辑，")
log("  不会出现'接口和命令行行为不一致'的经典问题。")

# ================================================================== 6 断点续跑
banner("6. 服务层断点续跑（Checkpoint）")
cp2 = InMemoryCheckpointer()
cursor2 = {"i": 0}


def act_loop(state, allowed):
    return {"tool": "kb_search", "args": {"query": "loop"}}


eng_a = AgentEngine(registry=registry, intent_fn=lambda q: "spec",
                    plan_fn=lambda q, t: ["检索"], act_fn=act_loop,
                    route_table={"spec": ["kb_search"]},
                    checkpointer=cp2, max_rounds=1, max_tool_calls=1)
svc_a = AgentService(eng_a)
a = svc_a.run("查制度", thread_id="T-RESUME")
log(f"第一段: rounds={a['rounds']} finished={a['finished']} "
    f"stop_reason={a['stop_reason']}")

cursor3 = {"i": 0}


def act_final(state, allowed):
    cursor3["i"] += 1
    if cursor3["i"] == 1:
        return {"tool": "kb_search", "args": {"query": "x"}}
    return {"final": "续跑得到的结论", "citations": []}


eng_b = AgentEngine(registry=registry, intent_fn=lambda q: "spec",
                    plan_fn=lambda q, t: ["检索"], act_fn=act_final,
                    route_table={"spec": ["kb_search"]},
                    checkpointer=cp2, max_rounds=5)
svc_b = AgentService(eng_b)
b = svc_b.resume("T-RESUME")
log(f"第二段: rounds={b['rounds']}（从 {a['rounds']} 继续累加，不是从 0 重来）")
log(f"        finished={b['finished']}  answer={b['answer']!r}")

banner("★ 下一步：接真模型")
log("""
1. core/embedder.py  -> BgeEmbedder.encode_batch / encode_one（L2 归一化）
2. core/reranker.py  -> BgeReranker.rerank（交叉编码）
3. .env 里填 DEEPSEEK_API_KEY
4. python -m rag1_knowledge_qa.cli ingest --docs data/raw     # 建库
5. python -m rag1_knowledge_qa.cli ask "..."                  # 问答
6. pip install fastapi uvicorn && uvicorn rag1_knowledge_qa.api_fastapi:app_factory --factory

本文件里的服务层、CLI、Agent、SSE、dispatch 全部**不需要改动**。
""")

io.open(os.path.join(ROOT, "demo_service_output.txt"), "w",
        encoding="utf-8").write("\n".join(OUT))
print("\n已写入 demo_service_output.txt")
