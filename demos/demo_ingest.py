# -*- coding: utf-8 -*-
"""demo_ingest.py —— 全流程端到端演示：文件 -> 索引 -> Agent -> 评测。

跑法：python demo_ingest.py
输出：demo_ingest_output.txt

这是把前面所有模块串起来的"总装配"演示，也是 Day 6 要打包进仓库的主脚本：

    ① loaders      解析 md / txt / json 多格式文档
    ② ingest       父子块切分 + 向量索引 + BM25 索引（含配置指纹）
    ③ pipeline     混合检索（BM25 + 向量 + RRF + Rerank）
    ④ agent        把检索封装成 MCP 工具，跑多步任务（含白名单/熔断）
    ⑤ sse          把 Agent 事件流转成 SSE 帧
    ⑥ evaluation   跑消融实验并生成 Markdown 报告

★ 嵌入用假实现（字符哈希），只验证链路；换成 BGE-M3 后代码一行不改。
"""
import io
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import AgentEngine, EventType
from core.evaluation import (
    EvalItem, render_ablation_table, run_retrieval_eval, save_eval_set,
    validate_eval_set, write_report,
)
from core.ingest import IngestConfig, ingest_dir, save_chunks, save_manifest
from core.mcp import ToolRegistry
from core.pipeline import HybridRetriever, RetrievalConfig, ablation_configs
from core.sse import parse_sse, stream_agent_events
from core.text import tokenize
from core.vectorstore import l2_normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(ROOT, "data", "demo_docs")
OUT = []


def log(line=""):
    OUT.append(line)
    print(line)


def banner(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


# ================================================================== 假嵌入
def fake_embed_batch(texts):
    """批量假嵌入（字符哈希 + L2 归一化）。真实项目换成 BGE-M3。"""
    out = []
    for t in texts:
        v = [0.0] * 64
        for tok in tokenize(t):
            h = sum(ord(c) * (i + 1) for i, c in enumerate(tok))
            v[h % 64] += 1.0
        out.append(l2_normalize(v))
    return out


def fake_embed_one(text):
    return fake_embed_batch([text])[0]


# ================================================================== 造语料
def prepare_docs():
    shutil.rmtree(DOCS_DIR, ignore_errors=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    files = {
        "01_检测技术要求.md": """# 技术要求

## 精度要求
检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。
在密集遮挡场景下，允许 mAP 下降不超过 3 个百分点。
小目标（面积小于 32x32 像素）召回率不低于 0.85。

## 数据集要求
训练集、验证集、测试集建议按 7:1:2 划分。
标注使用 Labelme 并转换为 YOLO 格式。
单张影像分辨率不低于 4000x3000，地面采样距离不高于 2 厘米。

## 模型选型
推荐 YOLOv8 作为基线，并与 Faster R-CNN 做对比实验。
""",
        "02_部署规范.md": """# 部署要求

## 端侧推理
模型应支持导出 ONNX 并量化，以满足端侧推理的时延要求。
推理时延在边缘设备上不宜超过 200 毫秒，显存占用不宜超过 2GB。

## 服务化
后端接口使用 FastAPI，支持 SSE 流式返回。
服务需支持 Docker 一键部署，并提供健康检查接口。
""",
        "03_验收标准.txt": """
验收标准

一、需提交材料
验收时需提交精度报告、混淆矩阵与 Failure Case 分析。
所有实验需给出可复现的随机种子与超参数表。

二、合格线
整体 mAP@0.5 不低于 0.88 视为合格，不低于 0.92 视为优秀。
""",
        "04_术语表.json": json.dumps({
            "mAP": "mean Average Precision，平均精度均值",
            "IoU": "Intersection over Union，交并比",
            "GSD": "Ground Sample Distance，地面采样距离",
            "ONNX": "开放神经网络交换格式，用于跨框架模型部署",
            "SSE": "Server-Sent Events，服务器推送事件",
        }, ensure_ascii=False),
    }
    for name, content in files.items():
        with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
            f.write(content)
    return list(files)


# ================================================================== 主流程
banner("0. 准备演示语料")
names = prepare_docs()
log(f"已写入 {len(names)} 个文档到 data/demo_docs/：")
for n in names:
    log(f"  - {n}")

# ------------------------------------------------------------------ 1 解析
banner("1. 多格式文档解析（loaders）")
from core.loaders import load_dir, summarize_load

load_result = load_dir(DOCS_DIR)
log(summarize_load(load_result))
log()
for d in load_result["docs"]:
    log(f"  {d.doc_id:<24} format={d.metadata['format']:<9} "
        f"{d.metadata['char_len']:>5} 字符")

# ------------------------------------------------------------------ 2 建库
banner("2. 建库流水线（ingest）：父子块 + 双子索引")
cfg = IngestConfig(chunk_size=64, overlap=8, parent_size=220,
                   embedding_model="fake-hash-64d（演示用，真机换 BGE-M3）",
                   min_chunk_tokens=5)
res = ingest_dir(DOCS_DIR, fake_embed_batch, config=cfg, verbose=False)
log(f"文档数     = {res.stats['documents']}")
log(f"父块数     = {res.stats['parents']}   （喂模型用，上下文完整）")
log(f"子块数     = {res.stats['chunks']}   （检索用，召回精准）")
log(f"平均块长   = {res.stats['avg_chunk_tokens']} tokens")
log(f"向量库条数 = {res.vector_store.count()}")
log(f"BM25 文档数= {len(res.bm25)}")
log(f"配置指纹   = {res.stats['config_fingerprint']}   ← 写进 manifest，保证可归因")
log()
log("★ 父子块验证：命中的子块 vs 回吐的父块")
sample_id = res.chunks[0].chunk_id
log(f"  子块 {sample_id}")
log(f"    {res.child_text(sample_id)[:70]}...")
log(f"  父块 {res.chunks[0].parent_id}")
log(f"    {res.doc_text(sample_id)[:70]}...")

# 落盘
save_manifest(res, os.path.join(ROOT, "reports", "ingest_manifest.json"))
save_chunks(res, os.path.join(ROOT, "data", "demo_chunks.json"))
log()
log("已写出 reports/ingest_manifest.json 与 data/demo_chunks.json")

# ------------------------------------------------------------------ 3 检索
banner("3. 混合检索（pipeline）：BM25 + 向量 + RRF + Rerank")


def fake_rerank(query, pairs):
    q = set(tokenize(query))
    scored = []
    for did, text in pairs:
        d = set(tokenize(text))
        scored.append((did, len(q & d) / (len(q) or 1)))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


retriever = HybridRetriever(
    bm25=res.bm25, vector_store=res.vector_store,
    embed_fn=fake_embed_one, rerank_fn=fake_rerank,
    doc_lookup=lambda d: res.child_text(d),
)
q = "端侧推理时延要求是多少"
r = retriever.retrieve(q, RetrievalConfig(use_vector=True, use_bm25=True,
                                          use_rerank=True, top_k=3))
log(f"query: {q}")
log(f"各路召回 trace: vector={len(r.trace.get('vector', []))} 条, "
    f"bm25={len(r.trace.get('bm25', []))} 条")
log(f"融合+精排后 top1: {res.child_text(r.doc_ids[0])[:60]}...")
log()
log("★ 回吐父块（喂给模型的上下文）:")
log(f"  {res.doc_text(r.doc_ids[0])[:100]}...")

# ------------------------------------------------------------------ 4 Agent
banner("4. Agent 多步任务（agent + mcp）：白名单 + 熔断 + 事件流")
registry = ToolRegistry()


@registry.tool(description="检索知识库，返回相关文档片段")
def kb_search(query: str, top_k: int = 3):
    """检索知识库

    query: 检索关键词
    top_k: 返回条数
    """
    hits = res.bm25.search(query, top_k=top_k, min_score=0.3)
    return {"count": len(hits),
            "hits": [{"id": d, "heading_path": res.metadata_for(d).get("heading_path", ""),
                      "text": res.doc_text(d)} for d, _ in hits]}


@registry.tool(description="查询验收材料清单")
def checklist_query(topic: str):
    """查询验收清单

    topic: 主题关键词
    """
    hits = res.bm25.search(f"验收 {topic}", top_k=2, min_score=0.3)
    return {"count": len(hits),
            "items": [res.child_text(d) for d, _ in hits]}


@registry.tool(description="提交验收结论（有副作用，需人工确认）", sensitive=True)
def submit_acceptance(conclusion: str):
    """提交验收结论

    conclusion: 结论文本
    """
    return {"submitted": True, "conclusion": conclusion}


ROUTE_TABLE = {
    "spec_inquiry": ["kb_search"],
    "acceptance": ["kb_search", "checklist_query"],
    "submit": ["kb_search", "submit_acceptance"],
    "general": [],
}

decisions = [
    {"tool": "kb_search", "args": {"query": "端侧 推理 时延 200 毫秒"}},
    {"tool": "checklist_query", "args": {"topic": "材料"}},
    {"final": "端侧推理时延要求不超过 200 毫秒、显存不超过 2GB [1]；"
              "验收需提交精度报告、混淆矩阵与 Failure Case 分析 [2]。",
     "citations": ["c1", "c2"]},
]
cursor = {"i": 0}


def act_fn(state, allowed):
    d = decisions[min(cursor["i"], len(decisions) - 1)]
    cursor["i"] += 1
    return d


engine = AgentEngine(
    registry=registry,
    intent_fn=lambda q: "acceptance" if ("验收" in q or "材料" in q) else "spec_inquiry",
    plan_fn=lambda q, t: ["检索规格要求", "查询验收清单", "汇总结论"],
    act_fn=act_fn,
    route_table=ROUTE_TABLE,
    max_rounds=6,
    max_tool_calls=6,
)
run = engine.run("部署时延要求和验收要交什么材料？", thread_id="demo-e2e")

log(f"意图      = {run.state.task_type}")
log(f"白名单    = {run.state.allowed_tools}")
log(f"执行计划  = {run.state.plan}")
log()
log("事件流:")
for e in run.events:
    d = e["data"]
    if e["type"] == EventType.TOOL_CALL:
        log(f"  [tool_call]   {d['tool']}({d['args']})")
    elif e["type"] == EventType.TOOL_RESULT:
        cnt = d["result"].get("count", "—") if isinstance(d["result"], dict) else "—"
        log(f"  [tool_result] {d['tool']} -> {cnt} 条")
    elif e["type"] == EventType.ANSWER:
        log(f"  [answer]      {d['text']}")
    elif e["type"] == EventType.DONE:
        log(f"  [done]        rounds={d['rounds']} tool_calls={d['tool_calls']}")
log()
log(f"轮次={run.state.rounds}  工具调用={run.state.tool_calls}  完成={run.state.finished}")

# ------------------------------------------------------------------ 5 SSE
banner("5. SSE 流式输出（sse）：前端轨迹面板的数据源")
frames = "".join(stream_agent_events(run.events))
parsed = [p for p in parse_sse(frames) if "event" in p]
log(f"生成 SSE 帧 {len(parsed)} 条，共 {len(frames)} 字节")
log()
log("前 3 帧原文（含 id / event / data，末尾空行表示事件结束）:")
for f in frames.split("\n\n")[:3]:
    if f.strip():
        log("  " + f.replace("\n", "\n  "))
        log("  " + "-" * 40)
log()
log("解析回来的事件类型序列:")
log(f"  {[p['event'] for p in parsed]}")
log()
log("★ 每条帧带自增 id，前端断线重连时通过 Last-Event-ID 续发 —— 长任务必需。")

# ------------------------------------------------------------------ 6 评测
banner("6. 评测与消融（evaluation）")


def gold_for(keyword, k=1):
    hits = res.bm25.search(keyword, top_k=k, min_score=0.3)
    return [d for d, _ in hits]


RAW = [
    ("q01", "mAP 精度要求是多少", "mAP 精度", "fact"),
    ("q02", "F1 要求是多少", "F1", "fact"),
    ("q03", "小目标召回率要求", "小目标 召回率", "fact"),
    ("q04", "数据集怎么划分", "7:1:2 划分", "fact"),
    ("q05", "标注用什么工具", "Labelme", "fact"),
    ("q06", "影像分辨率要求", "4000x3000", "term"),
    ("q07", "地面采样距离要求", "2 厘米", "term"),
    ("q08", "推荐什么基线模型", "YOLOv8", "fact"),
    ("q09", "导出什么格式并量化", "ONNX", "term"),
    ("q10", "端侧时延要求", "200 毫秒", "term"),
    ("q11", "显存占用要求", "2GB", "term"),
    ("q12", "后端用什么框架", "FastAPI", "fact"),
    ("q13", "部署方式要求", "Docker 一键部署", "fact"),
    ("q14", "验收要提交什么材料", "精度报告", "fact"),
    ("q15", "合格线是多少", "0.88", "term"),
    ("q16", "mAP 是什么意思", "mean Average Precision", "term"),
]
items = []
for qid, query, kw, qtype in RAW:
    g = gold_for(kw)
    items.append(EvalItem(qid, query, g, "", qtype, bool(g)))
items.append(EvalItem("q90", "公司的年假有多少天", [], "", "out_of_scope", False))
items.append(EvalItem("q91", "请给出某型号轴承的许用转速", [], "", "out_of_scope", False))

health = validate_eval_set(items)
log(f"评测集体检：{health['total']} 条（可答 {health['answerable']} / "
    f"不可答 {health['unanswerable']}），类型分布 {health['by_type']}")
log(f"体检结果：{'通过' if health['ok'] else '有问题 ' + str(health['problems'])}")
save_eval_set(items, os.path.join(ROOT, "data", "eval", "demo_ingest_eval.jsonl"))
log()

reports = []
for c in ablation_configs(top_k=5, recall_k=20):
    rep = run_retrieval_eval(items, lambda q, cc=c: retriever.retrieve(q, cc),
                             config_name=c.name)
    reports.append(rep)
    log(f"  {c.name:<26} n={rep.n:<3} hit@5={rep.overall.get('hit@5', 0):.4f}  "
        f"mrr@5={rep.overall.get('mrr@5', 0):.4f}  ndcg@5={rep.overall.get('ndcg@5', 0):.4f}")

write_report(reports[-1], os.path.join(ROOT, "reports", "demo_ingest_ablation.md"),
             extra_sections={"消融对比（四档配置）": render_ablation_table(reports)})
log()
log("详细报告 -> reports/demo_ingest_ablation.md")

# ------------------------------------------------------------------ 收尾
banner("★ 换成真模型后要改的东西")
log("""
1. fake_embed_batch  -> BgeEmbedder.encode_batch（GPU，L2 归一化）
2. fake_rerank       -> BgeReranker.rerank（交叉编码精排 top-20）
3. act_fn            -> 调 DeepSeek function calling：
                         tools = registry.openai_tools(allowed=allowed_tools)
                         decision = llm.chat_json(messages)
4. gold_for          -> 人工复核标注（现在是按关键词自动定位，存在循环论证）
5. 评测集            -> 扩到 120 条，覆盖 事实/多跳/术语/越界 四类
6. 再跑本脚本        -> reports/ 里的数字才能写进简历

其余代码（解析 / 父子块切分 / 双索引 / RRF 融合 / 白名单 / Checkpoint /
熔断 / 三层记忆 / 三级压缩 / SSE / 评测框架）一行都不用改。
""")

io.open(os.path.join(ROOT, "demo_ingest_output.txt"), "w",
        encoding="utf-8").write("\n".join(OUT))
print("\n已写入 demo_ingest_output.txt")
