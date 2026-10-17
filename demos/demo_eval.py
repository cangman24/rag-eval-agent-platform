# -*- coding: utf-8 -*-
"""demo_eval.py —— 演示 Day 3 的完整流程：评测集体检 -> 四档消融 -> 生成报告。

跑法：python demo_eval.py
产物：reports/demo_ablation.md（含消融对比表、分类型表、Badcase 清单）

★ 注意：这里用的是**假向量**（字符哈希），只验证框架可用。
  真实效果必须把 embed_fn 换成 BGE-M3、rerank_fn 换成 bge-reranker-v2-m3。
  报告里的数字在换成真模型前，一律不能写进简历。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.bm25 import BM25Index
from core.evaluation import (
    EvalItem, render_ablation_table, run_retrieval_eval, save_eval_set,
    validate_eval_set, write_report,
)
from core.pipeline import HybridRetriever, RetrievalConfig, ablation_configs
from core.retrieval import recursive_chunk
from core.text import tokenize
from core.vectorstore import InMemoryVectorStore, l2_normalize

ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ 语料
CORPUS = """第一章 总则
第一条 本规范适用于水稻穗与小麦头的无人机影像检测任务。
第二条 检测应采用可见光影像，必要时辅以多光谱数据。
第二章 技术要求
第一节 精度要求
检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。
在密集遮挡场景下，允许 mAP 下降不超过 3 个百分点。
第二节 数据集要求
训练集、验证集、测试集建议按 7:1:2 划分，标注使用 Labelme 并转换为 YOLO 格式。
单张影像分辨率建议不低于 4000x3000，地面采样距离不高于 2 厘米。
第三节 模型选型
推荐使用 YOLOv8 作为基线模型，并与 Faster R-CNN 做对比实验。
第三章 部署要求
模型应支持导出 ONNX 并量化，以满足端侧推理的时延要求。
推理时延在边缘设备上不宜超过 200 毫秒，显存占用不宜超过 2GB。
第四章 质量验收
验收时需提交精度报告、混淆矩阵与 Failure Case 分析。
"""


def fake_embed(text: str, dim: int = 64):
    v = [0.0] * dim
    for tok in tokenize(text):
        h = sum(ord(c) * (i + 1) for i, c in enumerate(tok))
        v[h % dim] += 1.0
    return l2_normalize(v)


def fake_rerank(query: str, pairs):
    """字符重合度重排（占位）。真实项目必须换成 BGE-Reranker。"""
    q = set(tokenize(query))
    scored = []
    for did, text in pairs:
        d = set(tokenize(text))
        overlap = len(q & d) / (len(q) or 1)
        scored.append((did, float(overlap)))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


# ------------------------------------------------------------------ 建索引
chunks = recursive_chunk(CORPUS, chunk_size=64, overlap=8)
DOCS = {f"c{i}": f"{c['heading_path']} {c['text']}" for i, c in enumerate(chunks)}

bm25 = BM25Index()
for did, text in DOCS.items():
    bm25.add(did, text)
bm25.finalize()

store = InMemoryVectorStore()
store.add(ids=list(DOCS), embeddings=[fake_embed(t) for t in DOCS.values()],
          documents=list(DOCS.values()),
          metadatas=[{"text": t} for t in DOCS.values()])

retriever = HybridRetriever(
    bm25=bm25, vector_store=store, embed_fn=fake_embed,
    rerank_fn=fake_rerank, doc_lookup=lambda d: DOCS.get(d),
)

# ------------------------------------------------------------------ 评测集
# ★ 真实项目里 gold_ids 必须人工标注。这里为了演示框架，用关键词定位。
def find_doc(*keywords) -> list:
    hits = []
    for did, text in DOCS.items():
        if all(k in text for k in keywords):
            hits.append(did)
    return hits


RAW_ITEMS = [
    ("q01", "mAP 精度要求是多少", ("0.90",), "fact"),
    ("q02", "F1 要求是多少", ("0.88",), "fact"),
    ("q03", "数据集怎么划分", ("7:1:2",), "fact"),
    ("q04", "标注用什么工具", ("Labelme",), "fact"),
    ("q05", "影像分辨率要求", ("4000x3000",), "term"),
    ("q06", "地面采样距离要求", ("2 厘米",), "term"),
    ("q07", "推荐什么基线模型", ("YOLOv8",), "fact"),
    ("q08", "要和什么模型做对比", ("Faster R-CNN",), "fact"),
    ("q09", "导出什么格式并量化", ("ONNX",), "term"),
    ("q10", "端侧时延要求", ("200 毫秒",), "term"),
    ("q11", "显存占用要求", ("2GB",), "term"),
    ("q12", "验收要提交什么材料", ("混淆矩阵",), "fact"),
    ("q13", "密集遮挡场景的容差", ("3 个百分点",), "multi_hop"),
    ("q14", "适用对象是什么", ("水稻穗",), "fact"),
]

items = []
for qid, query, kws, qtype in RAW_ITEMS:
    gold = find_doc(*kws)
    items.append(EvalItem(qid=qid, query=query, gold_ids=gold,
                          gold_answer="", qtype=qtype, is_answerable=bool(gold)))
# 越界负样本
items.append(EvalItem(qid="q90", query="公司的年假政策是怎样的", gold_ids=[],
                      gold_answer="", qtype="out_of_scope", is_answerable=False))
items.append(EvalItem(qid="q91", query="请给出该型号轴承的许用转速", gold_ids=[],
                      gold_answer="", qtype="out_of_scope", is_answerable=False))

# ------------------------------------------------------------------ 体检
health = validate_eval_set(items)
print("=" * 78)
print("评测集体检")
print("=" * 78)
print(f"总数={health['total']}  可答={health['answerable']}  "
      f"不可答={health['unanswerable']}")
print(f"类型分布={health['by_type']}")
print(f"体检结果={'通过' if health['ok'] else '有问题'}")
for p in health["problems"]:
    print("  !", p)

save_eval_set(items, os.path.join(ROOT, "data", "eval", "demo_eval.jsonl"))
print(f"\n评测集已写入 data/eval/demo_eval.jsonl（{len(items)} 条）")

# ------------------------------------------------------------------ 消融
print()
print("=" * 78)
print("消融实验（4 档配置）")
print("=" * 78)
reports = []
for cfg in ablation_configs(top_k=5, recall_k=20):
    rep = run_retrieval_eval(
        items,
        lambda q, c=cfg: retriever.retrieve(q, c),
        config_name=f"{cfg.name} ({cfg.label()})",
    )
    reports.append(rep)
    print(f"{cfg.name:<26} n={rep.n:<3} hit@5={rep.overall.get('hit@5', 0):.4f}  "
          f"mrr@5={rep.overall.get('mrr@5', 0):.4f}  "
          f"ndcg@5={rep.overall.get('ndcg@5', 0):.4f}  "
          f"P95={rep.latency.get('p95_ms', 0):.1f}ms")

# ------------------------------------------------------------------ 报告
report_path = os.path.join(ROOT, "reports", "demo_ablation.md")
write_report(
    reports[-1], report_path,
    extra_sections={"消融对比（四档配置）": render_ablation_table(reports)},
)
print(f"\n详细报告已写入 reports/demo_ablation.md")

print()
print("=" * 78)
print("★ 换成真模型后要做的事")
print("=" * 78)
print("""
1. embed_fn  -> BGE-M3（GPU 批处理，输出 L2 归一化向量）
2. rerank_fn -> bge-reranker-v2-m3（对融合后的 top-20 交叉编码精排）
3. gold_ids  -> 人工复核（现在是按关键词自动定位，存在循环论证风险）
4. 评测集     -> 扩到 120 条，覆盖 事实/多跳/术语/越界 四类
5. 再跑本脚本 -> reports/ 里的数字才能写进简历

另外，真实项目还应加一档基线：纯 LLM 直答（不检索），
用来证明"引入 RAG 确实带来了提升"，而不只是证明"混合检索比纯向量好"。
""")
