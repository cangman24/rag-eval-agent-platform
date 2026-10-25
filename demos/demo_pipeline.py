# -*- coding: utf-8 -*-
"""demo_pipeline.py —— 无需网络 / GPU / API Key 的完整检索链路演示。

跑法：python demo_pipeline.py
输出同时写到 demo_pipeline_output.txt

演示内容：
  1. 中文规范文档切分（保留 heading_path）
  2. BM25 关键词检索（含 min_score 阈值过滤）
  3. 向量检索（用本地 hash 假向量代替 BGE，验证链路而非模型质量）
  4. RRF 融合 -> Reranker 位置（此处用简单重排占位）
  5. 低置信拒答
  6. 用 core.metrics 算出一组真实的检索指标

★ 重要：本 demo 的"向量"是字符哈希假向量，只用于验证**代码链路正确**。
  真实效果必须换成 BGE-M3 —— 那时才把 demo 里的指标写进简历。
"""
import hashlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bm25 import BM25Index
from core.llm import should_refuse
from core.metrics import all_retrieval_metrics, aggregate
from core.retrieval import recursive_chunk, rrf_fuse
from core.text import tokenize
from core.vectorstore import InMemoryVectorStore, l2_normalize

OUT = []


def log(line=""):
    OUT.append(line)
    print(line)


# ------------------------------------------------------------------ 语料
DOC = """第一章 总则
第一条 本规范适用于水稻穗与小麦头的无人机影像检测任务。
第二条 检测应采用可见光影像，必要时辅以多光谱数据。
第二章 技术要求
第一节 精度要求
检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。
在密集遮挡场景下，允许 mAP 下降不超过 3 个百分点。
第二节 数据集要求
训练集、验证集、测试集建议按 7:1:2 划分，标注使用 Labelme 并转换为 YOLO 格式。
单张影像分辨率建议不低于 4000x3000，地面采样距离不高于 2 厘米。
第三章 部署要求
模型应支持导出 ONNX 并量化，以满足端侧推理的时延要求。
推理时延在边缘设备上不宜超过 200 毫秒。
"""


def fake_embed(text: str, dim: int = 64) -> list:
    """字符哈希假向量：仅用于打通链路，不代表语义质量。"""
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    return l2_normalize(vec)


def banner(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


# ------------------------------------------------------------------ 1 切分
banner("1. 文档切分（保留 heading_path，第X条作为正文保留）")
# ★ chunk_size 要避开"数值被切在边界"的情形：48 会把 "F1 不低于 0.88。"
#   从 "88" 处切开（见下方历史输出）。真实项目里这是必须处理的 —— 选
#   chunk_size 时先扫一遍语料，确认关键数值不被切割；这也是网格搜索
#   除了看指标之外要人工检查的一项。
chunks = recursive_chunk(DOC, chunk_size=64, overlap=8)
for i, c in enumerate(chunks):
    log(f"[{i}] tokens={c['token_len']:>3}  path={c['heading_path']!r}")
    log(f"      {c['text']}")

# ------------------------------------------------------------------ 2 BM25
banner("2. BM25 关键词检索")
bm25 = BM25Index()
for i, c in enumerate(chunks):
    # 把 heading_path 一并建索引：章节名本身也是强信号
    bm25.add(f"c{i}", f"{c['heading_path']} {c['text']}")
bm25.finalize()
log(f"索引文档数={len(bm25)}  avgdl={bm25.avgdl:.1f}")

for q in ["mAP 精度要求", "数据集怎么划分", "ONNX 量化 时延", "完全不相关的查询内容"]:
    hits = bm25.search(q, top_k=3)
    log("")
    log(f"query={q!r}")
    if not hits:
        log("   (无命中)")
    for doc_id, s in hits:
        idx = int(doc_id[1:])
        log(f"   {doc_id}  score={s:6.3f}  {chunks[idx]['text'][:34]}")

log("")
log("★ 注意最后一条：无关查询仍靠中文单字重合拿到低分 -> 这就是 min_score 阈值存在的理由")
noise = bm25.search("完全不相关的查询内容", top_k=1)
if noise:
    log(f"   噪声分={noise[0][1]:.3f}")
    cut = noise[0][1] + 1.0
    log(f"   设 min_score={cut:.1f} 后 -> {bm25.search('完全不相关的查询内容', min_score=cut)}")

# ------------------------------------------------------------------ 3 向量
banner("3. 向量检索（假向量，仅验证链路）")
store = InMemoryVectorStore()
store.add(
    ids=[f"c{i}" for i in range(len(chunks))],
    embeddings=[fake_embed(f"{c['heading_path']} {c['text']}") for c in chunks],
    documents=[c["text"] for c in chunks],
    metadatas=[{"heading_path": c["heading_path"]} for c in chunks],
)
log(f"向量库条数={store.count()}")
for q in ["mAP 精度要求", "ONNX 量化 时延"]:
    log("")
    log(f"query={q!r}")
    for doc_id, s in store.query(fake_embed(q), top_k=3):
        idx = int(doc_id[1:])
        log(f"   {doc_id}  cos={s:6.3f}  {chunks[idx]['text'][:34]}")

# ------------------------------------------------------------------ 4 RRF
banner("4. RRF 融合（奖励多路一致）")
q = "mAP 精度要求"
bm25_ids = [d for d, _ in bm25.search(q, top_k=5)]
vec_ids = [d for d, _ in store.query(fake_embed(q), top_k=5)]
log(f"BM25 top5 : {bm25_ids}")
log(f"向量 top5 : {vec_ids}")
fused = rrf_fuse({"bm25": bm25_ids, "vector": vec_ids}, k=60)
log("")
log("RRF 结果（Σ 1/(60+rank)）:")
for doc_id, s in fused:
    tags = []
    if doc_id in bm25_ids:
        tags.append(f"BM25#{bm25_ids.index(doc_id)+1}")
    if doc_id in vec_ids:
        tags.append(f"向量#{vec_ids.index(doc_id)+1}")
    idx = int(doc_id[1:])
    log(f"   {doc_id}  {s:.5f}  [{' '.join(tags)}]  {chunks[idx]['text'][:30]}")

# ------------------------------------------------------------------ 5 拒答
banner("5. 低置信拒答（★ 阈值必须按「分数尺度」来定）")
log("RRF 分数在 0.03 量级，BM25 分数在 1~10 量级 —— 两者不能共用一个阈值。")
log("实践中的做法：阈值绑定在**最终用于排序的那一路分数**上，并在评测集上扫出来。")
log("")

bm25_threshold = 2.5  # 由上一节实测：噪声 1.51，相关查询 2.6~9.7
for query, label in [("mAP 精度要求", "相关查询"), ("完全不相关的查询内容", "无关查询")]:
    hits = bm25.search(query, top_k=3)
    top = hits[0][1] if hits else 0.0
    decision = "拒答" if should_refuse([{"score": top}], bm25_threshold) else "正常作答"
    log(f"   BM25  score={top:6.3f}  {label:<6} -> {decision}")

log("")
log("RRF 尺度下（阈值需另取，例如 0.03）：")
fused_top = fused[0][1]
log(f"   RRF   score={fused_top:.5f}  -> "
    f"{'拒答' if should_refuse([{'score': fused_top}], 0.035) else '正常作答'}")
log("")
log("★ 面试点：低置信拒答的阈值不是拍脑袋定的，是在评测集上按"
    "「拒答正确率 vs 漏答率」权衡扫出来的。")

# ------------------------------------------------------------------ 6 指标
banner("6. 用 core.metrics 算一组真实指标（3 条 query 的评测演示）")
# gold 先跑检索再据实标注（真实项目里 gold 由人工标注，这里演示口径计算）
OUT_QUERIES = [("q1", "mAP 精度要求", 0), ("q2", "数据集怎么划分", 0), ("q3", "ONNX 量化 时延", 0)]
retrieved = {}
log(f"{'qid':<5}{'query':<20}{'top5':<26}{'gold':<6}hit@5  mrr@5  ndcg@5")
rows = []
for qid, query, _ in OUT_QUERIES:
    b = [d for d, _ in bm25.search(query, top_k=5)]
    v = [d for d, _ in store.query(fake_embed(query), top_k=5)]
    top5 = [d for d, _ in rrf_fuse({"bm25": b, "vector": v}, k=60)]
    retrieved[qid] = top5
    gold = [top5[0]]  # 演示：以最强召回的片段作为 gold（真实项目必须人工标注）
    m = all_retrieval_metrics(top5, gold, k=5)
    rows.append(m)
    log(f"{qid:<5}{query:<20}{','.join(top5):<26}{','.join(gold):<6}"
        f"{m['hit@5']:<7.2f}{m['mrr@5']:<7.2f}{m['ndcg@5']:<7.2f}")

agg = aggregate(rows)
log("")
log(f"聚合结果: {agg}")
log("")
log("★ 说明：这里 gold 直接取了 top1，所以 hit@5/mrr@5 必然是 1.0 —— 这样写只是")
log("  演示「指标怎么算」，不能当效果指标。真实评测必须：")
log("    ① gold 由人工标注（不看检索结果，避免循环论证）；")
log("    ② 用 120 条覆盖 事实/多跳/术语/越界 四类的评测集；")
log("    ③ 把假向量换成 BGE-M3 并接上 Reranker。")
log("  那时算出来的数才是能写进简历的。")

# ------------------------------------------------------------------ 写文件
io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_pipeline_output.txt"),
        "w", encoding="utf-8").write("\n".join(OUT))
