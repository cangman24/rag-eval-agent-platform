# -*- coding: utf-8 -*-
"""demo_evalrun.py —— 演示"一条命令产出可复现评测报告"。

跑法：python demo_evalrun.py
产物：reports/demo-run/ 目录，含
        SUMMARY.md            一页看完的结果（面试时打开这个）
        eval_manifest.json    运行指纹（回答"这数字是哪套参数跑的"）
        retrieval_*.md        每档逐条明细 + Badcase
        eval_set.jsonl        本次用的评测集副本

演示内容：
  1. 建库（真实文件 -> 父子块 -> 双索引）
  2. 四档检索消融 + 多 k 值
  3. 生成层评测 + LLM-as-Judge（用假模型，验证判分链路）
  4. 运行指纹与报告目录结构
  5. 两次 run 对比（改了参数后看有没有变好）
"""
import io
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.eval_runner import (
    EvalRunConfig, compare_runs, run_eval,
)
from core.evaluation import EvalItem, save_eval_set
from core.ingest import IngestConfig, ingest_dir
from core.judge import LLMJudge
from core.pipeline import HybridRetriever, ablation_configs, RetrievalConfig
from core.text import tokenize
from core.vectorstore import l2_normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(ROOT, "data", "demo_evalrun_docs")
OUT = []


def log(line=""):
    OUT.append(line)
    print(line)


def banner(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


# ================================================================== 语料
CORPUS = {
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
    "03_验收标准.md": """# 验收标准

## 提交材料
验收时需提交精度报告、混淆矩阵与 Failure Case 分析。
所有实验需给出可复现的随机种子与超参数表。

## 合格线
整体 mAP@0.5 不低于 0.88 视为合格，不低于 0.92 视为优秀。
""",
}


def prepare_docs():
    shutil.rmtree(DOCS_DIR, ignore_errors=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    for name, content in CORPUS.items():
        with open(os.path.join(DOCS_DIR, name), "w", encoding="utf-8") as f:
            f.write(content)


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * 64
        for tok in tokenize(t):
            v[sum(map(ord, tok)) % 64] += 1.0
        out.append(l2_normalize(v))
    return out


def fake_rerank(query, pairs):
    q = set(tokenize(query))
    scored = [(did, len(q & set(tokenize(text))) / (len(q) or 1))
              for did, text in pairs]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return scored


class ScriptedJudgeLLM:
    """假的评委模型：根据回答里是否含关键数值给分。

    这样能验证「判分链路 + 报告聚合」是对的，而不需要真调 API。
    """

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        prompt = messages[-1]["content"]
        # 回答段落在【模型回答】之后
        answer = prompt.split("【模型回答】")[-1]
        good = any(k in answer for k in ("0.90", "7:1:2", "ONNX", "200", "0.88"))
        scores = {"relevance": 5, "accuracy": 5 if good else 2,
                  "completeness": 4 if good else 2,
                  "groundedness": 5 if good else 2,
                  "reason": "含关键数值" if good else "缺少关键数值"}

        class R:
            pass
        r = R()
        r.text = json.dumps(scores, ensure_ascii=False)
        return r


# ================================================================== 建库
banner("0. 准备语料并建库")
prepare_docs()
res = ingest_dir(DOCS_DIR, fake_embed, config=IngestConfig(
    chunk_size=64, overlap=8, parent_size=220, min_chunk_tokens=5,
    embedding_model="fake-hash-64d"), verbose=True)
log(f"父块 {res.stats['parents']} / 子块 {res.stats['chunks']} / "
    f"指纹 {res.stats['config_fingerprint']}")

retriever = HybridRetriever(
    bm25=res.bm25, vector_store=res.vector_store,
    embed_fn=lambda q: fake_embed([q])[0], rerank_fn=fake_rerank,
    doc_lookup=lambda d: res.child_text(d),
)


def retrieve_with(cfg, query):
    return retriever.retrieve(query, cfg)


# ================================================================== 评测集
def gold_for(kw, k=1):
    hits = res.bm25.search(kw, top_k=k, min_score=0.3)
    if not hits:
        hits = res.bm25.search(kw, top_k=k, min_score=0.0)
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
]
items = [EvalItem(qid, q, gold_for(kw), "", t, bool(gold_for(kw)))
         for qid, q, kw, t in RAW]
items.append(EvalItem("q90", "公司的年假有多少天", [], "", "out_of_scope", False))

EVAL_SET = os.path.join(ROOT, "data", "eval", "demo_evalrun_eval.jsonl")
save_eval_set(items, EVAL_SET)
log()
log(f"评测集已写入 {os.path.relpath(EVAL_SET, ROOT)}（{len(items)} 条）")

# ================================================================== 跑评测
banner("1. 跑完整评测（四档消融 + 多 k + 生成层 LLM-as-Judge）")
judge_llm = ScriptedJudgeLLM()

# 检索片段缓存：answer_fn 顺便把片段存下来，评委直接复用，避免重复检索
_ctx = {}


def answer_fn(query: str) -> str:
    hits = retriever.retrieve(query, RetrievalConfig(top_k=3))
    texts = [res.child_text(d) for d in hits.doc_ids]
    _ctx[query] = "\n".join(texts)
    if not texts:
        return "知识库中未找到相关内容。"
    return "根据检索结果：" + texts[0][:200] + " [1]"


# ★ 必须只创建**一个** LLMJudge 实例。
#   初版在 judge_fn 里每次都 new 一个 LLMJudge，于是 manifest 里的 judge.calls
#   永远是 0 —— 计数器散落在几十个临时对象上，统计等于没做。
judge = LLMJudge(judge_llm, context_provider=lambda q: _ctx.get(q, ""))


def judge_fn(item, answer):
    return judge(item, answer)


OUT_DIR = os.path.join(ROOT, "reports")
result = run_eval(
    items, ablation_configs(top_k=5, recall_k=20),
    lambda cfg: (lambda q: retrieve_with(cfg, q)),
    EvalRunConfig(
        eval_set_path=EVAL_SET, out_dir=OUT_DIR, run_id="demo-run",
        top_k=5, recall_k=20, k_list=(5, 3),
        embedding_model="fake-hash-64d",
        ingest_meta=res.stats,
        judge_enabled=True, judge_model="scripted-judge",
        notes="demo：验证评测链路与报告可复现性",
    ),
    answer_fn=answer_fn, judge_fn=judge_fn,
    judge_report=judge.report,
    progress=log,
)

# ================================================================== 报告结构
banner("2. 报告目录与运行指纹")
log(f"目录：{os.path.relpath(result.out_dir, ROOT)}/")
for name in sorted(os.listdir(result.out_dir)):
    size = os.path.getsize(os.path.join(result.out_dir, name))
    log(f"  {name:<38} {size:>7} B")

log()
log("eval_manifest.json 的关键字段（★ 面试时被问「这数字怎么来的」就看它）：")
m = result.manifest
show_keys = [
    ("run_id", m["run_id"]),
    ("timestamp", m["timestamp"]),
    ("eval_set.n", m["eval_set"]["n"]),
    ("eval_set.sha256", m["eval_set"].get("sha256")),
    ("eval_set.by_type", m["eval_set"]["by_type"]),
    ("ingest.embedding_model", m["ingest"].get("embedding_model")),
    ("ingest.chunk_size", m["ingest"].get("chunk_size")),
    ("ingest.config_fingerprint", m["ingest"].get("config_fingerprint")),
    ("retrieval.top_k", m["retrieval"]["top_k"]),
    ("retrieval.configs", [c["name"] for c in m["retrieval"]["configs"]]),
]
for k, v in show_keys:
    log(f"  {k:<28} = {v}")
log(f"  {'judge':<28} = {m['judge']}")

# ================================================================== 对比
banner("3. 第二次 run（模拟改了 chunk_size 后再跑一次）并对比")
res2 = ingest_dir(DOCS_DIR, fake_embed, config=IngestConfig(
    chunk_size=40, overlap=4, parent_size=150, min_chunk_tokens=3,
    embedding_model="fake-hash-64d"))
retriever2 = HybridRetriever(
    bm25=res2.bm25, vector_store=res2.vector_store,
    embed_fn=lambda q: fake_embed([q])[0], rerank_fn=fake_rerank,
    doc_lookup=lambda d: res2.child_text(d),
)
result2 = run_eval(
    items, ablation_configs(top_k=5, recall_k=20),
    lambda cfg: (lambda q: retriever2.retrieve(q, cfg)),
    EvalRunConfig(
        eval_set_path=EVAL_SET, out_dir=OUT_DIR, run_id="demo-run-v2",
        top_k=5, recall_k=20, embedding_model="fake-hash-64d",
        ingest_meta=res2.stats,
        notes="demo：改 chunk_size 40/overlap 4 后再跑",
    ),
    progress=lambda m: None,
)

diff = compare_runs(result.out_dir, result2.out_dir)
log(diff)

# ================================================================== 打印摘要
banner("4. SUMMARY.md 摘要（面试时打开这一个文件就够）")
summary = open(result.summary_path, encoding="utf-8").read()
lines = summary.splitlines()
# 只打印前 60 行，避免刷屏
for line in lines[:60]:
    log(line)
if len(lines) > 60:
    log(f"...（完整 {len(lines)} 行见 {os.path.relpath(result.summary_path, ROOT)}）")

banner("★ 换真模型后要做的事")
log("""
1. fake_embed      -> BgeEmbedder.encode_batch（GPU，L2 归一化）
2. fake_rerank     -> BgeReranker.rerank（交叉编码精排）
3. ScriptedJudgeLLM-> 真 DeepSeek（LLMJudge(LLMClient(...))）
4. gold_for        -> 人工复核标注（现在是按关键词自动定位，有循环论证风险）
5. 评测集扩到 120 条，覆盖 事实/多跳/术语/越界
6. 然后一条命令出报告：

   python -m rag1_knowledge_qa.eval --eval-set data/eval/eval.jsonl \\
       --ablation --judge --k-list 5 3 10 --out reports

   看最近一次：  python -m rag1_knowledge_qa.eval --show
   对比两次：    python -m rag1_knowledge_qa.eval --compare reports/run-A reports/run-B

报告目录、指纹、对比功能**不需要改动**。
""")

io.open(os.path.join(ROOT, "demo_evalrun_output.txt"), "w",
        encoding="utf-8").write("\n".join(OUT))
print("\n已写入 demo_evalrun_output.txt")
