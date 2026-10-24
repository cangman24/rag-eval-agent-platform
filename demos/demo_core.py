# -*- coding: utf-8 -*-
"""演示：中文规范文档的切分与 heading_path 保留效果。"""
import io, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.retrieval import recursive_chunk, rrf_fuse

out = []
out.append("########## 1. 中文规范文档切分 ##########")
text = """第一章 总则
第一条 本规范适用于水稻穗与小麦头的无人机影像检测。
第二条 检测应以可见光影像为主，必要时辅以多光谱影像。
第二章 技术要求
第一节 精度要求
检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。
在密集遮挡场景下，允许 mAP 下降不超过 3 个百分点。
第二节 数据集要求
训练集、验证集、测试集建议按 7:1:2 划分。
标注应使用 Labelme 并转换为 YOLO 格式。
第三章 部署要求
模型应支持导出 ONNX 并量化，以满足端侧推理。
"""
for i, c in enumerate(recursive_chunk(text, chunk_size=60, overlap=8), 1):
    out.append(f"[chunk {i}] tokens={c['token_len']:>3}  heading_path={c['heading_path']!r}")
    out.append(f"          text={c['text']!r}")
out.append("")
out.append("说明：heading_path 按章 → 节的层级累积；「第X条」属于条文正文，")
out.append("      保留在 text 中而不进 heading_path —— 这样引用时能精确到节，正文也不丢。")

out.append("")
out.append("########## 2. RRF 融合演示（模拟混合检索） ##########")
vector_ranking = ["d07", "d02", "d15", "d01", "d09"]   # 纯向量 top5
bm25_ranking   = ["d02", "d21", "d07", "d33", "d01"]   # BM25 top5
fused = rrf_fuse({"vector": vector_ranking, "bm25": bm25_ranking}, k=60)
out.append(f"向量召回 : {vector_ranking}")
out.append(f"BM25 召回: {bm25_ranking}")
out.append("RRF 融合结果（分数 = Σ 1/(60+rank)）:")
for doc_id, score in fused:
    in_v = "向量" if doc_id in vector_ranking else "    "
    in_b = "BM25" if doc_id in bm25_ranking else "    "
    out.append(f"  {doc_id}  score={score:.5f}   命中于: {in_v} {in_b}")
out.append("")
out.append("说明：d02 在两路都命中，得分最高 —— 这就是 RRF 奖励「多路一致」的机制。")
out.append("      d21/d33 只被 BM25 召回，说明关键词路补回了向量漏掉的精确词项匹配。")

io.open("demo_output.txt", "w", encoding="utf-8").write("\n".join(out))
print("已写入 demo_output.txt")
