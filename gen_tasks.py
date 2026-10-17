# -*- coding: utf-8 -*-
"""生成 150 条 Agent 评测任务集（单步/双步/三步各 50 条）"""
import json
import os

tasks = []

# ========== 1. 单步任务：查知识库 (50条) ==========
topics = ["NDRE", "NDVI", "NDWI", "辐射校正", "几何校正", "正射影像", "小麦头计数", "倒伏面积", "墒情", "水稻穗检测"]
for i in range(50):
    topic = topics[i % len(topics)]
    tasks.append({
        "tid": f"s{i:03d}",
        "query": f"查询{topic}的规范要求",
        "expect_tools": ["kb_search"],
        "must_contain": [],
        "must_not_contain": ["我无法"],
        "task_type": "spec_inquiry"
    })

# ========== 2. 双步任务：查工单 + 查知识 (50条) ==========
for i in range(50):
    tid = f"T-{100 + i}"
    tasks.append({
        "tid": f"d{i:03d}",
        "query": f"帮我查一下工单 {tid} 的处理记录，看看符合规范吗",
        "expect_tools": ["ticket_query", "kb_search"],
        "must_contain": [],
        "must_not_contain": ["未找到", "我无法"],
        "task_type": "record_lookup"
    })

# ========== 3. 三步任务：查工单 + 查知识 + 查指标 (50条) ==========
for i in range(50):
    tid = f"T-{200 + i}"
    metrics = ["工单量", "违规率", "复核通过率"]
    metric = metrics[i % len(metrics)]
    tasks.append({
        "tid": f"t{i:03d}",
        "query": f"做合规审查：查工单 {tid} 的记录，核对 NDRE 规范，再看近30天{metric}",
        "expect_tools": ["ticket_query", "kb_search", "metric_query"],
        "must_contain": [],
        "must_not_contain": ["我无法"],
        "task_type": "full_review"
    })

os.makedirs("data", exist_ok=True)
with open("data/agent_tasks.jsonl", "w", encoding="utf-8") as f:
    for t in tasks:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print(f"已生成 {len(tasks)} 条任务 -> data/agent_tasks.jsonl")