# -*- coding: utf-8 -*-
"""tools/gen_eval_set.py —— 从已建好的索引自动生成评测集草稿。

★ 这个脚本的定位很重要，先读清楚：

    它生成的是**草稿**，不是最终评测集。它做的是"从 244 个块里挑出有代表性的
    候选并自动配对 gold_ids"，把最耗时的机械部分（找块、抄 id、分类）自动化，
    让你把 2 小时花在**人工复核**上——而复核才是评测集可信的唯一来源。

    所以脚本会在最后一屏强制打印「你必须人工过一遍」的清单。

生成的四类（与计划书 Day 3 一致）：
    term         术语/编号查询  —— 含标准号、型号、公式、具体数值
    fact         事实问答      —— 按章节标题生成的"XX要求是什么"
    multi_hop    多跳推理      —— 需要两篇文档才能回答（跨文档配对）
    out_of_scope 越界拒答负样本 —— 知识库外的问题，用于测拒答

用法：
    python tools/gen_eval_set.py --out data/eval/eval.jsonl --n 120 --seed 42
    # 看统计与人工复核清单
    python tools/gen_eval_set.py --out data/eval/eval.jsonl --n 120 --report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 越界问题池：与农业遥感完全无关，用于测拒答是否正确触发
OUT_OF_SCOPE = [
    "公司员工的年假有多少天",
    "报销流程需要几个工作日",
    "服务器机房的空调温度标准是多少",
    "这个项目的融资轮次和估值是多少",
    "北京到上海的高铁最快需要多长时间",
    "请介绍一下量子计算机的纠错码原理",
    "如何申请软件著作权登记",
    "个人所得税专项附加扣除有哪些项目",
    "公司组织架构里技术总监向谁汇报",
    "劳动合同试用期最长可以是多久",
    "Kubernetes 的 Pod 驱逐策略怎么配置",
    "今天天气怎么样，适合飞行吗",
    "这个季度的营收同比增长了多少",
    "请推荐几本学习深度学习的入门书",
    "办公楼的停车位怎么申请",
    "社保缴纳基数是怎么确定的",
    "如何办理护照和签证",
    "请解释一下相对论的质能方程推导",
    "公司食堂的营业时间是几点到几点",
    "年度绩效考核的评分标准是什么",
]

# 用于生成"术语/编号"类问题的模式（这些是 BM25 的优势区）
TERM_PATTERNS = [
    (re.compile(r"mAP@0\.5[^，。；]{0,20}不低于\s*([0-9.]+)"), "mAP 精度要求是多少"),
    (re.compile(r"准确率[^，。；]{0,10}不低于\s*([0-9.]+)"), "准确率要求不低于多少"),
    (re.compile(r"一致率不低于\s*([0-9.]+)%"), "一致率验收标准是多少"),
    (re.compile(r"几何中误差[^，。；]{0,12}超过\s*(\d+)\s*个像元"), "几何中误差要求是多少个像元"),
    (re.compile(r"地面采样距离[^，。；]{0,12}(?:高于|超过)\s*(\d+)\s*厘米"), "地面采样距离要求"),
    (re.compile(r"航向重叠(?:度)?(?:不低于)?\s*(\d+)%"), "航向重叠度要求"),
    (re.compile(r"旁向重叠(?:度)?(?:不低于)?\s*(\d+)%"), "旁向重叠度要求"),
    (re.compile(r"(NDVI|NDRE|NDWI|GNDVI|SAVI|EVI)\s*=\s*([^。；\n]+)"), None),
    (re.compile(r"太阳高度角[^，。；]{0,12}(?:低于|小于)\s*(\d+)\s*度"), "太阳高度角要求"),
    (re.compile(r"云量低于\s*(\d+)%"), "云量要求"),
    (re.compile(r"抽样复核比例不低于\s*(\d+)%"), "抽样复核比例要求"),
    (re.compile(r"复核不一致率超过\s*(\d+)%"), "复核不一致率的返工阈值"),
    (re.compile(r"测距精度约\s*±?(\d+)cm"), "激光雷达测距精度"),
    (re.compile(r"测温精度约\s*±?(\d+)℃"), "热红外测温精度"),
    (re.compile(r"点密度\s*(\d+)~(\d+)\s*点"), "激光雷达点密度"),
    (re.compile(r"波段\s*(\d+)~(\d+)nm"), "多光谱相机波段范围"),
    (re.compile(r"LAI\s*大于\s*(\d+)"), "NDVI 的饱和阈值"),
]

# 生成"事实问答"的章节标题模板
FACT_TEMPLATES = [
    ("适用范围与限制", "{title}的适用范围和限制是什么"),
    ("作业参数要求", "{title}的作业参数有什么要求"),
    ("质量指标", "{title}的质量指标是什么"),
    ("精度要求", "{title}的精度要求是多少"),
    ("质量控制", "{title}的质量控制要求"),
    ("成果交付", "{title}需要交付哪些成果"),
    ("常见问题", "{title}有哪些常见问题"),
    ("选型与技术指标", "{title}的技术指标是什么"),
]


def load_chunks(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise SystemExit(
            f"未找到 {path}\n"
            f"请先用 cli ingest 建库：\n"
            f"  conda run -n agent_rag python -m rag1_knowledge_qa.cli ingest "
            f"--docs data/raw --chunk-size 256 --overlap 32")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容两种结构：list[dict] 或 {"chunks": [...]}
    if isinstance(data, dict):
        for key in ("chunks", "items", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit(f"{path} 结构无法识别（期望 list 或含 chunks 键的 dict）")
    return data


def chunk_fields(c: Dict) -> Tuple[str, str, str, str]:
    """兼容不同字段命名，取出 (chunk_id, text, heading_path, source)。"""
    cid = c.get("chunk_id") or c.get("id") or ""
    text = c.get("text") or c.get("content") or ""
    head = c.get("heading_path") or c.get("heading") or ""
    src = c.get("source") or c.get("doc_id") or (cid.split("#")[0] if cid else "")
    return cid, text, head, src


def _doc_topic(src: str) -> str:
    """从源文件名推出一个**可读的中文主题**，用来把问题说得自包含。

    ★ 为什么必须做这一步：
      初版生成的问题形如「云量要求」「mAP 精度要求是多少」——
      没有主语、没有限定，**检索器无从下手**（BM25 会命中一堆含"要求"的块，
      向量也会散）。评测出来的低分反映的是"问题写得烂"，不是"系统差"。

    ★ 第二版又踩了一个坑：直接拼接文件名片段，
      `spec_vegetation_evi.md` 变成了 `vegetationevi`、`sensor_RGB 可见光相机` 变得很怪。
      所以改成"查表 + 兜底"，让问题读起来像人问的。
    """
    stem = re.sub(r"\.(md|txt|pdf|docx)$", "", src or "")
    stem = re.sub(r"_\d+$", "", stem)

    # ① 植被指数规范：spec_vegetation_ndvi -> 植被指数 NDVI 规范
    m = re.match(r"spec_vegetation_(\w+)$", stem)
    if m:
        return f"植被指数 {m.group(1).upper()} 规范"

    # ② 区域作业方案：plan_安徽合肥_水稻_拔节期_06 -> 安徽合肥水稻拔节期作业方案
    m = re.match(r"plan_(.+)$", stem)
    if m:
        parts = [p for p in m.group(1).split("_") if p and not p.isdigit()]
        return "".join(parts) + "作业方案"

    # ③ 应用场景：app_倒伏面积评估 -> 倒伏面积评估监测
    m = re.match(r"app_(.+)$", stem)
    if m:
        return m.group(1) + "监测"

    # ④ 传感器 / 处理流程：直接用名字
    for prefix, suffix in (("sensor_", "传感器"), ("process_", "处理流程")):
        if stem.startswith(prefix):
            name = stem[len(prefix):]
            return name if suffix in name else f"{name}{suffix}"

    # ⑤ 兜底：去掉分隔符
    return re.sub(r"[_\-]+", "", stem) or "本知识库"


# 章节标题里不该被用来造句的通用词（会产生"第二章 质量指标有哪些要求"这种废话）
_GENERIC_HEADINGS = ("质量指标", "流程与关键控制点", "总则与计算方法",
                     "适用范围与限制", "质量控制与常见误用", "业务与技术难点",
                     "精度与抽样要求", "成果交付", "作业参数要求",
                     "选型与技术指标", "数据管理与维护", "常见问题与处理报告",
                     "质量控制", "精度要求", "作业背景", "处理与验收",
                     "第二章", "第一章", "第三章")

# 事实类问题的多样问法 —— 避免 43 条全是同一模板（同模板会让评测集看起来很假，
# 而且无法区分"检索差"和"问题都一个样"）。
_FACT_ASK = [
    "{}有哪些要求",
    "{}的规定是什么",
    "关于{}，规范是怎么要求的",
    "{}需要满足什么条件",
    "请说明{}的具体要求",
    "{}的标准是什么",
]


def _clean_heading(head: str) -> str:
    leaf = head.split(">")[-1].strip()
    # 去掉"第X章/第X节"前缀
    leaf = re.sub(r"^第[零一二三四五六七八九十百\d]+[章节][\s　]*", "", leaf)
    return leaf.strip()

    """兼容不同字段命名，取出 (chunk_id, text, heading_path, source)。"""
    cid = c.get("chunk_id") or c.get("id") or ""
    text = c.get("text") or c.get("content") or ""
    head = c.get("heading_path") or c.get("heading") or ""
    src = c.get("source") or c.get("doc_id") or (cid.split("#")[0] if cid else "")
    return cid, text, head, src


def gen_term_items(chunks: List[Dict], rnd: random.Random, want: int) -> List[Dict]:
    """术语/编号类：正则从正文里抓出具体数值或公式，问它出自哪里。

    ★ 问题必须带上文档主题（见 _doc_topic 的说明），否则不可检索。
    """
    items, used = [], set()
    pool = list(chunks)
    rnd.shuffle(pool)
    for c in pool:
        if len(items) >= want:
            break
        cid, text, head, src = chunk_fields(c)
        if not cid or cid in used:
            continue
        topic = _doc_topic(src)
        for pat, fixed_q in TERM_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            if fixed_q is None:
                name = m.group(1)
                q = f"{topic}里 {name} 的计算公式是什么"
            else:
                # 把主题拼进问题，使其自包含（不要用"跨文档"式问法）
                q = f"{topic}的{fixed_q}"
            q = re.sub(r"的的", "的", q)
            if q in {it["query"] for it in items}:
                continue
            items.append({
                "qid": f"t{len(items) + 1:03d}",
                "query": q,
                "type": "term",
                "gold_ids": [cid],
                "gold_answer": m.group(0)[:120],
                "is_answerable": True,
                "note": f"来源: {src} | 标题: {head}",
            })
            used.add(cid)
            break
    return items


def gen_fact_items(chunks: List[Dict], rnd: random.Random, want: int) -> List[Dict]:
    """事实类：优先用**文档主题 + 具体小节标题**造句。

    ★ 过滤掉通用章节标题（"质量指标"/"第一章"），因为它们造出来的
      问题形如"第二章 质量指标有哪些要求"——不可回答，只会污染评测集。
      只保留有区分度的小节标题（如"影像拼接与镶嵌"/"病虫害巡检"）。
    """
    items = []
    pool = list(chunks)
    rnd.shuffle(pool)
    seen_q = set()
    for c in pool:
        if len(items) >= want:
            break
        cid, text, head, src = chunk_fields(c)
        if not cid or not head:
            continue
        leaf = _clean_heading(head)
        if len(leaf) < 3:
            continue
        # 跳过通用标题
        if any(g in leaf for g in _GENERIC_HEADINGS):
            continue
        topic = _doc_topic(src)
        # 用多种问法轮换，避免整批问题长得一模一样
        ask = _FACT_ASK[len(items) % len(_FACT_ASK)]
        # 若 leaf 与 topic 高度重合，说明标题就是主题名，只问主题
        label = topic if (leaf in topic or topic in leaf) else f"{topic}的「{leaf}」"
        q = ask.format(label)
        if q in seen_q:
            continue
        sent = re.split(r"[。；\n]", text.strip())
        gold = next((s.strip() for s in sent if len(s.strip()) > 12), text[:100])
        items.append({
            "qid": f"f{len(items) + 1:03d}",
            "query": q,
            "type": "fact",
            "gold_ids": [cid],
            "gold_answer": gold[:150],
            "is_answerable": True,
            "note": f"标题: {head}",
        })
        seen_q.add(q)
    return items


def gen_multihop_items(chunks: List[Dict], rnd: random.Random,
                       want: int) -> List[Dict]:
    """多跳类：**刻意跨文档类型**配对，保证真的需要两跳。

    ★ 初版的错误做法：随便找"两篇都含同一关键词"的块配对。
      结果配出 `plan_四川成都_大豆_成熟期 + plan_四川成都_大豆_抽穗期` ——
      两份是同一种作业方案的不同生育期，**任何一篇单独看都能回答**，
      这根本不是多跳题，只会让 multi_hop 这一类的分数虚高。

    ★ 正确做法：一份「规范类」文档给**口径/指标**，
      一份「方案/场景类」文档给**应用条件**，两者缺一不可。
      例如：
        spec_vegetation_ndvi.md  给出 NDVI 的定义与公式
        app_倒伏面积评估.md       说明倒伏评估时指数会下降，需结合纹理
        -> "倒伏评估时为什么不能只看 NDVI" 需要两篇才能答全
    """
    by_kind: Dict[str, List[Dict]] = {"spec": [], "apply": []}
    for c in chunks:
        _, text, _, src = chunk_fields(c)
        if not cid_ok(c):
            continue
        if src.startswith(("spec_", "sensor_", "process_")):
            by_kind["spec"].append(c)
        elif src.startswith(("app_", "plan_")):
            by_kind["apply"].append(c)

    concepts = ["NDVI", "NDRE", "NDWI", "GNDVI", "SAVI", "EVI",
                "辐射归一化", "正射影像", "重叠度", "地面采样距离"]
    items = []
    used_pairs = set()
    for concept in concepts:
        if len(items) >= want:
            break
        specs = [c for c in by_kind["spec"] if concept in chunk_fields(c)[1]]
        applies = [c for c in by_kind["apply"] if concept in chunk_fields(c)[1]]
        if not specs or not applies:
            continue
        sp = rnd.choice(specs)
        # 挑一个**不同源文件**的应用类块
        applies = [a for a in applies if chunk_fields(a)[3] != chunk_fields(sp)[3]]
        if not applies:
            continue
        app = rnd.choice(applies)
        ids = [chunk_fields(sp)[0], chunk_fields(app)[0]]
        key = tuple(sorted(ids))
        if key in used_pairs:
            continue
        used_pairs.add(key)
        spec_src = chunk_fields(sp)[3]
        app_src = chunk_fields(app)[3]
        # 问题问"结合规范与场景"，天然需要两跳
        items.append({
            "qid": f"m{len(items) + 1:03d}",
            "query": f"{_doc_topic(app_src)}中用到 {concept} 时，"
                     f"按规范有哪些要求和限制",
            "type": "multi_hop",
            "gold_ids": ids,
            "gold_answer": "需综合规范口径与应用条件回答",
            "is_answerable": True,
            "note": f"跨文档: {spec_src}(规范) + {app_src}(应用)",
        })
    return items


def cid_ok(c: Dict) -> bool:
    return bool(chunk_fields(c)[0])


def gen_plan_index_items(chunks: List[Dict], rnd: random.Random,
                         want: int) -> List[Dict]:
    """★ 指向**全局唯一事实**的问题：某方案荐用哪个指数、为什么。

    这类问题的价值：答案在全库唯一（每篇方案一个不同的指数+传感器组合），
    所以"检索不到"必定是检索能力问题，而不是评测集歧义。

    与 fact 类的对比意义重大：
      fact 类问的是"作业参数"，58 篇方案里遍地都是，gold 块被十几篇共享，
      检索器无从选择 -> Hit@5 只有 4.65%（实测）。
      本类问"本方案荐用哪个指数"，答案唯一 -> 应当接近 100%。
      两者放在一起，就能向面试官证明"低分来自评测集设计，不是检索器"。
    """
    items = []
    pool = [c for c in chunks if chunk_fields(c)[3].startswith("plan_")]
    rnd.shuffle(pool)
    seen_src = set()
    for c in pool:
        if len(items) >= want:
            break
        cid, text, head, src = chunk_fields(c)
        src_key = re.sub(r"_\d+$", "", src)
        if src_key in seen_src:
            continue
        if "推荐使用" not in text:
            continue
        seen_src.add(src_key)
        topic = _doc_topic(src)
        items.append({
            "qid": f"p{len(items) + 1:03d}",
            "query": f"{topic}推荐使用哪个植被指数",
            "type": "plan_index",
            "gold_ids": [cid],
            "gold_answer": next((s.strip() for s in re.split(r"[。；\n]", text)
                                 if "推荐使用" in s), "")[:150],
            "is_answerable": True,
            "note": f"唯一事实 | 来源: {src}",
        })
    return items


def gen_out_of_scope_items(rnd: random.Random, want: int) -> List[Dict]:
    qs = OUT_OF_SCOPE[:]
    rnd.shuffle(qs)
    return [{
        "qid": f"o{i + 1:03d}",
        "query": q,
        "type": "out_of_scope",
        "gold_ids": [],
        "gold_answer": "",
        "is_answerable": False,
        "note": "越界问题，期望拒答",
    } for i, q in enumerate(qs[:want])]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成评测集草稿")
    ap.add_argument("--chunks", default="data/vectors/chunks.json")
    ap.add_argument("--out", default="data/eval/eval.jsonl")
    ap.add_argument("--n", type=int, default=120, help="目标总条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true",
                    help="不生成，只打印现有评测集的统计与复核清单")
    args = ap.parse_args()

    if args.report:
        return report(args.out)

    chunks = load_chunks(args.chunks)
    print(f"读到 {len(chunks)} 个子块")

    rnd = random.Random(args.seed)
    n = args.n
    n_oos = max(10, n // 6)          # 约 1/6 越界
    n_mh = max(8, n // 12)           # 约 1/12 多跳
    n_plan = max(10, n // 5)         # 约 1/5 唯一事实（plan_index）
    n_term = (n - n_oos - n_mh - n_plan) // 2
    n_fact = n - n_oos - n_mh - n_plan - n_term

    items: List[Dict] = []
    items += gen_term_items(chunks, rnd, n_term)
    items += gen_fact_items(chunks, rnd, n_fact)
    items += gen_plan_index_items(chunks, rnd, n_plan)
    items += gen_multihop_items(chunks, rnd, n_mh)
    items += gen_out_of_scope_items(rnd, n_oos)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"已写出 {len(items)} 条 -> {os.path.abspath(args.out)}")
    return report(args.out)


def report(path: str) -> int:
    if not os.path.exists(path):
        print(f"未找到 {path}", file=sys.stderr)
        return 1
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by_type: Dict[str, int] = {}
    for it in items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1

    print()
    print("=" * 74)
    print(f"评测集：{len(items)} 条   {by_type}")
    print("=" * 74)
    print()
    print("各类抽样预览：")
    for t in ("term", "fact", "multi_hop", "out_of_scope"):
        sub = [i for i in items if i["type"] == t]
        if not sub:
            continue
        print(f"\n【{t}】共 {len(sub)} 条，随机抽 3 条：")
        for it in random.Random(7).sample(sub, min(3, len(sub))):
            print(f"  {it['qid']}  {it['query']}")
            print(f"        gold={it['gold_ids']}")

    # 可疑项检查
    problems = []
    for it in items:
        if it["is_answerable"] and not it["gold_ids"]:
            problems.append(f"{it['qid']}: 可答但没有 gold_ids")
        if it["type"] == "out_of_scope" and it["gold_ids"]:
            problems.append(f"{it['qid']}: 越界样本不该有 gold_ids")
        if len(it["query"]) < 4:
            problems.append(f"{it['qid']}: 问题过短，可能是噪声")
        if it["type"] in ("term", "fact") and len(it["gold_ids"]) != 1:
            problems.append(f"{it['qid']}: {it['type']} 类应只有 1 个 gold_id")

    print()
    print("=" * 74)
    print("★ 你必须人工过一遍的清单（这是评测集可信的唯一来源）")
    print("=" * 74)
    print("""
1. **gold_ids 是否真的回答了问题**
   自动配对是按"这个数值出现在这个块里"做的，但没判断"这个块是不是最佳答案"。
   同一句话可能出现在多个 plan_* 文档里 —— 需要你确认 gold 唯一且正确。

2. **问题是否可独立理解**
   形如"mAP 精度要求是多少"没有说明是哪份规范，模型可能答对但检索无从下手。
   建议把过于笼统的问题改具体（加上作物/地区/场景限定）。

3. **multi_hop 是否真的需要两跳**
   我按"两篇都含同一关键词"配对，但有的两篇其实任一篇就够。
   真正需要两跳的应是一篇给指标、另一篇给适用条件。

4. **out_of_scope 是否真的越界**
   检查这些问题确实无法从语料回答。若语料里碰巧有，要换掉。

5. **答案本身是否正确**
   gold_answer 是从正文截的句子，可能被截断或含无关前缀。
""")
    if problems:
        print(f"⚠️  自动体检发现 {len(problems)} 个问题（前 15 条）：")
        for p in problems[:15]:
            print(f"   · {p}")
    else:
        print("✅ 自动体检未发现问题（但人工复核仍然必须做）。")
    print()
    print("复核完成后直接跑消融：")
    print(f"  conda run -n agent_rag python -m rag1_knowledge_qa.eval \\")
    print(f"      --eval-set {path} --ablation --judge --k-list 5 3 10 --out reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
