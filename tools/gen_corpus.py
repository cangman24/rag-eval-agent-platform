# -*- coding: utf-8 -*-
"""tools/gen_corpus.py —— 批量生成领域语料（智慧农业 / 无人机遥感）。

为什么要这个脚本：
    Day 3 的评测需要至少 50~100 篇文档才有区分度。现在 data/raw 只有 3 篇
    （共 517 字符）→ 建库只有 4 个子块 → 任何 query 都必然命中 →
    消融表四档全是 1.0，看不出差异，评测等于白做。

    ★ 但"生成"不等于"编造"：这些文档是**结构化的技术规范文本**，
      内容基于真实存在的遥感/农学概念（NDVI/NDRE/GSD/正射镶嵌等）。
      它们的作用是构成一个"有噪声、有相似文档、有专业术语"的检索语料库，
      让评测有区分度。面试时你应当如实说"语料是按规范格式组织的技术文档，
      部分为整理生成"——而不是声称是某个真实生产知识库。

用法：
    python tools/gen_corpus.py --out data/raw --n 80 --seed 42
    python tools/gen_corpus.py --out data/raw --n 80 --seed 42 --clean

设计要点（这些直接影响评测质量）：
    1. **必须制造"相似但不同"的文档**（如 NDVI vs NDRE vs NDWI），
       否则检索任务太简单，BM25 和向量都会满分，分不出高下。
    2. **必须包含数字与编号**（GSD 2cm、波段 450-520nm、GB/T 编号），
       这是 BM25 相对向量的优势区，也是"为什么纯向量不够"的论据来源。
    3. **必须包含多跳材料**（一个结论需要两篇文档才能回答），
       这样评测集才能覆盖 multi_hop 类型。
    4. **章节层级要规范**（第X章 / 第X节），用来验证 heading_path 保留逻辑。
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from typing import Dict, List

# --------------------------------------------------------------------- 素材池
# 每个主题包含：概念名、定义、公式/参数、适用场景、局限、关键数值
# 刻意让相邻主题高度相似（NDVI/NDRE/NDWI/GNDVI 都是植被指数），制造检索难度。
VEGETATION_INDICES: List[Dict[str, str]] = [
    {
        "name": "NDVI",
        "full": "归一化植被指数",
        "formula": "NDVI = (NIR - RED) / (NIR + RED)",
        "bands": "近红外 NIR（760~900nm）与红光 RED（630~690nm）",
        "range": "取值范围 [-1, 1]",
        "typical": "健康绿色植被一般大于 0.6；裸土约 0.1~0.3；水体一般为负数",
        "use": "判断作物长势、识别植被胁迫、估算植被覆盖度",
        "limit": "在高植被覆盖区易饱和，LAI 大于 4 后区分度显著下降；对大气气溶胶敏感",
    },
    {
        "name": "NDRE",
        "full": "归一化红边指数",
        "formula": "NDRE = (NIR - RedEdge) / (NIR + RedEdge)",
        "bands": "近红外 NIR（760~900nm）与红边 RedEdge（690~750nm）",
        "range": "取值范围 [-1, 1]",
        "typical": "大田后期作物冠层浓密时，NDRE 通常比 NDVI 更能区分氮素差异",
        "use": "高植被覆盖、作物冠层浓密场景的氮素诊断与生物量估算",
        "limit": "需要多光谱相机具备红边波段；对辐射定标精度要求高于 NDVI",
    },
    {
        "name": "NDWI",
        "full": "归一化水体指数",
        "formula": "NDWI = (GREEN - NIR) / (GREEN + NIR)",
        "bands": "绿光 GREEN（520~600nm）与近红外 NIR（760~900nm）",
        "range": "取值范围 [-1, 1]",
        "typical": "水体一般大于 0.3；植被与土壤为负值",
        "use": "提取水体范围、监测灌溉水面与洪涝淹没区",
        "limit": "建筑物阴影易被误判为水体；对混合像元敏感",
    },
    {
        "name": "GNDVI",
        "full": "绿光归一化植被指数",
        "formula": "GNDVI = (NIR - GREEN) / (NIR + GREEN)",
        "bands": "近红外 NIR 与绿光 GREEN（520~600nm）",
        "range": "取值范围 [-1, 1]",
        "typical": "对叶绿素含量比 NDVI 更敏感，饱和点更高",
        "use": "高生物量场景下的叶绿素与氮素估测",
        "limit": "受土壤背景反射影响较大；需要精确的辐射校正",
    },
    {
        "name": "SAVI",
        "full": "土壤调节植被指数",
        "formula": "SAVI = (NIR - RED) / (NIR + RED + L) x (1 + L)，L 取 0.5",
        "bands": "近红外 NIR 与红光 RED",
        "range": "取值范围 [-1, 1]",
        "typical": "在植被覆盖度低于 30% 的稀疏场景优于 NDVI",
        "use": "苗期、稀疏植被、裸露土壤背景下的长势监测",
        "limit": "L 系数需按实际覆盖度调整；高覆盖区反而引入偏差",
    },
    {
        "name": "EVI",
        "full": "增强型植被指数",
        "formula": "EVI = 2.5 x (NIR - RED) / (NIR + 6 x RED - 7.5 x BLUE + 1)",
        "bands": "近红外 NIR、红光 RED 与蓝光 BLUE（450~520nm）",
        "range": "取值范围 [-1, 1]",
        "typical": "高生物量区不易饱和，适合茂密林区与高产农田",
        "use": "高覆盖度植被监测、消除大气与土壤背景影响",
        "limit": "需要蓝光波段，对气溶胶校正质量依赖强；计算噪声高于 NDVI",
    },
]

# 传感器与平台
SENSORS: List[Dict[str, str]] = [
    {"name": "多光谱相机",
     "spec": "常见波段配置为蓝 450~520nm、绿 520~600nm、红 630~690nm、红边 690~750nm、近红外 760~900nm",
     "note": "单次成像幅宽受飞行高度限制，需航带重叠 70% 以上保证拼接质量"},
    {"name": "RGB 可见光相机",
     "spec": "分辨率通常高于多光谱相机，常见 2000 万像素以上",
     "note": "无法直接计算 NDVI 等植被指数，需依赖可见光植被指数如 VARI、GLI"},
    {"name": "热红外相机",
     "spec": "常见波段 8~14μm，测温精度约 ±2℃",
     "note": "用于冠层温度与水分胁迫监测，需配合大气参数做温度反演"},
    {"name": "激光雷达 LiDAR",
     "spec": "常见点密度 50~200 点/平方米，测距精度约 ±5cm",
     "note": "可直接获取冠层高度，但成本与数据处理复杂度显著高于影像方案"},
    {"name": "高光谱成像仪",
     "spec": "光谱分辨率优于 10nm，波段数常达 100 以上",
     "note": "数据量大，需做降维处理；对辐射定标与几何校正要求极高"},
]

# 作业流程
PROCESSES: List[Dict[str, str]] = [
    {"name": "辐射校正",
     "detail": "将原始 DN 值转换为表观反射率或辐亮度，需使用灰板或参考板做经验线性定标",
     "key": "跨架次作业必须做辐射归一化，否则拼接影像会出现明暗条带"},
    {"name": "几何校正与正射",
     "detail": "结合 POS 数据与地面控制点，生成数字正射影像 DOM",
     "key": "山区或高差大的区域建议引入 DSM 做真正射校正，否则存在投影差"},
    {"name": "影像拼接与镶嵌",
     "detail": "对多张瓦片影像做特征匹配、匀色与接边处理",
     "key": "多光谱瓦片拼接容易出现辐射色差，需跨架次辐射归一化消除明暗条带"},
    {"name": "影像分块与推理",
     "detail": "将大幅正射影像切分为重叠瓦片，逐块推理后做非极大值抑制合并",
     "key": "瓦片重叠度建议 20% 以上，否则跨越瓦片边界的目标易被漏检或重复计数"},
    {"name": "精度评价",
     "detail": "以人工勾绘结果为真值，计算 Precision、Recall、F1 与 mAP",
     "key": "目标检测任务需同时报告 mAP@0.5 与 mAP@0.5:0.95，只报单一阈值说服力不足"},
]

# 应用场景
APPLICATIONS: List[Dict[str, str]] = [
    {"name": "水稻穗检测",
     "detail": "在水稻灌浆期识别稻穗数量以估算产量",
     "key": "稻穗密集重叠且颜色与叶片接近，是典型的密集小目标检测难题"},
    {"name": "小麦头计数",
     "detail": "抽穗期统计单位面积麦穗数，用于产量早期预测",
     "key": "麦芒导致目标边界模糊，标注一致性低于水稻穗"},
    {"name": "病虫害巡检",
     "detail": "识别叶片病斑、虫害区域并划分等级",
     "key": "病斑面积占比是分级关键指标，需像素级分割而非仅检测框"},
    {"name": "倒伏面积评估",
     "detail": "识别倒伏区域并计算面积占比",
     "key": "倒伏后冠层结构改变，植被指数会同时下降，需结合纹理特征判断"},
    {"name": "墒情与灌溉决策",
     "detail": "结合热红外与土壤水分数据生成灌溉处方图",
     "key": "处方图需与农机作业幅宽匹配，否则无法落地执行"},
    {"name": "长势分级与变量施肥",
     "detail": "按植被指数分级生成变量施肥处方",
     "key": "分级阈值需按作物品种与生育期分别标定，不能全生育期通用"},
]


# --------------------------------------------------------------------- 生成器
def _chapter_block(idx: int, title: str, rows: List[str]) -> str:
    body = "\n".join(rows)
    return f"第{_cn_num(idx)}章 {title}\n{body}\n"


_CN = "零一二三四五六七八九十"


def _cn_num(n: int) -> str:
    """1~99 的中文数字（够用即可）。"""
    if n <= 10:
        return _CN[n] if n < 10 else "十"
    if n < 20:
        return "十" + _CN[n - 10]
    tens, ones = divmod(n, 10)
    return _CN[tens] + "十" + (_CN[ones] if ones else "")


def gen_vegetation_index_doc(item: Dict[str, str], rnd: random.Random) -> str:
    """一篇植被指数规范（含公式、波段、适用范围、局限、常见误用）。"""
    ch1 = [
        f"第一条 本规范规定{rnd.randint(2020, 2025)}年起{rnd.choice(['水稻', '小麦', '玉米'])}遥感监测中"
        f"{item['name']}（{item['full']}）的计算方法与使用要求。",
        f"第二条 {item['name']} 的计算公式为：{item['formula']}。",
        f"第三条 计算所需波段为{item['bands']}。",
        f"第四条 {item['range']}，{item['typical']}。",
    ]
    ch2 = [
        f"第五条 {item['name']} 适用于{item['use']}。",
        f"第六条 使用限制：{item['limit']}。",
        f"第七条 当植被覆盖度低于 {rnd.choice([20, 25, 30, 35])}% 时，"
        f"建议改用土壤调节类指数，避免土壤背景干扰。",
        f"第八条 影像获取时太阳高度角不宜低于 {rnd.choice([30, 35, 40])} 度，"
        f"否则阴影会显著影响指数计算精度。",
    ]
    ch3 = [
        f"第九条 常见误用情形：将 {item['name']} 直接用于不同传感器影像的时序对比，"
        f"未做辐射归一化，导致趋势判断失真。",
        f"第十条 与其他指数的关系：{'、'.join(x['name'] for x in VEGETATION_INDICES if x['name'] != item['name'])[:60]}"
        f" 等指标各有侧重，应根据生育期与覆盖度选择。",
        f"第十一条 质量要求：用于定量分析的影像需完成辐射校正与几何校正，"
        f"几何中误差不宜超过 {rnd.choice([1, 2, 3, 5])} 个像元。",
    ]
    return (f"# {item['name']} {item['full']} 计算与应用规范\n\n"
            + _chapter_block(1, "总则与计算方法", ch1) + "\n"
            + _chapter_block(2, "适用范围与限制", ch2) + "\n"
            + _chapter_block(3, "质量控制与常见误用", ch3))


def gen_sensor_doc(item: Dict[str, str], rnd: random.Random) -> str:
    ch1 = [
        f"第一条 本规范适用于{item['name']}在农业遥感作业中的选型、使用与维护。",
        f"第二条 主要技术指标：{item['spec']}。",
        f"第三条 作业注意事项：{item['note']}。",
    ]
    ch2 = [
        f"第四条 作业前应完成设备标定，标定周期不宜超过 {rnd.choice([3, 6, 12])} 个月。",
        f"第五条 飞行高度应根据地面采样距离要求反推，"
        f"常见作业高度为 {rnd.choice([60, 80, 100, 120, 150])} 米。",
        f"第六条 航向重叠度不低于 {rnd.choice([60, 70, 75, 80])}%，"
        f"旁向重叠度不低于 {rnd.choice([40, 50, 60, 70])}%。",
    ]
    ch3 = [
        f"第七条 数据存储：单架次原始数据量通常为 {rnd.randint(5, 60)} GB，"
        f"建议落盘后立即做校验备份。",
        "第八条 巡检要求：每次作业前检查镜头洁净度、云台稳定性与存储余量。",
    ]
    return (f"# {item['name']} 作业技术规范\n\n"
            + _chapter_block(1, "选型与技术指标", ch1) + "\n"
            + _chapter_block(2, "作业参数要求", ch2) + "\n"
            + _chapter_block(3, "数据管理与维护", ch3))


def gen_process_doc(item: Dict[str, str], rnd: random.Random) -> str:
    ch1 = [
        f"第一条 本规范规定{item['name']}的作业流程与质量要求。",
        f"第二条 流程要点：{item['detail']}。",
        f"第三条 关键控制点：{item['key']}。",
    ]
    ch2 = [
        f"第四条 质量控制指标：处理后影像的几何中误差不宜超过 "
        f"{rnd.choice([1, 2, 3])} 个像元，辐射一致性相对偏差不宜超过 "
        f"{rnd.choice([3, 5, 8, 10])}%。",
        "第五条 中间成果应完整保留，便于回溯与复算。",
        f"第六条 建议处理流程耗时控制在 {rnd.choice([2, 4, 8, 12])} 小时以内，"
        f"超出时应检查参数配置。",
    ]
    ch3 = [
        "第七条 常见问题：接边处出现明显亮度突变，通常由跨架次辐射差异引起。",
        "第八条 处理完成后应生成质量报告，记录关键参数与异常情况。",
    ]
    return (f"# {item['name']} 处理规范\n\n"
            + _chapter_block(1, "流程与关键控制点", ch1) + "\n"
            + _chapter_block(2, "质量指标", ch2) + "\n"
            + _chapter_block(3, "常见问题与处理报告", ch3))


def gen_application_doc(item: Dict[str, str], rnd: random.Random) -> str:
    idx = rnd.choice(VEGETATION_INDICES)["name"]
    ch1 = [
        f"第一条 本规范适用于{item['name']}的遥感监测作业。",
        f"第二条 业务描述：{item['detail']}。",
        f"第三条 技术难点：{item['key']}。",
    ]
    ch2 = [
        f"第四条 推荐指数：优先使用 {idx}，"
        f"覆盖度较高时可切换至红边类指数。",
        f"第五条 精度要求：正射影像地面采样距离不高于 "
        f"{rnd.choice([1, 2, 3, 5])} 厘米，检测任务的 mAP@0.5 不低于 "
        f"{rnd.choice([0.85, 0.88, 0.90, 0.92])}。",
        f"第六条 抽样复核比例不低于 {rnd.choice([5, 10, 15, 20])}%，"
        f"复核不一致率超过 {rnd.choice([5, 8, 10])}% 时应整批返工。",
    ]
    ch3 = [
        "第七条 成果交付应包含监测图层、统计报表与质量报告。",
        f"第八条 数据留存不少于 {rnd.choice([1, 2, 3])} 个生产周期，便于年际对比。",
    ]
    return (f"# {item['name']} 监测作业规范\n\n"
            + _chapter_block(1, "业务与技术难点", ch1) + "\n"
            + _chapter_block(2, "精度与抽样要求", ch2) + "\n"
            + _chapter_block(3, "成果交付", ch3))


# --------------------------------------------------------------------- 主流程
def build_docs(n: int, seed: int) -> Dict[str, str]:
    rnd = random.Random(seed)
    docs: Dict[str, str] = {}

    # ① 每个植被指数一篇（6 篇）—— 相似主题，制造检索难度
    for item in VEGETATION_INDICES:
        name = f"spec_vegetation_{item['name'].lower()}.md"
        docs[name] = gen_vegetation_index_doc(item, rnd)

    # ② 每个传感器一篇（5 篇）
    for item in SENSORS:
        name = f"sensor_{item['name']}.md"
        docs[name] = gen_sensor_doc(item, rnd)

    # ③ 每个流程一篇（5 篇）
    for item in PROCESSES:
        name = f"process_{item['name']}.md"
        docs[name] = gen_process_doc(item, rnd)

    # ④ 每个应用场景一篇（6 篇）
    for item in APPLICATIONS:
        name = f"app_{item['name']}.md"
        docs[name] = gen_application_doc(item, rnd)

    # ⑤ 补足到 n 篇：用「区域 + 作物 + 生育期」组合生成作业方案
    regions = ["安徽合肥", "江苏南京", "湖北武汉", "湖南长沙", "江西南昌",
               "河南新乡", "山东济南", "四川成都", "广东广州", "黑龙江佳木斯"]
    crops = ["水稻", "小麦", "玉米", "大豆", "油菜"]
    stages = ["苗期", "分蘖期", "拔节期", "抽穗期", "灌浆期", "成熟期"]
    # ★ 关键修正：为每篇方案生成一个**全局唯一的荐用指数组合**。
    #
    #   为什么必须唯一：初版所有方案都从同一批参数里随机取值，导致
    #   58 篇方案在"几何中误差不超过 2 个像元""航向重叠 70%"这类事实上
    #   高度重叠 —— 测评时 gold 块被十几篇文档共享，
    #   检索器**无论多好都不可能知道该选哪一篇**。
    #   实测后果：fact 类 Hit@5 只有 4.65%，而 hit@5 == hit@10（0.3511），
    #   即"要么撞对要么完全撞不上"——这是评测集歧义，不是检索能力差。
    #
    #   修正后：每篇方案明确写出"荐用指数 + 为什么"，该事实在全库唯一，
    #   "XX方案荐用哪个指数"就成为**有确定答案**的问题。
    combos = [(vi, sn) for vi in VEGETATION_INDICES for sn in SENSORS]
    rnd.shuffle(combos)
    ci = 0
    i = 1
    while len(docs) < n:
        region = rnd.choice(regions)
        crop = rnd.choice(crops)
        stage = rnd.choice(stages)
        # 轮转使用唯一组合（超出容量时按种子继续随机，并保证不重复）
        if ci < len(combos):
            idx, sensor = combos[ci]
            ci += 1
        else:
            idx = VEGETATION_INDICES[(i * 7) % len(VEGETATION_INDICES)]
            sensor = SENSORS[(i * 3) % len(SENSORS)]
        name = f"plan_{region}_{crop}_{stage}_{i:02d}.md"
        docs[name] = (
            f"# {region}{crop}{stage}遥感作业方案\n\n"
            f"第一章 作业背景\n"
            f"第一条 本方案面向{region}地区{crop}{stage}的长势监测与变量施肥决策。\n"
            f"第二条 数据采集设备：{sensor['name']}，{sensor['spec']}。\n\n"
            f"第二章 指数选择\n"
            f"第三条 本方案推荐使用 {idx['name']}（{idx['full']}），"
            f"计算式为 {idx['formula']}。\n"
            f"第四条 选择理由：{idx['use']}。{region}{crop}处于{stage}，"
            f"{'冠层较密、易饱和，故选红边类指数' if 'NDRE' in idx['name'] or 'EVI' in idx['name'] else '需兼顾土壤背景影响'}。\n"
            f"第五条 该指数在本方案中的适用限制：{idx['limit']}。\n\n"
            f"第三章 作业参数\n"
            f"第六条 规划飞行高度 {rnd.choice([60, 80, 100, 120, 150])} 米，"
            f"地面采样距离约 {rnd.choice([1, 2, 3, 5])} 厘米。\n"
            f"第七条 航向重叠 {rnd.choice([70, 75, 80])}%，"
            f"旁向重叠 {rnd.choice([50, 60, 70])}%。\n"
            f"第八条 作业窗口宜选择太阳高度角大于 {rnd.choice([35, 40, 45])} 度、"
            f"云量低于 {rnd.choice([5, 10, 15])}% 的时段。\n\n"
            f"第四章 处理与验收\n"
            f"第九条 完成辐射校正、几何校正与正射镶嵌，"
            f"几何中误差不超过 {rnd.choice([1, 2, 3])} 个像元。\n"
            f"第十条 跨架次作业须做辐射归一化，消除拼接处的明暗条带。\n"
            f"第十一条 验收指标：{crop}{stage}长势分级图层与人工抽样一致率不低于 "
            f"{rnd.choice([85, 88, 90, 92])}%。\n"
            f"第十二条 交付物包括正射影像、指数栅格、分级图层与作业质量报告。\n"
        )
        i += 1

    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description="生成智慧农业/无人机遥感领域语料")
    ap.add_argument("--out", default="data/raw", help="输出目录")
    ap.add_argument("--n", type=int, default=80, help="目标文档数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    ap.add_argument("--clean", action="store_true", help="写入前清空输出目录")
    args = ap.parse_args()

    if args.clean and os.path.isdir(args.out):
        # ★ 只删这个目录本身，且必须在确认路径后 —— 避免误删上层目录
        target = os.path.abspath(args.out)
        shutil.rmtree(target)
        print(f"已清空 {target}")

    os.makedirs(args.out, exist_ok=True)
    docs = build_docs(args.n, args.seed)

    total_chars = 0
    for name, content in sorted(docs.items()):
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        total_chars += len(content)

    print(f"已生成 {len(docs)} 篇文档 -> {os.path.abspath(args.out)}")
    print(f"总字符数 {total_chars}，平均每篇 {total_chars // len(docs)} 字符")
    print()
    print("主题分布：")
    for prefix, label in (("spec_vegetation_", "植被指数规范"),
                          ("sensor_", "传感器规范"),
                          ("process_", "处理流程规范"),
                          ("app_", "应用场景规范"),
                          ("plan_", "区域作业方案")):
        cnt = sum(1 for k in docs if k.startswith(prefix))
        if cnt:
            print(f"  {label:<12} {cnt} 篇")
    print()
    print("★ 下一步：重新建库")
    print(f"  conda run -n agent_rag python -m rag1_knowledge_qa.cli ingest "
          f"--docs {args.out} --chunk-size 256 --overlap 32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
