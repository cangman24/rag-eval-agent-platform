# create_demo_corpus.py
from pathlib import Path

def build_demo_corpus():
    raw_dir = Path("data") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "crop_ndvi.md": """# NDVI植被指数
NDVI（归一化植被指数），计算公式：NDVI = (NIR - RED) / (NIR + RED)。
无人机多光谱影像可以计算NDVI，用来判断作物长势、识别植被胁迫。
NDVI取值范围 [-1, 1]。健康绿色植被NDVI一般大于0.6；裸土大约0.1~0.3；水体一般为负数。

## NDRE指数
NDRE是归一化红边指数，使用红边波段替代红光波段，适合高植被覆盖、作物冠层浓密场景。
NDRE = (NIR - RedEdge) / (NIR + RedEdge)，大田后期作物监测效果优于NDVI。
""",
        "uav_mosaic.md": """# 无人机影像拼接（Mosaic）
无人机航测需要对多张瓦片影像做辐射校正、配准，生成正射影像DOM。
多光谱瓦片拼接容易出现辐射色差，需要跨架次辐射归一化消除明暗条带。

## DSM 数字表面模型
DSM记录地表+植被顶部高度，从多光谱或者RGB点云生成；可用于株高估算。
""",
        "data_process.md": """# 大田无人机数据预处理流程
1. 原始影像采集
2. 辐射定标
3. 影像拼接，生成瓦片
4. 计算植被指数 NDVI / NDRE / GNDVI
5. 掩膜：去除土壤、阴影、云
6. 语义分割（U-Net）分割作物区域
"""
    }

    for fname, content in files.items():
        fp = raw_dir / fname
        fp.write_text(content, encoding="utf-8")
        print(f"✅ Created: {fp}")
    print("\n🎉 Demo corpus generated under data/raw, ready for ingest.")

if __name__ == "__main__":
    build_demo_corpus()
