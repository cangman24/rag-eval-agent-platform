import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

print("===== 跳过图片、DS_Store文件，下载 bge-reranker-v2-m3 =====")
snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir="models/bge-reranker-v2-m3",
    ignore_patterns=["*.jpg","*.png","*.webp",".DS_Store"]
)

print("\n✅ reranker模型下载完成！")
