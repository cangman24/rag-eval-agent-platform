# ---------------------------------------------------------------------------
# Dockerfile —— 多阶段构建，CPU 版
#
# ★ 三个关键取舍，都写在这里方便面试时讲：
#
#   ① 为什么用 `python:3.10-slim` 而不是 `nvidia/cuda`？
#      因为镜像层的价值是「可复现地把服务跑起来」，而 GPU 由宿主机提供。
#      用 CPU 版 torch（约 200MB）而不是 cu121 版（约 2.5GB），
#      镜像从 ~7GB 压到 ~2GB。真要 GPU 推理，用
#      `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121`。
#
#   ② 为什么先 COPY pyproject.toml 再 COPY 源码？
#      Docker 层缓存。依赖几乎不变、源码天天改。
#      顺序对了，改一行 py 只需重建最后一层（秒级）。
#
#   ③ 为什么模型和数据用 volume 而不是 COPY 进去？
#      模型 6.4GB、索引 90KB 但可由语料重建。打进镜像会让镜像臃肿且
#      每次换语料都要重新 build。用挂载，换模型只改环境变量。
#
#   构建：
#     docker build -t domain-rag-qa:0.1.0 .
#   运行：
#     docker run --rm -p 8000:8000 --env-file .env \
#       -v "$PWD/models:/app/models:ro" \
#       -v "$PWD/data:/app/data" \
#       domain-rag-qa:0.1.0
# ---------------------------------------------------------------------------

# ============================================================ 阶段一：依赖
FROM python:3.10-slim AS deps

# 允许切 cu121 轮子（默认 CPU 版）
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ★ 先只拷依赖描述文件，让依赖层可缓存
COPY pyproject.toml README.md ./

# 装运行时依赖。注意 transformers 锁 4.44.x：
# sentence-transformers / FlagEmbedding 尚不兼容 5.x
RUN pip install --upgrade pip setuptools wheel \
 && pip install --index-url ${TORCH_INDEX} torch \
 && pip install \
      "transformers==4.44.2" \
      "sentence-transformers>=3.0,<4" \
      "chromadb>=0.5,<0.6" \
      "openai>=1.51,<2" \
      "python-dotenv>=1.0" \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.30" \
      "sse-starlette>=2.1" \
      "pydantic>=2.8" \
      "pypdf>=4.3" \
      "python-docx>=1.1"

# ============================================================ 阶段二：运行
FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    # 让 `python -m rag1_knowledge_qa.cli` 能直接 import core
    PYTHONPATH=/app

WORKDIR /app

# 从 deps 阶段搬运已装好的 site-packages（这是多阶段的收益：镜像里
# 不含编译工具链，体积和攻击面都更小）
COPY --from=deps /usr/local/lib/python3.10/site-packages \
                /usr/local/lib/python3.10/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# 应用代码（这一层会随源码变更而重建）
COPY core/ ./core/
COPY rag1_knowledge_qa/ ./rag1_knowledge_qa/
COPY rag2_kb_agent/ ./rag2_kb_agent/
COPY tools/ ./tools/

# 非 root 运行（安全基线）。挂载的 data/ 需要写权限，故显式建目录并授权。
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data /app/models /app/reports \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# ★ 健康检查打到 /health，它不触发检索也不调用 LLM（见 core.service.health），
#   所以不会因为模型冷启动或 API 抖动而误报不健康。
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# start-period 给了 90 秒：装配时要加载 BGE-M3 + Reranker（实测约 6 秒，
# 但冷启动叠加磁盘 IO 可能到几十秒），留足余量避免被误杀。
CMD ["uvicorn", "rag1_knowledge_qa.server:app", "--host", "0.0.0.0", "--port", "8000"]
