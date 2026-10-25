# ---------------------------------------------------------------------------
# Makefile —— 常用操作入口
#
# 为什么需要它：项目走到现在有十几条要记的命令（建库 / 消融 / 对比 run /
# 评测集体检 / demo）。记命令本身就是负担，而且容易记错参数导致
# 跑出来的数字对不上号。把它们固化成目标，就不会跑错。
#
# 用法（Windows 上用 Git Bash 或 make 的移植版）：
#
#   make test           # 765 项离线单测（不需要任何第三方包）
#   make demo           # 跑全部 demo
#   make ingest         # 用 data/raw 建库
#   make eval           # 检索层四档消融 + LLM-as-Judge
#   make eval-show      # 看最近一次 run 的指标
#   make lint           # ruff 检查
#   make docker-build   # 构建镜像
#
# ★ PY / PY2 的区别很重要：
#   PY   = 系统 Python，只跑 core/ 与测试（零第三方依赖）
#   PY2  = 装了 torch/transformers 的环境，跑建库与评测
# 这样"验证代码是好的"与"跑真模型"互不阻塞 ——
# 环境装不上也能先确认代码没问题。
# ---------------------------------------------------------------------------

PY  ?= python
PY2 ?= conda run -n agent_rag python
EVAL_SET ?= data/eval/eval.jsonl
OUT ?= reports
CHUNK_SIZE ?= 512
OVERLAP ?= 64

.DEFAULT_GOAL := help
.PHONY: help test test-fast demo ingest eval eval-show eval-compare gen-corpus \
        gen-eval review-eval lint fmt typecheck serve docker-build docker-up \
        docker-down check-size clean

# ------------------------------------------------------------------ 帮助
help:  ## 显示所有可用目标
	@echo "常用目标："
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ 测试
test:  ## 跑全部离线单测（765 项，不需要第三方依赖）
	$(PY) -m unittest discover -s tests -v

test-fast:  ## 快速跑一遍（静默模式，只看结果）
	$(PY) -m unittest discover -s tests

demo:  ## 跑全部 demo（每个都应 exit 0）
	@for d in demo_core demo_pipeline demo_eval demo_agent demo_ingest \
	          demo_service demo_agent2 demo_evalrun demo_agenteval demo_mcp; do \
	  printf "  %-18s " $$d; \
	  if $(PY) demos/$$d.py > /dev/null 2>&1; then echo "OK"; else echo "FAIL"; fi; \
	done

# ------------------------------------------------------------------ 数据
gen-corpus:  ## 生成 83 篇语料（可复现，固定 seed）
	$(PY) tools/gen_corpus.py --out data/raw --n 80 --seed 42

ingest:  ## 用 data/raw 建库（需要 PY2 环境 + BGE 模型）
	$(PY2) -m rag1_knowledge_qa.cli ingest \
	  --docs data/raw --chunk-size $(CHUNK_SIZE) --overlap $(OVERLAP)

gen-eval:  ## 从已建索引生成评测集草稿
	$(PY) tools/gen_eval_set.py --out $(EVAL_SET) --n 120 --seed 42

review-eval:  ## 打印评测集统计与必做的复核清单
	$(PY) tools/gen_eval_set.py --out $(EVAL_SET) --report

# ------------------------------------------------------------------ 评测
eval:  ## 检索层四档消融 + LLM-as-Judge（一条命令出可复现报告）
	$(PY2) -m rag1_knowledge_qa.eval \
	  --eval-set $(EVAL_SET) --ablation --judge --k-list 5 3 10 \
	  --out $(OUT) --notes "chunk=$(CHUNK_SIZE) overlap=$(OVERLAP)"

eval-show:  ## 看最近一次 run 的指标
	$(PY2) -m rag1_knowledge_qa.eval --show --out $(OUT)

eval-compare:  ## 对比两次 run：make eval-compare A=reports/run-x B=reports/run-y
	@test -n "$(A)" -a -n "$(B)" || \
	  (echo "用法: make eval-compare A=reports/run-x B=reports/run-y"; exit 1)
	$(PY2) -m rag1_knowledge_qa.eval --compare $(A) $(B)

grid:  ## chunk_size 网格搜索（找最优切分，用于回答"为什么选 512"）
	@for cs in 256 512 768; do \
	  for ov in 32 64 128; do \
	    echo "=== chunk_size=$$cs overlap=$$ov ==="; \
	    $(PY2) -m rag1_knowledge_qa.cli ingest --docs data/raw \
	      --chunk-size $$cs --overlap $$ov; \
	    $(PY2) -m rag1_knowledge_qa.eval --eval-set $(EVAL_SET) \
	      --out $(OUT) --notes "cs=$$cs ov=$$ov"; \
	  done; \
	done

# ------------------------------------------------------------------ 服务
serve:  ## 启动 HTTP 服务（需 pip install -e ".[ml,serve]"）
	$(PY2) -m rag1_knowledge_qa.server

# ------------------------------------------------------------------ 质量
lint:  ## ruff 静态检查
	$(PY) -m ruff check .

fmt:  ## ruff 自动格式化
	$(PY) -m ruff format .

typecheck:  ## mypy 类型检查
	$(PY) -m mypy core rag1_knowledge_qa rag2_kb_agent

# ------------------------------------------------------------------ Docker
docker-build:  ## 构建镜像
	docker build -t domain-rag-qa:0.1.0 .

docker-up:  ## 起服务（需要先准备好 .env 与 models/）
	docker compose up --build -d && docker compose logs -f rag

docker-down:  ## 停服务
	docker compose down

# ------------------------------------------------------------------ 运维
check-size:  ## 检查有没有不该入库的大文件（>100KB）
	@echo ">100KB 且未被忽略的文件："
	@git ls-files | while read f; do \
	  if [ -f "$$f" ] && [ $$(wc -c < "$$f") -gt 102400 ]; then \
	    printf "  %8d B  %s\n" $$(wc -c < "$$f") "$$f"; \
	  fi; \
	done; echo "  （空则正常）"

clean:  ## 清理缓存与临时产物（不动语料、索引、报告）
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf tests/.tmp .pytest_cache .ruff_cache .mypy_cache 2>/dev/null || true
	@echo "已清理缓存"
