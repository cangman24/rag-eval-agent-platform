# domain-rag-qa · 领域 RAG 问答 + 受控 Agent 平台（双项目）

> 一套共享基建 `core/`（21 个模块，19 个纯标准库）上的两个可独立运行的应用：
>
> - **项目一 `rag1_knowledge_qa`**：垂直领域 RAG 知识问答——混合检索、父子块双层索引、低置信拒答、引用溯源、逐层消融评测。
> - **项目二 `rag2_kb_agent`**：企业知识库 Agent 平台——受控状态机、Checkpoint 续跑、工具白名单、三道熔断、三层记忆与上下文压缩。
>
> 全部业务逻辑**零第三方依赖可单测**（767 项，约 1 秒）；评测报告带运行指纹与评测集哈希，**可复现**。
> 需要 GPU 与 API Key 的真模型链路与离线逻辑之间由"可注入接缝"隔开——接上 BGE-M3 与 DeepSeek 后，核心代码一行不改。

![python](https://img.shields.io/badge/python-%3E%3D3.10-blue) ![tests](https://img.shields.io/badge/tests-767%20passing-brightgreen) ![deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen) ![license](https://img.shields.io/badge/license-MIT-blue)

---

## 快速开始（5 分钟，无需 GPU / API Key / 任何第三方包）

```bash
python -m unittest discover -s tests   # 767 项单测，约 1 秒（12 项因可选依赖跳过）
python demos/demo_ingest.py            # 全流程：文件 → 父子块切分 → 双索引 → 混合检索 → Agent → 评测
python demos/demo_agent2.py            # Agent 六场景：多步任务 / 幻觉工具 / 越权拦截 / 兜底 / 预算 / 断点续跑
```

其余 7 个离线 demo 见 `demos/`（切分与 RRF 融合、检索链路、消融、服务层与 SSE、评测运行器、Agent 层消融）。
`make demo` 可一键冒烟全部 9 个；`make test` 跑测试；`make lint` 静态检查。

---

## 项目一 · 领域 RAG 知识问答（`rag1_knowledge_qa`）

面向**领域文档分散、术语密集、编号型号多**的场景：答案必须可溯源、检索必须可度量、调参必须可回归。

**链路**：多格式解析 → 父子块切分（子块检索、父块生成）→ BM25 + 向量双路召回 → RRF 融合 → 交叉编码精排 → 低置信拒答 / 带引用生成 → CLI / FastAPI / SSE 输出。

**真实语料消融**（83 篇文档，评测集 119 条 / 99 条可答，本机 BGE-M3 嵌入 + bge-reranker-v2-m3 精排，报告与运行指纹见 [`reports/run-20261016-100221-991c15/`](reports/run-20261016-100221-991c15/)）：

| 配置 | Hit@5 | MRR@5 | NDCG@5 | P95 延迟 |
|---|---|---|---|---|
| A 纯向量 | 0.6061 | 0.4524 | 0.4660 | 48.6 ms |
| B + BM25 / RRF | 0.6667 | 0.4843 | 0.5049 | 52.4 ms |
| C + Reranker 精排 | **0.6970** | **0.5827** | **0.5874** | 183.5 ms |
| D + 编号规则召回 | 0.6970 | 0.5827 | 0.5874 | 180.2 ms |

> D 档与 C 档持平：规则召回只在含编号/型号的查询上生效，当前评测集该类样本不足——这是正确的观察，不是隐藏的问题。
> `reports/` 另保留了从 Hit@5 0.35 到 0.70 的完整调参历史（5 次运行指纹）。

**同一套消融框架可离线复现**（哈希假嵌入，4 篇文档 / 18 条评测，验证框架本身能否拆开各模块贡献，`python demos/demo_ingest.py`）：

| 配置 | Hit@5 | MRR@5 | NDCG@5 |
|---|---|---|---|
| A 纯向量 | 0.8750 | 0.6219 | 0.6858 |
| B + BM25 / RRF | 1.0000 | 0.8385 | 0.8789 |
| C + Reranker | 1.0000 | 0.9271 | 0.9457 |

**设计决策精选**（详见 [`rag1_knowledge_qa/README.md`](rag1_knowledge_qa/README.md) 与 [`docs/technical-notes-v1.md`](docs/technical-notes-v1.md)）：RRF 为何优于加权求和（量纲不可比）；父子块为何用嵌套切分而非位置反查；中文 BM25 的单字噪声与 `min_score`；拒答判定为何必须匹配固定话术；引用为何只在确定作答后推送。

## 项目二 · 企业知识库 Agent 平台（`rag2_kb_agent`）

面向**多步任务跨系统、长链路易失控**的场景：每一次工具调用都受权限约束，每一轮都可恢复，每一次运行都可回放。

| 机制 | 实现 | 实测证据 |
|---|---|---|
| 受控状态机 | 意图 → 规划 → act/verify 条件边循环 | 3 轮完成"检索制度 → 查工单 → 汇总结论" |
| Checkpoint 续跑 | 每轮落盘，`resume()` 跳过已完成节点 | 限 1 轮熔断后续跑，`intent_fn` 只被调用 1 次 |
| 工具白名单 | `task_type → allowed_tools`，提示词层 + 引擎层双重收敛 | 越权调用被硬拦截，邮件从未发出，且不计入工具预算 |
| 三道熔断 | 最大轮次 / 最大工具次数 / 总耗时 + 死循环检测 | 三个闸门分别触发 |
| 三层记忆 + 压缩 | 短期 LRU / 长期情景召回 / 用户画像；工具结果无条件截断 + 三级阈值压缩 | 轮数 1→50，prompt 稳定在 ~750 token；重载压缩节省 72.3% |
| 引用溯源 | 解析回答里的 `[1][2]` 反查真实 chunk_id | 越界宁可不给，也不给错引用 |
| 决策解析容错 | function calling 的字符串参数 / 非法 JSON / 多调用 / 幻觉工具 / 空回复全覆盖 | 五类异常形态全部有单测 |

评测与项目一共用一套框架（把 `retrieve_fn` 换成 Agent 的 `run()`），记录六类指标：任务成功率 / 平均轮次 / 工具调用 / 工具失败 / 输入 token / P95 延迟。

### 推理轨迹：SSE 实时输出 + 可视化面板

```bash
# 起 Agent 服务（8100）+ 网页客户端（8080，同源代理两个后端）
uvicorn rag2_kb_agent.agent_api:app_factory --factory --port 8100
python tools/serve_ui.py            # 打开 http://127.0.0.1:8080 → 切「Agent 轨迹」页签
```

引擎在**每个 LangGraph 节点执行完**就把事件推给调用方（`run_stream()`），不是跑完再回放：

```
start → intent → plan → tool_call → tool_result → verify → answer → done → summary
```

面板能做的事，正是"定位错误节点"那句话的落点：

- **时间线**按节点级事件渲染，每条带相对耗时（`+12ms`）
- `tool_error` / `limit` / `error` **标红**，可勾「只看错误」直接筛出失败节点
- 点任意一条展开**原始 payload**（工具入参、拦截原因、熔断原因）
- 顶部汇总：轮次 / 工具调用 / 事件数 / 错误节点数 / stop_reason / thread_id
- 轨迹落盘 `data/checkpoints/<tid>.events.jsonl`，可从下拉框**回放历史 thread**

### 耗时归因：span 树 + 甘特条

事件流告诉你"发生了什么"，但回答不了"慢在哪"。所以每一次运行还会长出一棵 **span 树**
（`core/observability.py`）：根是整次 run，孩子是各图节点，节点的孩子是它调的工具：

```
本次 175.2ms · 9 个 span
run                |############################################|    175.2ms (100.0%)
  node:intent        |#                                           |      0.0ms (  0.0%)
  node:tools         |    #################################       |    134.7ms ( 76.9%)
    tool:kb_search     |    #################################       |    133.5ms ( 76.2%)
  node:verify        |                                        #   |      0.0ms (  0.0%)
错误节点：run > node:tools > tool:send_mail（blocked）
```

面板（轨迹页签顶部）按它渲染甘特条，并给出三件可据此定位的事：

- **瓶颈**：按**自身耗时**（总耗时扣掉直接子节点）排序，否则父节点恒为第一、归因等于没做
- **错误节点完整路径**：`run > node:tools > tool:send_mail`，`blocked` / `error` / `limit` 都算
- **占比**：每条 span 占总时长的百分比

span 走**独立通道**（不进事件流）—— 事件是扁平时间线、span 是嵌套树，混在一起两头都不像；
落盘 `data/checkpoints/<tid>.spans.jsonl`（JSONL 追加写，进程被杀也不丢已跑完的部分）。
接口：`GET /agent/spans/{tid}`（只取树）/ `GET /agent/trace/{tid}`（事件 + span）。

13 类事件定义在 `core/agent.py` 的 `EventType`，span 结构在 `core/observability.py`；
离线自证跑 `python demos/demo_agent_trace.py`（23 项自检，不需要 GPU / API Key / langgraph）。

> 口径说明：本项目的 SSE 是**节点级实时**；项目一的 `/chat/stream` 是**分阶段**（检索命中 → 引用 → 答案整段），不是 token 级流式 —— 底层 LLM 客户端未接流式生成。

---

## 接真模型（可选，需要 GPU + DeepSeek Key）

```bash
conda create -n agent_rag --clone fieldai -y
conda run -n agent_rag pip install "transformers==4.44.2" --force-reinstall --no-deps
conda run -n agent_rag pip install -e ".[ml,serve]"
export HF_ENDPOINT=https://hf-mirror.com        # bge-m3 与 bge-reranker-v2-m3 各约 2.2GB → models/

# 建库 → 问答 → 一条命令出可复现评测报告
conda run -n agent_rag python -m rag1_knowledge_qa.cli ingest --docs data/raw
conda run -n agent_rag python -m rag1_knowledge_qa.eval --eval-set data/eval/eval.jsonl --ablation --judge --k-list 5 3 10 --out reports
```

> `transformers` 必须锁 4.44.x：sentence-transformers / FlagEmbedding 尚不兼容 5.x（conda 克隆环境常自带 5.19.0，不降级会在加载模型时直接报错）。

真机冒烟（会调用真实 API、产生费用）：`python demos/live/smoke_real_agent.py`。

## 仓库结构

```
core/                21 个共享模块：检索（BM25/RRF/切分）· 评测（指标/消融/Judge）· Agent（状态机/白名单/熔断）· 基建（LLM 客户端/记忆/SSE/服务层）
rag1_knowledge_qa/   项目一装配层：CLI / FastAPI / uvicorn 入口 / 评测命令 / 生产装配
rag2_kb_agent/       项目二装配层：工具与引擎装配 / 决策解析 / 提示词组装
demos/               9 个零依赖 demo + demos/live/ 2 个真机 demo 与冒烟脚本
tests/               767 项零依赖单测（含工程配置一致性守护：Makefile 路径、CI 边界、行尾、密钥防护）
tools/               语料与评测集生成器（固定 seed，可复现）
data/                83 篇语料 + 119 条评测集（gold 标注）+ 生成脚本的输入
reports/             历次评测运行（含运行指纹、评测集 sha256、消融表、Badcase）
docs/                完整技术笔记（历史版 README）
```

## 为什么这些逻辑能离线验证

向量库、LLM 调用、Embedding、Reranker、Checkpointer 全部是**可注入接口**：用内存实现 + 假 transport 就能覆盖重试、退避、限流、缓存、拒答、熔断、续跑——环境问题（chromadb 装不上、API 限流、GPU 没就绪）不阻塞业务逻辑验证，真模型接上后代码一行不改。这条边界由 CI 守护：测试环境**故意不装** torch / chromadb / langgraph / fastapi。

Agent 引擎提供**双实现**：默认的自实现状态机（零依赖、每一步可见、可离线单测）+ LangGraph 适配层（`core/agent_langgraph.py`，接口完全一致，一行切换）。

## 已知局限（如实声明）

- 意图识别为规则匹配（对权限决策而言是刻意的保守选择，未接语义分类）
- 单线程顺序执行，不支持并行工具调用；未做多租户隔离与索引增量更新
- 记忆层为内存 + 磁盘 JSON，未接 Redis（存储接口已抽象，替换成本低）
- Dockerfile / docker-compose.yml 已编写并通过配置一致性测试，但未在本机构建过镜像
- LangGraph 适配层已通过语义替身测试，真库 API 细节需在装有 langgraph 的环境确认
- 评测集 gold 标注由生成脚本自动定位，大规模人工复核未完成

## License

MIT，见 [LICENSE](LICENSE)。
