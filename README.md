# Day 1~7 起步包 · 两个项目的已验证代码

> 配套 `../7天Agent项目冲刺计划.md`。
> **689 项单测全绿**，全部离线可跑（无网络、无 GPU、无 API Key、无第三方依赖）。
> **18 个 core 模块 + 两个项目装配层 + 两份项目 README + 10 个可运行 demo。**

```bash
cd _starter
python -m unittest discover -s tests -v    # 期望 689 tests OK
python demo_core.py                        # 切分与 RRF 融合
python demo_pipeline.py                    # 检索链路（BM25+向量+RRF+拒答+指标）
python demo_eval.py                        # 检索层消融
python demo_agent.py                       # Agent 引擎四机制（脚本化决策）
python demo_ingest.py                      # 全流程：文件 -> 索引 -> Agent -> 评测
python demo_service.py                     # 服务层 + CLI + SSE + dispatch
python demo_agent2.py                      # 真 function calling 决策循环
python demo_evalrun.py                     # ★ 一条命令产出可复现评测报告
python demo_agenteval.py                   # ★ Agent 层四档消融（六类指标）
```

---

## 一、这个包里有什么

### core/（18 个模块，零第三方依赖）

| 文件 | 内容 | 状态 |
|---|---|---|
| `metrics.py` | Hit@K / MRR@K / Recall@K / Precision@K / NDCG@K / 拒答正确率 / 幻觉率 / 聚合 / 延迟分位 | ✅ 已单测 |
| `retrieval.py` | `rrf_fuse`、`weighted_fuse`（对照组）、`recursive_chunk`（递归切分 + 标题层级） | ✅ 已单测 |
| `text.py` | 中文分词（单字 + bigram）、文本归一化（含 CRLF）、度量词提取、Jaccard | ✅ 已单测 |
| `bm25.py` | BM25 自实现（k1/b 可调、IDF 平滑、`min_score` 阈值过滤） | ✅ 已单测 |
| `vectorstore.py` | 向量库抽象 + `InMemoryVectorStore`（含落盘）+ `ChromaVectorStore`（延迟导入） | ✅ 已单测 |
| `llm.py` | LLM 客户端：错误分类 / 指数退避重试 / 令牌桶限流 / 磁盘缓存 / **拒答判定** | ✅ 已单测 |
| `pipeline.py` | 检索编排：各路**逐路可开关** + 4 档消融配置 | ✅ 已单测 |
| `evaluation.py` | 检索层评测框架：体检 / 分类型统计 / Markdown 报告 / Badcase 清单 | ✅ 已单测 |
| `judge.py` | **LLM-as-Judge**：四维打分 Prompt + 强容错解析（JSON/中文键/正则兜底） | ✅ 已单测 |
| `eval_runner.py` | **评测编排**：运行指纹 / 报告目录 / 多 k / 两次 run 对比 | ✅ 已单测 |
| **`agent_eval.py`** | **★ Agent 层评测**：任务成功判定 / **六类指标** / 四档消融配置 / 报告渲染 | ✅ 已单测 |
| `mcp.py` | **MCP 风格工具层**：Schema 自动生成与校验、**白名单过滤**、双格式转换 | ✅ 已单测 |
| `agent.py` | **★ Agent 主循环**：状态机 / **Checkpoint 续跑** / **三道熔断 + 死循环检测** | ✅ 已单测 |
| `memory.py` | **三层记忆 + 三级上下文压缩**（含工具结果无条件截断） | ✅ 已单测 |
| `loaders.py` | 多格式解析（md/txt/json/jsonl/docx/pdf/html），依赖可选、坏文件不中断 | ✅ 已单测 |
| `ingest.py` | **建库流水线**：父子块切分 + 双索引 + 配置指纹 + 逐项参数落 stats | ✅ 已单测 |
| `sse.py` | SSE 协议：帧格式化、事件流转换、`parse_sse` 往返校验、带序号 `SSEStream` | ✅ 已单测 |
| `service.py` | **框架无关服务层**：问答 / 拒答 / 引用 / SSE 流 / Agent 执行 / 轨迹 / dispatch | ✅ 已单测 |

### rag1_knowledge_qa/（项目一：RAG 垂直领域知识问答）

| 文件 | 内容 | 状态 |
|---|---|---|
| `README.md` | **★ 项目 README**（架构图 + 指标表 + 6 个设计决策）——可直接作为 GitHub README | ✅ |
| `eval.py` | **★ 评测命令行**：`--ablation` / `--judge` / `--show` / `--compare` | ✅ 已单测 |
| `cli.py` | 命令行入口（`ingest` / `ask` / `retrieve` / `health`） | ✅ 已单测 |
| `bootstrap.py` | 生产装配；顶部 docstring 是嵌入/精排的参考实现 | 🔶 需你补两个类 |
| `api_fastapi.py` | FastAPI 接线（6 个路由） | 🔶 需 `pip install fastapi` |

### rag2_kb_agent/（项目二：LangGraph 企业知识库 Agent 平台）

| 文件 | 内容 | 状态 |
|---|---|---|
| `README.md` | **★ 项目 README**（架构图 + 机制表 + 7 个设计决策） | ✅ |
| `app.py` | 装配：路由表 / 规则意图识别 / 计划 / 记忆 / 工具封装 / **消融引擎** | ✅ 已单测 |
| `decisions.py` | **★ 真模型决策解析**：function calling → act_fn；幻觉/越权/多调用；**引用反查** | ✅ 已单测 |
| `prompts.py` | **★ messages 组装**：按白名单注入工具、三层记忆、工具结果按预算截断 | ✅ 已单测 |

### tests/（13 个文件，689 项用例）与 reports/

**外部依赖只在四处，且全部延迟导入**：

| 需要依赖 | 用在哪 | 装不上怎么办 |
|---|---|---|
| `chromadb` | `ChromaVectorStore` | `InMemoryVectorStore` 完全替代 |
| `openai` | `LLMClient` 默认 transport | 注入 transport 即可离线跑 |
| `pypdf` / `python-docx` | `loaders` 的 pdf / docx | 抛 `MissingDependency` 并给出安装命令 |
| `fastapi` | `api_fastapi.py` | 用 `cli.py`，或直接用 `core.service` |
| `langgraph` | **本包不依赖** | `core/agent.py` 是自实现的等价状态机 |

---

## 二、为什么这些能离线验证（这是本包的设计核心）

计划里说过 `config.py` / `embedder.py` 这类文件我不能替你写，因为要联网和 GPU。但这六个模块不同——**它们把"外部世界"抽成了可注入的接缝**：

| 模块 | 接缝 | 效果 |
|---|---|---|
| `vectorstore` | `BaseVectorStore` 抽象 | 用内存实现验证检索逻辑，不必等 chromadb 装好 |
| `llm` | `transport` 参数 | 注入 fake 就能测**重试、退避、限流、缓存**，完全不需要 API Key |
| `bm25` | 自实现 | 零依赖，可测 IDF、阈值过滤、排序稳定性 |
| `text` | 自实现 | 零依赖，可测中文切分策略 |
| `metrics` | 纯函数 | 可直接断言数学正确性 |
| `pipeline` | `embed_fn` / `rerank_fn` 可注入 | 用假向量验证**编排与消融逻辑**，真模型接上后代码一行不改 |
| `evaluation` | `retrieve_fn` / `judge_fn` 可注入 | 评测框架本身可测，不必等真模型 |

**这套"接缝式设计"本身就是面试素材。** 被问"你怎么保证检索逻辑是对的"，你可以答：

> 我把向量库、LLM 调用、Embedding、Reranker 都做成了可替换接口。检索与评测逻辑用内存实现 + 假 transport 覆盖了 193 个用例，跑一遍 0.09 秒。这样环境问题（chromadb 装不上、API 限流、GPU 没就绪）不会阻塞我验证业务逻辑——真模型接上后代码一行不改。

---

## 三、评测框架输出长什么样（`reports/demo_ingest_ablation.md`）

这是 Day 3 的最终产物，也是面试时你打开给面试官看的东西。**真实文件链路**（4 篇文档 → 9 个子块）跑出的消融表：

```markdown
| 配置 | hit@5 | mrr@5 | recall@5 | precision@5 | ndcg@5 | n | P95 延迟(ms) |
|---|---|---|---|---|---|---|---|
| A_vector_only (向量)               | 0.8750 | 0.6219 | 1.0000 | 0.2000 | 0.6858 | 16 | … |
| B_hybrid_rrf (向量+BM25)           | 1.0000 | 0.8385 | 1.0000 | 0.2000 | 0.8789 | 16 | … |
| C_hybrid_rerank (向量+BM25+Rerank) | 1.0000 | 0.9271 | 1.0000 | 0.2000 | 0.9457 | 16 | … |
| D_hybrid_rerank_rules (…+规则)     | 1.0000 | 0.9271 | 1.0000 | 0.2000 | 0.9457 | 16 | … |
```

这份表是在**文件系统里的真实文档**上跑出来的（解析 → 父子块切分 → 双索引 → 检索 → 评测），
只有嵌入是假实现。**Hit@5 0.875 → 1.0、MRR 0.62 → 0.93 的递进是真的分解出了各模块贡献。**

> ⚠️ D 档与 C 档持平，因为评测 query 里没有多位数字，规则召回没被触发。
> 这本身是个正确的观察——面试时可以说"规则召回只在含编号/型号的查询上生效，
> 我的评测集这类样本不足，所以这一档没有增益"。

报告还包含：**分类型汇总**（fact / term / out_of_scope）、**Badcase 回流清单**、**延迟分位**。

---

## 三、测试过程中发现并修掉的 6 个真实 bug

这些不是"写错了改一改"，每个都对应一个**真实的领域问题**，全部可以直接当面试回答：

### 1. `第X条` 被当标题剥离，导致正文整句消失 ⭐ 最有价值

中文规范文档里 `第X章` / `第X节` 是真标题，但 **`第X条` 后面直接跟条文正文**（"第一条 本规范适用于…"）。最初把 `第X条` 也当标题剥掉，**整句正文消失**，是单测报 `chunk 数量 = 0` 才定位到。

修正：按标记类型区分——章/节进 `heading_path`，`第X条` 保留在正文里。

**面试话术**：
> 我按标题层级维护一个栈。章、节作为真标题进层级路径，但 `第X条` 我特意不剥离——它后面直接跟条文正文，剥掉会导致正文丢失。这个坑是靠单测发现的：我先写了期望的 chunk 数，跑出来是 0。

### 2. `min_score` 阈值：中文 BM25 的固有噪声 ⭐

中文走单字 + bigram 切分时，**任何一段中文查询几乎总能撞上几个高频单字**（"的"、"不"、"是"），这些字 IDF 很低但非零，会让无关文档拿到正分。实测：

```
query='完全不相关的查询内容'
   c3  score= 1.094   ← 噪声，靠单字重合
   c0  score= 0.677
query='ONNX 量化 时延'
   c3  score= 8.250   ← 相关，量级完全不同
```

所以 `BM25Index.search` 加了 `min_score` 参数，等价于一次"低置信拒答"。

**面试话术**：
> BM25 在中文场景有个容易被忽略的问题：单字切分会让无关查询也拿到正分。我在检索层加了分数下限过滤，阈值是在评测集上按「拒答正确率 vs 漏答率」权衡扫出来的，不是拍脑袋。

### 3. 单数字污染词表

`extract_measures` 最初用 `\d+(?:\.\d+)?%?` 匹配数值，结果把 "3 个要点" 的 `3`、"7:1:2 划分" 的 `7` 和 `1` 全收进词表，稀释 IDF。

修正后只收**小数（0.946）、百分比（75%）、多位整数（30600 / 2022）**。

### 4. 磁盘缓存"看起来启用但永远不命中"

`DiskCache.set` 用 `try/except OSError: pass` 吞掉了写失败，导致缓存静默失效——非常难排查。

修正：加 `_ensure_dir()` 每次写前检查目录，`set` 返回 bool，异常不再静默。

### 5. 重试策略必须区分可恢复错误

初版对所有异常都重试。正确做法（`classify_error`）：

| 状态码 | 处理 |
|---|---|
| 429 / 5xx / 超时 / 连接错误 | **可重试**，指数退避 + 抖动 |
| 400 / 401 / 403 / 404 | **不可重试**，立即抛出 |

重试 401（密钥错）只会浪费时间并可能触发风控。

### 6. 低置信拒答的阈值不能跨尺度复用

RRF 融合分数在 **0.03** 量级，BM25 分数在 **1~10** 量级——**两者不能共用同一个阈值**。实践做法是把阈值绑定在最终用于排序的那一路分数上。

---

## 四、Agent 四大工程机制（`demo_agent_output.txt`）

Day 4/5 的四条简历要点，全部在 `core/agent.py` 落地并有单测覆盖：

| 机制 | 实现 | 实测证据 |
|---|---|---|
| **受控状态机** | 意图识别 → 规划 → act/verify 条件边循环 | 3 轮完成多步任务，事件流可完整回放 |
| **Checkpoint 续跑** | 每轮落盘；`resume()` 跳过已完成节点 | 第一段限 1 轮被熔断 → 续跑从 1 继续累加到 5，**`intent_fn` 只被调用 1 次**（没有重新识别意图） |
| **工具白名单** | `task_type → allowed_tools`，越权调用抛 `ToolNotAllowed` | 意图是 `policy_inquiry`（只许 kb_search），模型发邮件被拦截，**`MAIL_LOG` 保持为空**，且被拦调用不计预算 |
| **三道熔断** | 最大轮次 / 最大工具次数 / 总耗时 + 死循环检测 | 三个闸门分别触发；重复工具序列被 `detect_loop` 识别 |

**三级上下文压缩实测**（`token_budget=400`）：

| 负载 | 压缩档位 | Token | 节省 |
|---|---|---|---|
| 轻载（3 条） | `none` | 104 → 104 | 0.0% |
| 中等（9 条） | `full_compress` | 528 → 369 | 30.1% |
| 重载（21 条） | `full_compress` | 1696 → 469 | **72.3%** |

> ★ 注意：**工具结果截断是无条件的第一步**（不受阈值影响）。一次召回 20 条 chunk 的
> 长 JSON 常占上下文大头，而模型真正需要的只是前几条——这一步零成本、性价比最高。

### 为什么"自实现状态机"而不是直接用 langgraph

| 理由 | 说明 |
|---|---|
| **可离线验证** | Checkpoint / 白名单 / 熔断这些**工程逻辑的正确性**与 langgraph 装不装得上无关，必须先能单测 |
| **迁移成本极低** | 状态字段、节点划分、条件边语义都对齐 langgraph；真机把 `AgentEngine.run` 换成 `StateGraph` 编译即可 |
| **面试可讲** | "我自实现过一遍，所以知道每一步在解决什么问题" 远强于 "我调了库" |

### 真机接 DeepSeek 时只改一个函数

```python
# act_fn 现在是脚本化决策队列；真机换成"让 DeepSeek 选工具"
def act_fn(state, allowed_tools):
    messages = build_messages(state)                 # 含记忆 + 检索上下文
    tools = registry.openai_tools(allowed=allowed_tools)   # ★ 白名单已生效
    decision = llm.chat_json(messages)               # 输出 {"tool":..,"args":..} 或 {"final":..}
    return decision

# 其余（白名单、checkpoint、熔断、事件流、记忆、压缩）一行都不用改
```

---

## 五、RRF 为什么优于加权求和（面试高频）

```
BM25 分数范围：[0, +∞)   例如 8.25、6.41、1.09
向量相似度范围：[-1, 1]   例如 0.57、0.34、0.30
```

两者**量纲不可比**。加权求和要么先归一化（对分布敏感、需反复调权重），要么直接相加（BM25 会完全压制向量分数）。

RRF 只用**排名**：`score = Σ 1/(k + rank)`，天然免疫量纲问题且几乎免调参。`demo_core.py` 的输出演示了它的机制——两路都命中的文档得分最高。

---

## 六、Day 1 你还需要自己写的

| 文件 | 要点 | 为什么不能替你写 |
|---|---|---|
| `core/config.py` | pydantic-settings 读 `.env` | 依赖你的路径与 Key |
| `core/embedder.py` | BGE-M3 封装（GPU、batch、L2 归一化） | 需 GPU 验证 |
| `core/reranker.py` | bge-reranker-v2-m3 封装 | 需 GPU 验证 |
| `core/loaders.py` | pdf / md / txt / docx 解析 | 可用 AI 生成后自审 |
| `core/logging.py` | loguru + 耗时埋点 | 简单，AI 可写 |
| `core/cache.py` | **直接复用 `core/llm.py` 的 `DiskCache`** | ✅ 已给 |

接进 `pipeline.py` 只需两个函数，签名已经约定好：

```python
from core.pipeline import HybridRetriever, ablation_configs
from core.embedder import BgeEmbedder      # 你自己写
from core.reranker import BgeReranker      # 你自己写

embedder = BgeEmbedder("models/bge-m3")                    # 输出 L2 归一化向量
reranker = BgeReranker("models/bge-reranker-v2-m3")        # (query, pairs) -> [(id, score)]

retriever = HybridRetriever(
    bm25=bm25, vector_store=store,
    embed_fn=embedder.encode_one,          # 必须 L2 归一化（内积 == 余弦）
    rerank_fn=reranker.rerank,
    doc_lookup=lambda d: DOCS.get(d),
)
for cfg in ablation_configs(top_k=5):
    rep = run_retrieval_eval(items, lambda q, c=cfg: retriever.retrieve(q, c),
                             config_name=cfg.name)
```

---

## 七、接进你的项目

```bash
cp -r _starter/core   <你的项目>/core
cp -r _starter/tests  <你的项目>/tests
cd <你的项目> && python -m unittest discover -s tests   # 193 tests OK
```

**建议 Day 1 第一件事就是跑这 193 个测试**——它立刻告诉你 Python 环境是否可用，比写完 200 行代码才发现环境有问题省事得多。

---

## 八、全流程端到端演示（`demo_ingest_output.txt`）

`demo_ingest.py` 是把所有模块串起来的"总装配"，也是 Day 6 要打包进仓库的主脚本：

```
① loaders      4 篇 md/txt/json 文档 -> 683 字符，0 失败
② ingest       7 个父块 + 9 个子块，双索引（向量 9 条 / BM25 9 条），配置指纹 95adfe7168b0e7d5
③ pipeline     混合检索：vector 召回 9 条 + bm25 召回 8 条 -> RRF -> Rerank -> top1 命中"200 毫秒"
④ agent        2 次工具调用、3 轮完成多步任务；白名单 ['kb_search','checklist_query']
⑤ sse          10 条事件帧 / 1657 字节；解析回来类型序列完整
⑥ evaluation   18 条评测集（16 可答 + 2 越界），四档消融，生成 Markdown 报告
```

**父子块在真实数据上的效果**（第 2 节输出）：

```
子块 01_检测技术要求.md#p0#c0
  检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。…
父块 01_检测技术要求.md#p0
  检测精度要求 mAP@0.5 不低于 0.90，F1 不低于 0.88。… 小目标（面…   ← 更完整
```

检索命中子块（精准），喂给模型的是父块（上下文完整）——这就是"父子块双层索引"的价值。

## 九、进度对照

| Day | 计划要求 | 本包已覆盖 |
|---|---|---|
| Day 1 | 环境 + `core/` 基建 | ✅ metrics / retrieval / text / bm25 / vectorstore / llm / **loaders** 七件套；剩 embedder / reranker / config / logging 需你补（要 GPU 与真模型） |
| Day 2 | RAG 检索链路 | ✅ `pipeline.py` + **`ingest.py`**（父子块切分、双索引、配置指纹） |
| Day 3 | ★ 评测体系 | ✅ `evaluation.py`（体检 / 消融 / 分类型 / 报告 / Badcase 清单）；剩评测集人工复核与真模型替换 |
| Day 4 | LangGraph Agent | ✅ `mcp.py` + `agent.py`（状态机 / Checkpoint 续跑 / 白名单 / 熔断 / 事件流） |
| Day 5 | 记忆 + 压缩 | ✅ `memory.py`（三层记忆 + 三级压缩） |
| Day 6 | 打包 + 部署 + 简历回填 | 🔶 **`sse.py` + `demo_ingest.py` 已就绪**，可直接作为仓库主脚本；剩 FastAPI 路由接线、README 架构图、真机跑数 |
| Day 7 | 演练 + 投递 | ⬜ |

> ⚠️ 红线再强调：**Day 3 / Day 5 跑出来是多少，简历就写多少。**
> 这套指标代码是对的，所以它算出来的结果就是你的真实水平，也是面试时唯一站得住的东西。
