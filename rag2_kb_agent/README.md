# 基于 LangGraph 的企业知识库 Agent 平台

> 面向**知识分散、跨系统人工查找、多步任务无法闭环**的场景，构建受控、可恢复、可评测的 Agent 平台。
> 支持 MCP 工具体系与权限收敛、Checkpoint 断点续跑、三道熔断、三层记忆与上下文压缩、SSE 轨迹可观测。

**核心机制**（全部有单测覆盖，见 `tests/test_agent.py` / `tests/test_decisions.py`）：

| 机制 | 实现 | 实测证据 |
|---|---|---|
| **受控状态机** | 意图识别 → 规划 → act/verify 条件边循环 | 3 轮完成"检索制度 → 查工单 → 汇总结论" |
| **Checkpoint 续跑** | 每轮落盘；`resume()` 跳过已完成节点 | 限 1 轮熔断后续跑，`rounds` 从 1 累加到 2，**`intent_fn` 只被调用 1 次** |
| **工具白名单** | `task_type → allowed_tools`，越权抛 `ToolNotAllowed` | 意图 `spec_inquiry` 下模型发邮件被拦，**邮件从未发出**，且不计入工具预算 |
| **三道熔断** | 最大轮次 / 最大工具次数 / 总耗时 + 死循环检测 | 三个闸门分别触发 |
| **上下文预算** | 工具结果无条件截断 + 只留最近 N 轮明细 | 轮数 1→50 时 prompt 稳定在 ~750 token |
| **引用溯源** | 解析回答里的 `[1][2]` 反查真实 chunk_id | 回答 `... [1] ... [2]` → citations `['c0', 'c2']` |

---

## 1. 解决的四个真实问题

| # | 问题 | 方案 |
|---|---|---|
| 1 | 多步任务涉及**长链路工具调用，易超时、易循环、失败后无法续跑** | 显式状态机 + **Checkpoint** + 三道熔断 + 死循环检测 |
| 2 | 多轮对话导致**上下文持续膨胀，Token 成本与延迟同步上升** | **三层记忆** + 三级阈值压缩 + 工具结果无条件截断 |
| 3 | 模型**选错工具或越权调用**，检索低置信时仍强行作答 | **工具白名单**（提示词层 + 引擎层双重收敛）+ 低置信拒答 |
| 4 | 上线后**缺少量化评估手段，优化全凭感觉** | 事件流 + 轨迹面板数据源 + Badcase 回流 + 消融实验 |

---

## 2. 系统架构

```
                        ┌──────────────────────────────────────┐
   用户问题 ──────────►  │ 意图识别（规则，决定权限；不消耗模型）  │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  route_table: task_type → 工具白名单   │
                        └──────────────────┬───────────────────┘
                                           ▼
        ┌──────────────────────────────────────────────────────────┐
        │                  受控状态机（core/agent.py）               │
        │                                                          │
        │   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────────┐    │
        │   │ plan   │──►│  act   │──►│ verify │──►│ finalize │    │
        │   └────────┘   └───┬────┘   └───┬────┘   └──────────┘    │
        │                    │            │ 不通过 & 轮次未满        │
        │                    │            └──────────┐             │
        │                    │◄──────────────────────┘             │
        │        ┌───────────┴────────────┐                        │
        │        ▼                        ▼                        │
        │  ┌───────────────┐      ┌──────────────────┐             │
        │  │ 工具白名单校验  │      │ 决策解析（decisions）│             │
        │  │ ToolNotAllowed│      │ function calling  │             │
        │  └───────┬───────┘      └──────────────────┘             │
        │          ▼                                               │
        │  ┌────────────────────────────────────────┐              │
        │  │ MCP 工具层：kb_search / ticket_query /   │              │
        │  │            metric_query（+ 敏感工具审批） │              │
        │  └────────────────────────────────────────┘              │
        │                                                          │
        │   熔断：max_rounds · max_tool_calls · total_timeout       │
        │   Checkpoint：每轮落盘，中断后 resume 不重跑已完成节点      │
        │   记忆：短期 LRU / 长期情景召回 / 用户画像 + 三级压缩        │
        └──────────────────────────┬───────────────────────────────┘
                                   ▼
                        ┌──────────────────────────┐
                        │ SSE 事件流（轨迹可观测）    │
                        │ start→intent→plan→tool_call│
                        │ →tool_result→verify→answer │
                        │ →done                     │
                        └──────────────────────────┘
```

---

## 3. 代码结构

```
rag2_kb_agent/
├── app.py         装配：工具 + 引擎 + 决策器 + 记忆 合流（路由表 / 意图识别 / 计划）
├── decisions.py   ★ 真模型决策解析：function calling → act_fn 决策；引用反查
├── prompts.py     ★ system_prompt 按白名单 + 记忆组装；工具结果按预算截断
└── README.md      本文件

复用 core/（21 个模块，全库 765 项零依赖单测）：
    mcp.py      工具注册中心 + Schema 自动生成 + 白名单过滤
    agent.py    状态机 + Checkpoint + 三道熔断 + 事件流
    memory.py   三层记忆 + 三级上下文压缩
    llm.py      LLM 客户端（重试 / 限流 / 磁盘缓存）
    sse.py      SSE 协议
    service.py  框架无关服务层（AgentService：执行 / 轨迹 / 工具清单）
```

---

## 4. 快速开始

### 4.1 零依赖试跑

```bash
cd _starter
python demo_agent2.py     # 六个场景：多步任务 / 幻觉工具 / 越权 / 兜底 / 预算 / 续跑
```

### 4.2 接入真 DeepSeek

```python
import os
from core.llm import LLMClient, DiskCache
from rag2_kb_agent.app import build_tool_calling_engine, build_registry_from_index

llm = LLMClient(api_key=os.environ["DEEPSEEK_API_KEY"],
                model="deepseek-chat",
                cache=DiskCache("data/cache/llm"))     # 缓存能显著降低评测成本
registry = build_registry_from_index(index)             # 把检索索引封装成 MCP 工具
engine = build_tool_calling_engine(llm, registry, checkpointer=...)

run = engine.run("工单 T-100 的关闭流程是否合规？")
print(run.state.answer, run.state.citations)
```

`ToolCallingDecider` **用 `inspect` 自动探测**客户端是否支持 `tools` 参数：
支持则走原生 function calling，不支持则回退纯文本解析。无需你传开关。

---

## 5. 关键设计决策（面试可讲）

### 5.1 为什么自实现状态机而不是直接用 langgraph

| 理由 | 说明 |
|---|---|
| **可离线验证** | Checkpoint / 白名单 / 熔断是**工程逻辑**，其正确性与 langgraph 装不装得上无关，必须先能单测（全库 765 项单测，约 1.2 秒跑完） |
| **迁移成本极低** | 状态字段、节点划分、条件边语义都对齐 langgraph；真机把 `AgentEngine.run` 换成 `StateGraph` 编译即可 |
| **面试可讲** | "我自实现过一遍，所以知道每一步在解决什么" 远强于 "我调了库" |

### 5.2 权限决策不交给模型

`intent_fn` 用**规则**而不是 LLM：① 零成本零延迟；② **权限决策不能有不确定性**；
③ 未识别时退到 `general`（空白名单）—— 最保守的默认。
LLM 可以用来做语义分类，但"能给模型哪些工具"这个决定必须由代码做。

### 5.3 工具白名单在两层同时生效

1. **提示词层**：`format_tool_list` 只列出白名单内的工具（降低幻觉调用概率）
2. **引擎层**：`ToolRegistry.call(allowed=...)` 硬拦截（幻觉调用也拦得住）

只有提示词约束是不够的——模型仍可能凭记忆调一个见过的工具名。
只有引擎拦截也不够——每轮都让模型调错再纠正，浪费轮次和 Token。

### 5.4 幻觉工具调用不能静默丢弃

模型调了未注册的 `web_search` 时，初版把它当"没有决策"处理，结果整轮空转。
现在它会：① 走到引擎产生一条**可观测的 `tool_error` 事件**；
② 下一轮 prompt 里带上"该工具被拒绝"，让模型改口。

### 5.5 工具结果必须**无条件**截断，不能等超窗

一次检索返回 20 条 chunk、每条上千字符，第 4 轮就爆窗。
`format_scratchpad` 的策略是：
- 单条结果先截断到 `max_tool_result_chars`（默认 600）
- 只保留最近 `recent_rounds`（默认 3）轮明细，更早的压成一行计数摘要

实测：轮数 1→50，prompt 规模稳定在 ~750 token。**这是结构性保证，不是事后补救。**

### 5.6 引用溯源必须反查 chunk_id

模型在回答里写 `[1] [2]`，但它**不会告诉你 1 和 2 是哪个 chunk**。
如果不解析，`citations` 就是空的——前端渲染不出可点击来源，"引用溯源"这个卖点等于没实现。

`resolve_citations` 按"工具调用顺序拼接后的第 n 个 hit"映射下标，越界则丢弃
（**宁可不给引用，也不给错引用**）。

### 5.7 `except TypeError` 兜底是个陷阱

初版为兼容不支持 `tools` 参数的客户端写成：

```python
try:
    response = self.llm.chat(messages, tools=tools)
except TypeError:                    # ← 危险！
    response = self.llm.chat(messages)
```

问题：**函数体内部**抛出的任何 `TypeError` 都会被误判成"不支持 tools"，
然后带着错误的假设重试一次，把真实 bug 掩盖掉。

正确做法：初始化时用 `inspect.signature` 探测一次并缓存。

---

## 6. 评测（Agent 层）

与项目一共用 `core/evaluation.py`：把 `retrieve_fn` 换成 Agent 的 `run()` 即可复用同一套消融框架。

| 档位 | 配置 |
|---|---|
| Baseline | 单轮直答（不调工具） |
| +混合检索 | 加检索工具，不做规划 |
| +记忆 | 加三层记忆 |
| +上下文压缩 | 加三级压缩 |
| Full Harness | 全开 |

记录 **6 类指标**：任务成功率 / 平均模型轮次 / 工具调用次数 / 工具失败次数 / 输入 Token 总量 / P95 延迟。

---

## 7. 局限与后续

- 意图识别目前是规则匹配，**未接语义分类**（但对权限决策而言这是有意的保守选择）
- 未实现真正的并发 Sub-agent（`core/agent.py` 是单线程顺序执行）
- 记忆层用内存 + 磁盘 JSON，**未接 Redis**（接口已抽象，替换成本低）
- 未做多租户隔离
