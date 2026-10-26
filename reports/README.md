# reports/ —— 评测报告

> **`latest/` 是当前唯一的权威报告；`archive/` 是调参历史。**
> 数字一律引用 `latest/`，历史 run 只用来回答"为什么参数是现在这个值"。

每个 run 目录（含 `latest/`）是一次完整评测的产物：

| 文件 | 内容 |
|---|---|
| `eval_manifest.json` | **运行指纹**：run_id、时间、评测集 sha256、切分参数与配置指纹、检索口径、评委信息，以及 `model_info`（模型实测）与 `corpus_fingerprint`（语料指纹） |
| `eval_set.jsonl` | 本次运行使用的评测集快照（与 manifest 里的 sha256 对应） |
| `retrieval_A~D_*.md` | 四档消融的逐档报告（指标表 + 分类型统计 + Badcase 清单） |
| `SUMMARY.md` | 总结报告（人读入口） |

## 复现与校验

```bash
python -m rag1_knowledge_qa.eval --eval-set data/eval/eval.jsonl --ablation --judge \
    --k-list 5 3 10 --docs data/raw --out reports --notes "<备注>"
python tools/verify_evidence_chain.py reports/latest   # 语料/模型/维度三项校验
```

`verify_evidence_chain.py` 校验：索引里的每篇文档是否仍在磁盘上且内容逐字一致、
manifest 是否写明模型名（不允许 `unknown`）、向量维度是否实测且等于 1024。
退出码 0 = 证据链完整，1 = 有缺口，2 = 报告缺失。

> 历史 run 的 `embedding_model` 字段为 `unknown`（评测入口只读环境变量、建库参数没回写），
> 因此**无法自证**跑在什么模型上——这个问题已在 `model_info` 实测回写后修掉，
> 旧报告的问题与影响见 [`archive/README.md`](archive/README.md)。
