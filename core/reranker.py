# core/reranker.py
from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BgeReranker:
    def __init__(self, model_dir: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, pairs: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """对候选做交叉编码精排。

        Args:
            query: 用户问题
            pairs: [(chunk_id, 正文)] —— ★ 第一个元素是 **id**，不是正文。

        Returns:
            [(chunk_id, score)]，按分数**降序**。

        ★ 这里曾有个会静默毁掉精排的 bug：
          初版返回的是 (passage_text, score)，而 `core/pipeline.py` 的契约是
          (doc_id, score)。后果是下游拿到"正文当 id"去查文档，
          `doc_lookup(正文)` 必然返回空串 —— 于是：
            ① 引用溯源显示空片段
            ② 评测时 rerank 档的 doc_id 与 gold_ids 对不上，
               消融表里 +Reranker 那一档的指标会莫名其妙地掉
          而且**不会报错**，只是结果悄悄错掉。
          排序本身是对的（按 score 降序），所以"看起来能用"—— 这才是它危险的地方。
        """
        if not pairs:
            return []
        doc_ids = [doc_id for doc_id, _ in pairs]
        texts_p = [text for _, text in pairs]
        texts_q = [query] * len(pairs)
        enc = self.tokenizer(
            texts_q, texts_p, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        scores = self.model(**enc).logits.squeeze(-1).cpu().numpy()
        # ★ 用 doc_ids 而不是 texts_p 组装返回值
        scored = list(zip(doc_ids, scores.tolist()))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored
