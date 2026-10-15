# core/embedder.py
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2归一化，内积等价余弦相似度"""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm[norm == 0] = 1e-8
    return vec / norm

class BgeEmbedder:
    def __init__(self, model_dir: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码，返回L2归一化向量列表"""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.model(**encoded)
        # BGE‑M3 使用 CLS 向量
        emb = out.last_hidden_state[:, 0, :]
        emb_np = emb.cpu().numpy()
        emb_norm = l2_normalize(emb_np)
        return emb_norm.tolist()

    def encode_one(self, text: str) -> List[float]:
        vecs = self.encode_batch([text])
        return vecs[0]
