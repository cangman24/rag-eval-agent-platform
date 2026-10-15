# -*- coding: utf-8 -*-
"""core/vectorstore.py —— 向量库抽象 + 两个实现。

为什么做抽象层：
  1. **可测试性**：InMemoryVectorStore 零依赖、可单测，把检索逻辑的正确性
     和"ChromaDB 装不装得上"解耦。Day 1 装 chromadb 失败也不会阻塞进度。
  2. **可替换性**：简历上写 Milvus 就只需再写一个实现类，
     业务代码一行不改 —— 这也是面试问"你怎么做技术选型"时的答案。

距离约定：统一用 **余弦相似度**（越大越相似）。
  存储时对向量做 L2 归一化，则内积 == 余弦相似度，检索更快。
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def l2_normalize(vec: Sequence[float]) -> List[float]:
    """L2 归一化。归一化后 dot(a, b) == cosine(a, b)。"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class BaseVectorStore:
    """向量库接口。所有实现都必须满足这个契约。"""

    def add(self, ids: Sequence[str], embeddings: Sequence[Sequence[float]],
            documents: Sequence[str], metadatas: Optional[Sequence[dict]] = None) -> None:
        raise NotImplementedError

    def query(self, embedding: Sequence[float], top_k: int = 20) -> List[Tuple[str, float]]:
        """返回 [(doc_id, similarity)] 降序。"""
        raise NotImplementedError

    def search(self, embedding: Sequence[float], top_k: int = 20) -> List[Tuple[str, float]]:
        """`query` 的别名（与 BM25Index.search 保持命名一致，便于融合层统一调用）。"""
        return self.query(embedding, top_k=top_k)

    def get(self, doc_id: str) -> Optional[dict]:
        raise NotImplementedError

    def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def persist(self) -> None:  # pragma: no cover - 由实现决定
        pass


class InMemoryVectorStore(BaseVectorStore):
    """零依赖内存实现。用于单测与"chromadb 装不上时的退路"。"""

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self._vectors: Dict[str, List[float]] = {}
        self._docs: Dict[str, str] = {}
        self._meta: Dict[str, dict] = {}

    def add(self, ids, embeddings, documents, metadatas=None) -> None:
        if not (len(ids) == len(embeddings) == len(documents)):
            raise ValueError("ids / embeddings / documents 长度必须一致")
        metadatas = metadatas if metadatas is not None else [{} for _ in ids]
        for i, doc_id in enumerate(ids):
            vec = list(embeddings[i])
            self._vectors[doc_id] = l2_normalize(vec) if self.normalize else vec
            self._docs[doc_id] = documents[i]
            self._meta[doc_id] = dict(metadatas[i])

    def query(self, embedding, top_k: int = 20):
        q = l2_normalize(embedding) if self.normalize else list(embedding)
        scored: List[Tuple[str, float]] = []
        for doc_id, vec in self._vectors.items():
            scored.append((doc_id, dot(q, vec)))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    def get(self, doc_id):
        if doc_id not in self._vectors:
            return None
        return {
            "id": doc_id,
            "document": self._docs[doc_id],
            "metadata": self._meta[doc_id],
            "embedding": self._vectors[doc_id],
        }

    def delete(self, ids) -> None:
        for doc_id in ids:
            self._vectors.pop(doc_id, None)
            self._docs.pop(doc_id, None)
            self._meta.pop(doc_id, None)

    def count(self) -> int:
        return len(self._vectors)

    def all_metadata(self) -> Dict[str, dict]:
        return dict(self._meta)

    # ---- 落盘（可选，评测时避免重复建库） ----
    def persist(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "normalize": self.normalize,
            "ids": list(self._vectors.keys()),
            "vectors": [self._vectors[i] for i in self._vectors],
            "docs": [self._docs[i] for i in self._vectors],
            "meta": [self._meta[i] for i in self._vectors],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "InMemoryVectorStore":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        store = cls(normalize=payload.get("normalize", True))
        store.add(payload["ids"], payload["vectors"], payload["docs"], payload["meta"])
        return store


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 实现（真机使用）。

    延迟导入 chromadb：这样即使依赖没装，本模块仍可被 import，
    InMemoryVectorStore 也照常工作。
    """

    def __init__(self, collection_name: str = "kb", persist_dir: str = "data/vectors",
                 metric: str = "cosine") -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "未安装 chromadb。请执行：pip install chromadb==0.5.5\n"
                "若安装失败，可退回 InMemoryVectorStore + persist() 落盘。"
            ) from exc

        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": metric}
        )

    def add(self, ids, embeddings, documents, metadatas=None) -> None:
        self.collection.add(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=[dict(m) for m in (metadatas or [{} for _ in ids])],
        )

    def query(self, embedding, top_k: int = 20):
        res = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        # cosine 空间下 distance = 1 - similarity
        return [(doc_id, 1.0 - float(d)) for doc_id, d in zip(ids, dists)]

    def get(self, doc_id):
        res = self.collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
        ids = res.get("ids") or []
        if not ids:
            return None
        return {
            "id": ids[0],
            "document": (res.get("documents") or [""])[0],
            "metadata": (res.get("metadatas") or [{}])[0],
            "embedding": (res.get("embeddings") or [[]])[0],
        }

    def delete(self, ids) -> None:
        self.collection.delete(ids=list(ids))

    def count(self) -> int:
        return int(self.collection.count())
