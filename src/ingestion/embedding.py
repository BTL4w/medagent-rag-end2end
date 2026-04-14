from __future__ import annotations

from typing import Iterable, List

import numpy as np
try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    torch = None
    SentenceTransformer = None


class TextEmbedding:
    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        if SentenceTransformer is None or torch is None:
            raise ImportError("sentence-transformers and torch are required for TextEmbedding")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_batch(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
