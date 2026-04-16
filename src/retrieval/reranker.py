from __future__ import annotations

from typing import Dict, List

import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker based on bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 1024,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=self.device,
            trust_remote_code=True,
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        Rerank candidate docs using cross-encoder score.

        Each candidate should include at least:
          - "text": text used for scoring
        """
        if not query or not candidates:
            return []

        pairs = []
        for item in candidates:
            text = str(item.get("text", "")).strip()
            pairs.append([query, text])

        scores = self.model.predict(pairs)
        enriched = []
        for item, score in zip(candidates, scores):
            row = dict(item)
            row["rerank_score"] = float(score)
            enriched.append(row)

        enriched.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
        return enriched[:top_k]
