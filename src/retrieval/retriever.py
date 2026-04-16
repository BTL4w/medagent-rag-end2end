from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.ingestion.embedding import TextEmbedding
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


def _match_to_dict(match: Any) -> Dict[str, Any]:
    if isinstance(match, dict):
        data = match
    else:
        data = {
            "id": getattr(match, "id", None),
            "score": getattr(match, "score", None),
            "metadata": getattr(match, "metadata", None),
        }
    metadata = data.get("metadata") or {}
    return {
        "id": data.get("id"),
        "score": data.get("score"),
        "doc_id": metadata.get("doc_id"),
        "chunk_index": metadata.get("chunk_index"),
        "section": metadata.get("section"),
        "subsection": metadata.get("subsection"),
        "text": metadata.get("text") or metadata.get("enriched_text") or "",
        "metadata": metadata,
    }


def retrieve(
    query: str,
    top_k: int = 5,
    *,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    fetch_k: Optional[int] = None,
    use_reranker: bool = True,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Dense retrieval from Pinecone, optional reranking.

    - Embeds query with `BAAI/bge-m3`
    - Searches Pinecone namespace
    - Reranks with `BAAI/bge-reranker-v2-m3` (max_length=1024) by default
    """
    if not query or not query.strip():
        return []

    load_dotenv()
    resolved_index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "").strip()
    resolved_namespace = namespace or os.getenv("PINECONE_NAMESPACE", "default").strip()
    if not resolved_index_name:
        raise ValueError("Missing index_name. Pass argument or set PINECONE_INDEX_NAME in environment.")

    dense_k = fetch_k or max(top_k * 3, top_k)
    embedder = TextEmbedding(model_name="BAAI/bge-m3")
    query_vec = embedder.embed([query])
    if query_vec.shape[0] == 0:
        return []

    store = VectorStore(index_name=resolved_index_name, namespace=resolved_namespace)
    result = store.query(
        vector=query_vec[0].tolist(),
        top_k=dense_k,
        include_values=False,
        include_metadata=True,
        filter=filter,
    )
    matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
    candidates = [_match_to_dict(m) for m in matches]
    if not candidates:
        return []

    if use_reranker:
        reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3", max_length=1024)
        return reranker.rerank(query=query, candidates=candidates, top_k=top_k)

    return candidates[:top_k]
