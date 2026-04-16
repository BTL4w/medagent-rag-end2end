from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.ingestion.embedding import TextEmbedding
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import VectorStore


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


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
        "score": float(data.get("score", 0.0) or 0.0),
        "doc_id": metadata.get("doc_id"),
        "chunk_index": metadata.get("chunk_index"),
        "section": metadata.get("section"),
        "subsection": metadata.get("subsection"),
        "text": metadata.get("text") or metadata.get("enriched_text") or "",
        "metadata": metadata,
    }


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        return [1.0 for _ in values]
    return [(v - v_min) / (v_max - v_min) for v in values]


def _bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    query_tokens = _tokenize(query)
    if not query_tokens or not docs:
        return [0.0 for _ in docs]

    tokenized_docs = [_tokenize(doc) for doc in docs]
    doc_lens = [len(toks) for toks in tokenized_docs]
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)

    # Document frequency per term
    df: Dict[str, int] = {}
    for toks in tokenized_docs:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1

    N = len(tokenized_docs)
    query_tf = Counter(query_tokens)
    scores: List[float] = []

    for toks, dl in zip(tokenized_docs, doc_lens):
        tf_doc = Counter(toks)
        score = 0.0
        for term, qf in query_tf.items():
            if term not in tf_doc:
                continue
            term_df = df.get(term, 0)
            idf = math.log(1 + (N - term_df + 0.5) / (term_df + 0.5))
            tf = tf_doc[term]
            denom = tf + k1 * (1 - b + b * (dl / max(avgdl, 1e-9)))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9)) * qf
        scores.append(score)
    return scores


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    *,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    fetch_k: Optional[int] = None,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid search:
    1) Dense retrieve from Pinecone
    2) Lexical BM25 scoring on retrieved candidates
    3) Weighted score fusion
    4) Optional cross-encoder reranking
    """
    if not query or not query.strip():
        return []

    load_dotenv()
    resolved_index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "").strip()
    resolved_namespace = namespace or os.getenv("PINECONE_NAMESPACE", "default").strip()
    if not resolved_index_name:
        raise ValueError("Missing index_name. Pass argument or set PINECONE_INDEX_NAME in environment.")

    dense_weight = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.5"))
    bm25_weight = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
    reranker_model = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3").strip()
    reranker_max_length = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))
    use_reranker = os.getenv("USE_RERANKER", "true").strip().lower() in {"1", "true", "yes", "y"}

    dense_k = fetch_k or max(top_k * 4, top_k)
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

    dense_scores = [c.get("score", 0.0) for c in candidates]
    bm25_scores = _bm25_scores(query=query, docs=[c.get("text", "") for c in candidates])
    dense_norm = _normalize(dense_scores)
    bm25_norm = _normalize(bm25_scores)

    fused: List[Dict[str, Any]] = []
    for cand, d_score, b_score in zip(candidates, dense_norm, bm25_norm):
        row = dict(cand)
        row["dense_score"] = float(cand.get("score", 0.0))
        row["bm25_score"] = float(b_score)
        row["hybrid_score"] = dense_weight * d_score + bm25_weight * b_score
        fused.append(row)
    fused.sort(key=lambda x: x.get("hybrid_score", float("-inf")), reverse=True)

    # Keep wider pool before rerank
    rerank_pool = fused[: max(top_k * 2, top_k)]
    if use_reranker:
        reranker = Reranker(model_name=reranker_model, max_length=reranker_max_length)
        return reranker.rerank(query=query, candidates=rerank_pool, top_k=top_k)
    return rerank_pool[:top_k]
