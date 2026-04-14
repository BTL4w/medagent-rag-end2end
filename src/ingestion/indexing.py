from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from src.ingestion.chunking import MarkdownChunker
from src.ingestion.embedding import TextEmbedding


def _safe_metadata(value: Any) -> Any:
    """Convert metadata values to Pinecone-safe JSON scalars."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [item for item in value if isinstance(item, (str, int, float, bool))]
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _build_vector_payloads(chunks, vectors) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        raw_meta = chunk.metadata or {}
        payload = {
            "id": chunk.chunk_id,
            "values": vector.tolist(),
            "metadata": {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section or "",
                "subsection": chunk.subsection or "",
                "text": chunk.original_content or "",
                "enriched_text": chunk.enriched_content,
            },
        }
        for key, value in raw_meta.items():
            payload["metadata"][f"meta_{key}"] = _safe_metadata(value)
        payloads.append(payload)
    return payloads


def _ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
) -> None:
    existing = pc.list_indexes().names()
    if index_name in existing:
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def run_indexing(
    input_path: str,
    index_name: str,
    namespace: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    metric: str,
    cloud: str,
    region: str,
) -> None:
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment.")

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("[1/4] Chunking documents...")
    chunker = MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.process_jsonl(input_path)
    if not chunks:
        raise ValueError("No chunks generated from input file.")
    print(f"Generated {len(chunks)} chunks.")

    print("[2/4] Creating embeddings...")
    embedder = TextEmbedding(model_name=model_name)
    vectors = embedder.embed_batch([chunk.enriched_content for chunk in chunks], batch_size=batch_size)
    print(f"Embeddings shape: {vectors.shape}")

    print("[3/4] Connecting Pinecone...")
    pc = Pinecone(api_key=api_key)
    _ensure_index(
        pc=pc,
        index_name=index_name,
        dimension=int(vectors.shape[1]),
        metric=metric,
        cloud=cloud,
        region=region,
    )
    index = pc.Index(index_name)

    print("[4/4] Upserting vectors...")
    payloads = _build_vector_payloads(chunks, vectors)
    for i in tqdm(range(0, len(payloads), batch_size), desc="Upsert batches"):
        batch = payloads[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)

    print("Indexing completed successfully.")
    print(f"Index: {index_name}")
    print(f"Namespace: {namespace}")
    print(f"Total vectors: {len(payloads)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk + Embed + Upsert to Pinecone.")
    parser.add_argument("--input", required=True, help="Path to processed JSONL file.")
    parser.add_argument("--index-name", required=True, help="Pinecone index name.")
    parser.add_argument("--namespace", default="default", help="Pinecone namespace.")
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name.")
    parser.add_argument("--chunk-size", type=int, default=600, help="Chunk size.")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding/upsert batch size.")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "dotproduct", "euclidean"])
    parser.add_argument("--cloud", default="aws", help="Pinecone serverless cloud.")
    parser.add_argument("--region", default="us-east-1", help="Pinecone serverless region.")
    args = parser.parse_args()

    run_indexing(
        input_path=args.input,
        index_name=args.index_name,
        namespace=args.namespace,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        metric=args.metric,
        cloud=args.cloud,
        region=args.region,
    )


if __name__ == "__main__":
    main()
