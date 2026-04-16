from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone


class VectorStore:
    """Pinecone vector store adapter for retrieval layer."""

    def __init__(
        self,
        index_name: str,
        namespace: str = "default",
        api_key: Optional[str] = None,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("PINECONE_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("Missing PINECONE_API_KEY. Set it in env or pass api_key.")

        self.index_name = index_name
        self.namespace = namespace
        self.client = Pinecone(api_key=self.api_key)
        self.index = self.client.Index(index_name)

    def upsert(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upsert vectors into current namespace.
        Expected each item: {"id": str, "values": list[float], "metadata": dict}
        """
        if not vectors:
            return {"upserted_count": 0}
        return self.index.upsert(vectors=vectors, namespace=self.namespace)

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        include_values: bool = False,
        include_metadata: bool = True,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query top-k nearest vectors in namespace."""
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            include_values=include_values,
            include_metadata=include_metadata,
            filter=filter,
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
    ) -> Dict[str, Any]:
        """Delete vectors by ids, by filter, or all in namespace."""
        if delete_all:
            return self.index.delete(delete_all=True, namespace=self.namespace)
        if ids:
            return self.index.delete(ids=ids, namespace=self.namespace)
        if filter:
            return self.index.delete(filter=filter, namespace=self.namespace)
        raise ValueError("Provide ids, filter, or set delete_all=True.")

    def fetch(self, ids: List[str]) -> Dict[str, Any]:
        """Fetch vectors by IDs in namespace."""
        return self.index.fetch(ids=ids, namespace=self.namespace)

    def describe_index_stats(self) -> Dict[str, Any]:
        """Return index stats (includes namespace counts)."""
        return self.index.describe_index_stats()
