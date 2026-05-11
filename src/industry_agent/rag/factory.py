"""Retriever factory for switching between sqlite / vector / hybrid modes."""

from __future__ import annotations

from typing import Any

from industry_agent.config import settings
from industry_agent.rag.hybrid_retriever import HybridRetriever
from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.rag.vector_store import DisabledVectorSearcher, SQLiteVectorSearcher, describe_vector_retrieval


class VectorOnlyRetriever:
    """Dense-only retriever wrapper with normalized score metadata."""

    def __init__(self) -> None:
        self.searcher = SQLiteVectorSearcher()

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        results = self.searcher.search(query, limit=limit)
        normalized: list[dict[str, Any]] = []
        for row in results:
            item = dict(row)
            item["_score"] = round(float(item.get("_vector_score", 0.0)) * 10.0, 3)
            item.setdefault("_retrieval_source", "vector")
            normalized.append(item)
        return normalized

    def retrieval_status(self) -> dict[str, Any]:
        return {
            "strategy": "vector_only",
            "channels": ["vector"],
            "vector": describe_vector_retrieval(),
        }


def create_retriever(mode: str | None = None) -> Any:
    normalized = (mode or settings.retrieval_mode).strip().lower()
    if normalized in {"sqlite", "lexical"}:
        return SQLiteRetriever(vector_searcher=DisabledVectorSearcher())
    if normalized == "vector":
        return VectorOnlyRetriever()
    return HybridRetriever()
