"""Retriever factory for switching between sqlite / vector / hybrid modes."""

from __future__ import annotations

import os
from typing import Any

from industry_agent.config import settings
from industry_agent.rag.hybrid_retriever import HybridRetriever
from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.rag.vector_store import DisabledVectorSearcher, SQLiteVectorSearcher, describe_vector_retrieval

try:
    from industry_agent.rag.cross_encoder import CrossEncoderReranker
except ImportError:
    CrossEncoderReranker = None  # type: ignore[assignment]


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

    cross_encoder = None
    if CrossEncoderReranker is not None:
        enable_ce = os.getenv("INDUSTRY_AGENT_ENABLE_CROSS_ENCODER", "0").strip().lower()
        if enable_ce in {"1", "true", "on"}:
            cross_encoder = CrossEncoderReranker()
    return HybridRetriever(cross_encoder=cross_encoder)
