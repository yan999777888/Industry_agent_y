"""Hybrid retriever using sparse SQLite retrieval plus dense vector retrieval with RRF."""

from __future__ import annotations

from typing import Any

from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.rag.vector_store import (
    DisabledVectorSearcher,
    SQLiteVectorSearcher,
    VectorSearcher,
    describe_vector_retrieval,
)

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = RRF_K,
    key_field: str = "chunk_id",
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, row in enumerate(ranked):
            key = str(row.get(key_field, "")).strip()
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(key, dict(row))

    merged: list[dict[str, Any]] = []
    for key, score in scores.items():
        row = dict(doc_map[key])
        row["_rrf_score"] = round(score, 6)
        merged.append(row)
    merged.sort(key=lambda item: float(item.get("_rrf_score", 0.0)), reverse=True)
    return merged


class HybridRetriever:
    """Hybrid sparse+dense retriever following the Industry_agent_y strategy."""

    def __init__(
        self,
        sqlite_retriever: SQLiteRetriever | None = None,
        vector_retriever: VectorSearcher | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self.sqlite_retriever = sqlite_retriever or SQLiteRetriever(vector_searcher=DisabledVectorSearcher())
        self.vector_retriever = vector_retriever or SQLiteVectorSearcher()
        self.rrf_k = rrf_k

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        fetch_limit = max(limit * 2, 10)
        sparse_results = self.sqlite_retriever.search(query, limit=fetch_limit)
        vector_results = self.vector_retriever.search(query, limit=fetch_limit)
        if not vector_results:
            return sparse_results[:limit]
        return reciprocal_rank_fusion([sparse_results, vector_results], k=self.rrf_k)[:limit]

    def retrieval_status(self) -> dict[str, Any]:
        return {
            "strategy": "hybrid_rrf",
            "channels": ["sqlite", "vector", "rrf"],
            "rrf_k": self.rrf_k,
            "vector": describe_vector_retrieval(),
        }

    def search_with_debug(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        fetch_limit = max(limit * 2, 10)
        sparse_results = self.sqlite_retriever.search(query, limit=fetch_limit)
        vector_results = self.vector_retriever.search(query, limit=fetch_limit)
        fused = reciprocal_rank_fusion([sparse_results, vector_results], k=self.rrf_k)
        return {
            "results": fused[:limit],
            "debug": {
                "sparse_count": len(sparse_results),
                "vector_count": len(vector_results),
                "fused_count": len(fused),
                "rrf_k": self.rrf_k,
            },
        }
