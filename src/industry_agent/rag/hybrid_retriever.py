"""Hybrid retriever combining sparse (SQLite) and dense (FAISS) retrieval.

Implements Reciprocal Rank Fusion (RRF) for merging ranked lists,
following the pattern from all-in-rag Chapter 4.
"""

from __future__ import annotations

from typing import Any

from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.rag.vector_store import VectorStoreRetriever, VectorStoreConfig
from industry_agent.config import settings

# RRF constant — standard value from the original paper
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = RRF_K,
    key_field: str = "chunk_id",
) -> list[dict[str, Any]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists, each containing dicts with key_field.
        k: RRF tuning constant (default 60).
        key_field: Field used to identify unique documents across lists.

    Returns:
        Merged and re-ranked list of dicts with _rrf_score added.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_key = str(doc.get(key_field, ""))
            if not doc_key:
                continue
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
            if doc_key not in doc_map:
                doc_map[doc_key] = dict(doc)

    merged: list[dict[str, Any]] = []
    for doc_key, rrf_score in rrf_scores.items():
        row = doc_map[doc_key]
        row["_rrf_score"] = round(rrf_score, 6)
        merged.append(row)

    merged.sort(key=lambda x: x["_rrf_score"], reverse=True)
    return merged


class HybridRetriever:
    """Combines SQLite keyword search with FAISS vector search using RRF.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("电钻指示灯闪烁", limit=5)
    """

    def __init__(
        self,
        sqlite_retriever: SQLiteRetriever | None = None,
        vector_retriever: VectorStoreRetriever | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self.sqlite_retriever = sqlite_retriever or SQLiteRetriever()
        self.vector_retriever = vector_retriever or VectorStoreRetriever()
        self.rrf_k = rrf_k

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Hybrid search: sparse + dense with RRF fusion.

        Fetches 2x candidates from each source, fuses with RRF,
        then returns top-limit results.
        """
        fetch_limit = max(limit * 2, 10)

        # Sparse retrieval (SQLite keyword-based)
        sparse_results = self.sqlite_retriever.search(query, limit=fetch_limit)

        # Dense retrieval (FAISS vector-based)
        vector_results = self.vector_retriever.search(query, limit=fetch_limit)

        # If vector index is not available, fall back to sparse-only
        if not vector_results:
            return sparse_results[:limit]

        # RRF fusion
        merged = reciprocal_rank_fusion(
            [sparse_results, vector_results],
            k=self.rrf_k,
        )

        return merged[:limit]

    def search_with_debug(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        """Search with debug info showing each retriever's contribution."""
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
