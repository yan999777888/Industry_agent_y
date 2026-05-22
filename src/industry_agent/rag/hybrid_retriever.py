"""Hybrid retriever using sparse SQLite retrieval plus dense vector retrieval with RRF."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.rag.vector_store import (
    DisabledVectorSearcher,
    MilvusVectorSearcher,
    SQLiteVectorSearcher,
    VectorSearcher,
    describe_vector_retrieval,
)
from industry_agent.config import settings

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = RRF_K,
    key_field: str = "chunk_id",
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}
    vec_scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, row in enumerate(ranked):
            key = str(row.get(key_field, "")).strip()
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(key, dict(row))
            # Track best _vector_score across lists
            vs = float(row.get("_vector_score") or 0.0)
            if vs > vec_scores.get(key, 0.0):
                vec_scores[key] = vs

    merged: list[dict[str, Any]] = []
    for key, score in scores.items():
        row = dict(doc_map[key])
        row["_rrf_score"] = round(score, 6)
        row["_vector_score"] = vec_scores.get(key, 0.0)
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
        cross_encoder: Any = None,
    ) -> None:
        self.sqlite_retriever = sqlite_retriever or SQLiteRetriever(vector_searcher=DisabledVectorSearcher())
        if vector_retriever is not None:
            self.vector_retriever = vector_retriever
        elif settings.vector_backend == "milvus":
            self.vector_retriever = MilvusVectorSearcher()
        else:
            self.vector_retriever = SQLiteVectorSearcher()
        self.rrf_k = rrf_k
        self.cross_encoder = cross_encoder

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        fetch_limit = max(limit * 20, 180)
        sparse_results = self.sqlite_retriever.search(query, limit=fetch_limit)
        vector_results = self.vector_retriever.search(query, limit=fetch_limit)
        print(f"HYBRID_TRACE: q={query[:30]} vec={len(vector_results)} sparse={len(sparse_results)}", flush=True)
        if not vector_results:
            if self.cross_encoder is not None:
                out = self.cross_encoder.rerank(query, sparse_results)[:limit]
                print(f"HYBRID_TRACE_EARLY: len={len(out)} keys={sorted(k for k in out[0].keys() if k.startswith('_')) if out else []}", flush=True)
                return out
            return sparse_results[:limit]
        fused = reciprocal_rank_fusion([sparse_results, vector_results], k=self.rrf_k)
        if self.cross_encoder is not None:
            fused = self.cross_encoder.rerank(query, fused)
        if fused:
            _r0 = fused[0]
            print(f"HYBRID_TRACE_FUSED: len={len(fused)} _rrf={_r0.get('_rrf_score', 'MISS')} _vec={_r0.get('_vector_score', 'MISS')} _ce={_r0.get('_cross_encoder_score', 'MISS')}", flush=True)
        return fused[:limit]

    def retrieval_status(self) -> dict[str, Any]:
        _vb = "milvus" if isinstance(self.vector_retriever, MilvusVectorSearcher) else "sqlite"
        return {
            "strategy": "hybrid_rrf",
            "channels": ["keyword", _vb, "rrf"],
            "vector_backend": _vb,
            "rrf_k": self.rrf_k,
            "vector": describe_vector_retrieval(backend=_vb),
        }

    def search_with_debug(self, query: str, *, limit: int = 5) -> dict[str, Any]:
        fetch_limit = max(limit * 20, 180)
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
