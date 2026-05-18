"""Retriever factory for switching between sqlite / vector / hybrid / DashScope modes."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

from industry_agent.config import settings
from industry_agent.rag.hybrid_retriever import HybridRetriever
from industry_agent.rag.retriever import SQLiteRetriever
from industry_agent.rag.vector_store import DisabledVectorSearcher, SQLiteVectorSearcher, describe_vector_retrieval

logger = logging.getLogger(__name__)

try:
    from industry_agent.rag.cross_encoder import CrossEncoderReranker
except ImportError:
    CrossEncoderReranker = None  # type: ignore[assignment]

try:
    from industry_agent.rag.dashscope import DashScopeReranker
except ImportError:
    DashScopeReranker = None  # type: ignore[assignment]

try:
    from industry_agent.rag.bm25_retriever import BM25Retriever
except ImportError:
    BM25Retriever = None  # type: ignore[assignment]


def _load_all_chunks(db_path: str) -> list[dict[str, Any]]:
    """Load all chunks from SQLite index for BM25 index building."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT chunk_id, title, text, product_name FROM chunks"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


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

    # ── DashScope mode ──────────────────────────────────────────────
    if settings.dashscope_enabled:
        db_path = str(settings.processed_dir / "index.sqlite")

        # BM25 retriever
        bm25 = None
        if BM25Retriever is not None:
            bm25_index_path = settings.processed_dir / "bm25_index.pkl"
            chunks = _load_all_chunks(db_path)
            if bm25_index_path.exists():
                bm25 = BM25Retriever(index_path=bm25_index_path)
                if not bm25.is_loaded:
                    bm25 = BM25Retriever(chunks, index_path=bm25_index_path)
            else:
                bm25 = BM25Retriever(chunks, index_path=bm25_index_path)
        else:
            logger.warning("BM25Retriever not available (install rank_bm25)")

        # DashScope reranker
        reranker = None
        if DashScopeReranker is not None:
            reranker = DashScopeReranker(
                api_key=settings.dashscope_api_key,
                model=settings.dashscope_rerank_model,
                top_k=settings.dashscope_rerank_top_k,
                base_url=settings.dashscope_rerank_url,
            )
        else:
            logger.warning("DashScopeReranker not available")

        sqlite_retriever = SQLiteRetriever(
            vector_searcher=SQLiteVectorSearcher(),
            bm25_retriever=bm25,
        )
        return HybridRetriever(
            sqlite_retriever=sqlite_retriever,
            cross_encoder=reranker,
        )

    # ── Original mode ───────────────────────────────────────────────
    cross_encoder = None
    if CrossEncoderReranker is not None:
        enable_ce = os.getenv("INDUSTRY_AGENT_ENABLE_CROSS_ENCODER", "1").strip().lower()
        if enable_ce in {"1", "true", "on"}:
            cross_encoder = CrossEncoderReranker()
    return HybridRetriever(cross_encoder=cross_encoder)
