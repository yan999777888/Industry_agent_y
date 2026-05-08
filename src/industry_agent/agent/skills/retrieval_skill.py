"""Retrieval skill — wraps hybrid/sparse/dense retriever for the orchestrator.

Provides a unified retrieval interface that can operate in three modes:
- "sqlite": keyword-based retrieval only (current default)
- "vector": FAISS semantic retrieval only
- "hybrid": combined sparse+dense with RRF fusion
"""

from __future__ import annotations

from typing import Any

from industry_agent.agent.skills import BaseSkill, SkillResult
from industry_agent.config import settings


class RetrievalSkill(BaseSkill):
    """Unified retrieval skill supporting multiple backends."""

    name = "retrieval"
    description = "检索技能：根据用户问题从知识库中检索相关文档片段"

    def __init__(self, mode: str | None = None) -> None:
        self.mode = mode or settings.retrieval_mode
        self._retriever = None

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = self._create_retriever()
        return self._retriever

    def _create_retriever(self) -> Any:
        from industry_agent.rag.retriever import SQLiteRetriever

        if self.mode == "sqlite":
            return SQLiteRetriever()
        elif self.mode == "vector":
            from industry_agent.rag.vector_store import VectorStoreRetriever

            return VectorStoreRetriever()
        elif self.mode == "hybrid":
            from industry_agent.rag.hybrid_retriever import HybridRetriever

            return HybridRetriever()
        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")

    def execute(
        self,
        *,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> SkillResult:
        """Retrieve documents for the given query.

        Args:
            query: The search query string.
            limit: Maximum number of results.

        Returns:
            SkillResult with data=list[dict] of retrieved chunks.
        """
        try:
            results = self.retriever.search(query, limit=limit)
            return SkillResult(
                success=True,
                data=results,
                metadata={
                    "mode": self.mode,
                    "query": query,
                    "result_count": len(results),
                },
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                error=str(exc),
                metadata={"mode": self.mode, "query": query},
            )

    def is_available(self) -> bool:
        try:
            _ = self.retriever
            return True
        except Exception:
            return False
