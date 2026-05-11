"""Retrieval skill wrapping the configured retriever stack."""

from __future__ import annotations

from typing import Any

from industry_agent.agent.skills import BaseSkill, SkillResult
from industry_agent.config import settings
from industry_agent.rag.factory import create_retriever


class RetrievalSkill(BaseSkill):
    name = "retrieval"
    description = "检索技能：根据用户问题从知识库中检索相关文档片段"

    def __init__(self, mode: str | None = None) -> None:
        self.mode = (mode or settings.retrieval_mode).strip().lower()
        self._retriever: Any | None = None

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = self._create_retriever()
        return self._retriever

    def _create_retriever(self) -> Any:
        return create_retriever(self.mode)

    def execute(self, *, query: str, limit: int = 10, **kwargs: Any) -> SkillResult:
        try:
            results = self.retriever.search(query, limit=limit)
            return SkillResult(
                success=True,
                data=results,
                metadata={"mode": self.mode, "query": query, "result_count": len(results)},
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                error=str(exc),
                metadata={"mode": self.mode, "query": query},
            )
