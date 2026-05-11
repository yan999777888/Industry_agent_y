"""Compatibility embedding helpers adapted to the current vector pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from industry_agent.config import settings
from industry_agent.rag.vector_store import VectorSearchConfig, _create_embedding_model

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]


class EmbeddingManager:
    """Thin compatibility wrapper over the current embedding backends."""

    def __init__(self, model_name: str | None = None, *, dimensions: int | None = None) -> None:
        self.model_name = (model_name or settings.embedding_model).strip()
        self.config = VectorSearchConfig(
            embedding_model=self.model_name,
            dimensions=dimensions or VectorSearchConfig().dimensions,
        )
        self._model = _create_embedding_model(self.config)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def dimension(self) -> int:
        return int(getattr(self._model, "dimensions", self.config.dimensions))

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> Any:
        del batch_size, show_progress, normalize
        vectors = [self._model.embed(text) for text in texts]
        if np is not None:
            return np.asarray(vectors, dtype=np.float32)
        return vectors

    def encode_query(self, query: str) -> Any:
        encoded = self.encode([query])
        if np is not None:
            return encoded[0]
        return encoded[0] if encoded else []
