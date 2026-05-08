"""Embedding model manager for Chinese text vectorization.

Uses BAAI/bge-small-zh-v1.5 by default (same as all-in-rag project).
Provides lazy-loaded model with caching for efficient batch encoding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from industry_agent.config import settings

_MODEL_CACHE: dict[str, Any] = {}


def _get_model(model_name: str | None = None) -> Any:
    """Load and cache a SentenceTransformer model."""
    name = model_name or settings.embedding_model
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


class EmbeddingManager:
    """Manages text embedding generation for RAG retrieval.

    Usage:
        mgr = EmbeddingManager()
        vectors = mgr.encode(["如何安装电钻", "电池充电方法"])
        query_vec = mgr.encode_query("电钻指示灯闪烁")
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> Any:
        """Encode a list of texts into dense vectors.

        Returns:
            np.ndarray of shape (len(texts), dim), L2-normalized if normalize=True.
        """
        import numpy as np

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> Any:
        """Encode a single query string into a vector.

        Returns:
            np.ndarray of shape (dim,), L2-normalized.
        """
        import numpy as np

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()
