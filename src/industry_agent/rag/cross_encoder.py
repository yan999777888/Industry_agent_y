"""Optional cross-encoder re-ranker for improving retrieval precision.

Plugs between hybrid retrieval and evidence filtering to re-score candidates
using a cross-encoder model. Controlled by INDUSTRY_AGENT_ENABLE_CROSS_ENCODER.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Re-ranks retrieval candidates using a cross-encoder model.

    Runs on CPU. For 30 candidates, re-ranking takes ~8-15s.
    Use top_k to control speed vs. quality trade-off.
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int = 30,
        device: str = "cpu",
    ):
        self.model_name = model_name or os.getenv(
            "INDUSTRY_AGENT_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER
        )
        self.top_k = top_k
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512,
                )
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-rank top-K candidates by cross-encoder relevance score."""
        to_score = candidates[: self.top_k]
        if not to_score:
            return candidates

        pairs = [
            (query, f"{c.get('title', '')} {c.get('text', '')}"[:1024])
            for c in to_score
        ]

        try:
            scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Cross-encoder reranking failed: %s", exc)
            return candidates

        for chunk, score in zip(to_score, scores):
            chunk["_cross_encoder_score"] = round(float(score), 6)

        to_score.sort(
            key=lambda c: float(c.get("_cross_encoder_score", 0.0)),
            reverse=True,
        )

        return to_score + candidates[self.top_k :]
