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
        device: str | None = None,
    ):
        self.model_name = model_name or os.getenv(
            "INDUSTRY_AGENT_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER
        )
        self.top_k = top_k
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
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

        # Build adjacency map for short-chunk context padding
        adj_context: dict[str, tuple[str, str]] = {}
        by_manual: dict[str, list[dict[str, Any]]] = {}
        for c in candidates:
            mid = str(c.get("manual_id") or "")
            if mid:
                by_manual.setdefault(mid, []).append(c)
        for group in by_manual.values():
            group.sort(key=lambda x: int(x.get("chunk_index", 0) or 0))
            for i, c in enumerate(group):
                prev_text = group[i - 1].get("text", "") if i > 0 else ""
                next_text = group[i + 1].get("text", "") if i < len(group) - 1 else ""
                adj_context[str(c.get("chunk_id", ""))] = (prev_text, next_text)

        pairs = []
        for c in to_score:
            title = str(c.get("title", "") or "")
            text = str(c.get("text", "") or "")
            char_count = c.get("char_count", 0) or len(text)
            if isinstance(char_count, (int, float)) and 0 < char_count < 300:
                prev_text, next_text = adj_context.get(str(c.get("chunk_id", "")), ("", ""))
                if prev_text:
                    text = f"{prev_text[-150:]} {text}"
                if next_text:
                    text = f"{text} {next_text[:150]}"
            pairs.append((query, f"{title} {text}"[:1024]))

        try:
            scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
        except Exception as exc:
            logger.warning("Cross-encoder reranking failed: %s", exc)
            return candidates

        for chunk, score in zip(to_score, scores):
            chunk["_cross_encoder_score"] = round(float(score), 6)

        # Short chunk boost — counteract cross-encoder length bias
        for chunk in to_score:
            cc = chunk.get("char_count", 0)
            if isinstance(cc, (int, float)) and cc > 0:
                boost = max(0.0, 1.0 - cc / 100.0) * 1.0
                chunk["_cross_encoder_score"] = round(
                    float(chunk["_cross_encoder_score"]) * (1.0 + boost), 6
                )

        to_score.sort(
            key=lambda c: float(c.get("_cross_encoder_score", 0.0)),
            reverse=True,
        )

        return to_score + candidates[self.top_k :]
