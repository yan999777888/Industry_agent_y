"""DashScope (Alibaba Cloud Bailian) API clients for embedding and reranking.

Provides DashScopeEmbeddingModel and DashScopeReranker that use the
text-embedding-v4 and qwen3-rerank models through DashScope's HTTP APIs.
Both classes implement the same duck-type interface as the existing
SentenceTransformerEmbeddingModel and CrossEncoderReranker respectively.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DashScopeError(Exception):
    """Base exception for DashScope API errors."""


class RateLimitError(DashScopeError):
    """Rate limit exceeded (HTTP 429)."""


class AuthError(DashScopeError):
    """Authentication failure (HTTP 401/403)."""


# ---------------------------------------------------------------------------
# Embedding  --  text-embedding-v4
# ---------------------------------------------------------------------------

class DashScopeEmbeddingModel:
    """text-embedding-v4 via DashScope's native embedding endpoint.

    Supports ``text_type`` to distinguish document indexing vs query encoding.
    Dimensions default to 1024 (the model supports 64-2048 via MRL).
    """

    MAX_BATCH_SIZE = 10  # DashScope official per-request batch limit

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
        base_url: str = "https://dashscope.aliyuncs.com",
        max_retries: int = 3,
        request_timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._client: httpx.Client | None = None
        # Native embedding endpoint under root (not under /compatible-mode/v1)
        root_url = "https://dashscope.aliyuncs.com"
        self._embed_url = (
            f"{root_url}/api/v1/services/embeddings/"
            f"text-embedding/text-embedding"
        )

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.request_timeout)
        return self._client

    # ── Public API ──────────────────────────────────────────────────────

    def embed(self, text: str, text_type: str = "document") -> list[float]:
        """Embed a single text (default: document mode for indexing)."""
        return self.embed_batch([text], text_type=text_type)[0]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single *query* text (uses text_type='query')."""
        return self.embed(text, text_type="query")

    def embed_batch(
        self,
        texts: list[str],
        text_type: str = "document",
    ) -> list[list[float]]:
        """Embed multiple texts, automatically splitting into API batches.

        Args:
            texts: List of strings to embed.
            text_type: ``"document"`` for indexing, ``"query"`` for search.

        Returns:
            List of embedding vectors (each a list of floats, dimension=1024).
        """
        if not texts:
            return []
        all_results: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start : start + self.MAX_BATCH_SIZE]
            all_results.extend(self._call_api(batch, text_type))
        return all_results

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── Internal ─────────────────────────────────────────────────────────

    def _call_api(
        self, texts: list[str], text_type: str
    ) -> list[list[float]]:
        """Call the DashScope native embedding endpoint with retry logic.

        Uses the native DashScope endpoint (fully qualified) rather than the
        OpenAI-compatible one, because the native endpoint supports the
        ``text_type`` parameter which significantly affects retrieval quality.
        """
        # Filter out empty texts to avoid API errors
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * self.dimensions] * len(texts)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": {
                "texts": valid_texts,
            },
            "parameters": {
                "text_type": text_type,
                "dimensions": self.dimensions,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = self._embed_url

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post(url, json=payload, headers=headers)

                # Rate limit
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response, default=5.0)
                    logger.warning(
                        "DashScope embed rate limited (429), retrying "
                        "after %.1fs (attempt %d/%d)",
                        retry_after, attempt, self.max_retries,
                    )
                    time.sleep(retry_after)
                    continue

                # Auth failure
                if response.status_code in (401, 403):
                    raise AuthError(
                        f"DashScope embedding auth failed "
                        f"({response.status_code}): {response.text[:200]}"
                    )

                # Log 4xx response body for debugging (except auth which is handled above)
                if 400 <= response.status_code < 500 and response.status_code not in (401, 403, 429):
                    logger.error(
                        "DashScope embed client error %s: %s",
                        response.status_code, response.text[:500],
                    )

                response.raise_for_status()
                data = response.json()

                # Native format: output.embeddings[*].embedding -> list[float]
                embeddings = data.get("output", {}).get("embeddings", [])
                # Try both "embedding" and "text_embedding" field names
                first = embeddings[0] if embeddings else {}
                vec_key = "embedding" if "embedding" in first else "text_embedding"
                return [
                    e[vec_key]
                    for e in sorted(embeddings, key=lambda x: x.get("embedding_index", 0))
                ]

            except (AuthError, DashScopeError):
                raise
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2.0 ** attempt
                    logger.info("DashScope embed HTTP error, retry in %.1fs: %s", wait, exc)
                    time.sleep(wait)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2.0 ** attempt
                    logger.info("DashScope embed timeout, retry in %.1fs", wait)
                    time.sleep(wait)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(1.0)

        raise DashScopeError(
            f"Embedding failed after {self.max_retries} retries"
        ) from last_exc


# ---------------------------------------------------------------------------
# Reranker  --  qwen3-rerank
# ---------------------------------------------------------------------------

class DashScopeReranker:
    """Re-ranks candidates using DashScope's qwen3-rerank API.

    Interface matches what ``HybridRetriever`` expects:
    ``rerank(query, candidates) -> list[dict]`` sorted by relevance.
    """

    MAX_DOCS = 500  # DashScope per-request limit

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        top_k: int = 20,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1",
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    # ── Public API ──────────────────────────────────────────────────────

    def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Re-rank top-K candidates, return full list sorted by relevance.

        Args:
            query: The original user query.
            candidates: List of chunk dicts with at least ``title`` and ``text``.

        Returns:
            The full candidates list with the top ``top_k`` re-sorted
            by ``_cross_encoder_score`` (0.0-1.0). Candidates beyond top_k
            retain their original order at the end.
        """
        to_score = candidates[: self.top_k]
        if not to_score:
            return candidates

        # Build document strings (title + text, truncated for API limits)
        documents: list[str] = []
        for c in to_score:
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
            combined = f"{title} {text}".strip()
            # qwen3-rerank supports 4K tokens per doc; 2048 chars is safe
            documents.append(combined[:2048])

        try:
            scores = self._call_rerank_api(query, documents)
        except Exception as exc:
            logger.warning("DashScope reranking failed (using fallback order): %s", exc)
            return candidates

        for chunk, score in zip(to_score, scores):
            chunk["_cross_encoder_score"] = round(float(score), 6)

        to_score.sort(
            key=lambda c: float(c.get("_cross_encoder_score", 0.0)),
            reverse=True,
        )
        return to_score + candidates[self.top_k :]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── Internal ─────────────────────────────────────────────────────────

    def _call_rerank_api(
        self, query: str, documents: list[str]
    ) -> list[float]:
        """Call the DashScope rerank endpoint."""
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/reranks"

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post(url, json=payload, headers=headers)

                if response.status_code == 429:
                    retry_after = _parse_retry_after(response, default=5.0)
                    logger.warning(
                        "DashScope rerank rate limited, retry in %.1fs", retry_after
                    )
                    time.sleep(retry_after)
                    continue

                if response.status_code in (401, 403):
                    raise AuthError(
                        f"DashScope rerank auth failed ({response.status_code})"
                    )

                response.raise_for_status()
                data = response.json()

                # Format: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}
                results = sorted(
                    data.get("results", []),
                    key=lambda x: int(x.get("index", 0)),
                )
                return [float(r.get("relevance_score", 0.0)) for r in results]

            except (AuthError, DashScopeError):
                raise
            except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(2.0 ** attempt)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(1.0)

        raise DashScopeError(
            f"Rerank failed after {self.max_retries} retries"
        ) from last_exc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_retry_after(
    response: httpx.Response, default: float = 5.0
) -> float:
    """Extract Retry-After header as seconds, or return default."""
    raw = response.headers.get("Retry-After", "")
    if raw.isdigit():
        return float(raw)
    return default
