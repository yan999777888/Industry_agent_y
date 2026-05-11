"""Lightweight vector index for hybrid retrieval.

The default implementation is dependency-free: it turns chunk text into a
normalized hashing n-gram vector and stores the vector as a SQLite BLOB.  This
is not a replacement for a neural embedding model, but it gives the project a
real vector retrieval channel that can later be swapped for BGE/E5 embeddings
without changing the retriever contract.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from industry_agent.config import settings

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from industry_agent.kb.models import KnowledgeChunk


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("INDUSTRY_AGENT_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
)
DEFAULT_VECTOR_DIMENSIONS = int(os.getenv("INDUSTRY_AGENT_VECTOR_DIMENSIONS", "384"))
VECTOR_RETRIEVAL_ENABLED = os.getenv("INDUSTRY_AGENT_ENABLE_VECTOR", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9._-]*|[0-9]+(?:\.[0-9]+)*")
_CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class VectorSearchConfig:
    """Runtime config for embedding retrieval."""

    enabled: bool = VECTOR_RETRIEVAL_ENABLED
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int = DEFAULT_VECTOR_DIMENSIONS
    index_path: Path = settings.processed_dir / "index.sqlite"


class VectorSearcher(Protocol):
    """Minimal searcher protocol expected by the hybrid retriever."""

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        """Return vector candidates shaped like SQLite chunk rows."""


class HashingEmbeddingModel:
    """Deterministic text-to-vector model with no external dependencies."""

    def __init__(self, *, dimensions: int = DEFAULT_VECTOR_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in _extract_features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big", signed=False)
            index = raw % self.dimensions
            sign = 1.0 if (raw >> 8) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbeddingModel:
    """Thin wrapper around a local sentence-transformers embedding model."""

    def __init__(self, model_name: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it before using a neural embedding model."
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimensions = int(self.model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in vector]


class DisabledVectorSearcher:
    """No-op searcher used when vector retrieval is explicitly disabled."""

    def __init__(self, config: VectorSearchConfig | None = None) -> None:
        self.config = config or VectorSearchConfig(enabled=False)

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        return []


class SQLiteVectorSearcher:
    """Vector searcher backed by the `chunk_vectors` table in index.sqlite."""

    def __init__(
        self,
        db_path: Path = settings.processed_dir / "index.sqlite",
        *,
        config: VectorSearchConfig | None = None,
    ) -> None:
        self.db_path = db_path
        self.config = config or VectorSearchConfig(index_path=db_path)
        self._model: HashingEmbeddingModel | SentenceTransformerEmbeddingModel | None = None

    @property
    def model(self) -> HashingEmbeddingModel | SentenceTransformerEmbeddingModel:
        if self._model is None:
            self._model = _create_embedding_model(self.config)
        return self._model

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        if not self.config.enabled or not self.db_path.exists():
            return []

        try:
            query_vector = self.model.embed(query)
        except Exception:
            return []
        if not any(query_vector):
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT chunks.*, chunk_vectors.vector
                    FROM chunk_vectors
                    JOIN chunks ON chunks.chunk_id = chunk_vectors.chunk_id
                    """
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return []

        scored: list[dict[str, Any]] = []
        for row in rows:
            vector = decode_vector(row["vector"])
            if len(vector) != len(query_vector):
                continue
            score = dot_product(query_vector, vector)
            if score <= 0:
                continue
            record = dict(row)
            record.pop("vector", None)
            record["_vector_score"] = round(score, 6)
            record["_retrieval_channels"] = ["vector"]
            record.setdefault("fts_hit", 0)
            record.setdefault("fts_rank", None)
            scored.append(record)

        scored.sort(key=lambda item: float(item.get("_vector_score", 0.0)), reverse=True)
        return scored[:limit]


def build_chunk_vector_index(
    conn: sqlite3.Connection,
    chunks: list["KnowledgeChunk"],
    *,
    config: VectorSearchConfig | None = None,
) -> dict[str, Any]:
    """Create and populate the chunk vector table inside the SQLite index."""

    active = config or VectorSearchConfig()
    model = _create_embedding_model(active)
    dimensions = getattr(model, "dimensions", active.dimensions)
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunk_vectors;
        DROP TABLE IF EXISTS vector_metadata;

        CREATE TABLE chunk_vectors (
          chunk_id TEXT PRIMARY KEY,
          embedding_model TEXT NOT NULL,
          dimensions INTEGER NOT NULL,
          vector BLOB NOT NULL
        );

        CREATE TABLE vector_metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO chunk_vectors (chunk_id, embedding_model, dimensions, vector)
        VALUES (?, ?, ?, ?)
        """,
        [
            (
                chunk.chunk_id,
                active.embedding_model,
                dimensions,
                encode_vector(model.embed(_chunk_embedding_text(chunk))),
            )
            for chunk in chunks
        ],
    )
    conn.executemany(
        "INSERT INTO vector_metadata (key, value) VALUES (?, ?)",
        [
            ("embedding_model", active.embedding_model),
            ("dimensions", str(dimensions)),
            ("chunk_count", str(len(chunks))),
            ("status", "built"),
        ],
    )
    return {
        "enabled": active.enabled,
        "embedding_model": active.embedding_model,
        "dimensions": dimensions,
        "chunk_count": len(chunks),
        "table": "chunk_vectors",
    }


def describe_vector_retrieval(
    *,
    db_path: Path = settings.processed_dir / "index.sqlite",
    config: VectorSearchConfig | None = None,
) -> dict[str, str | int | bool]:
    """Return deployment-facing vector retrieval status."""

    active = config or VectorSearchConfig(index_path=db_path)
    status = {
        "enabled": active.enabled,
        "embedding_model": active.embedding_model,
        "dimensions": active.dimensions,
        "index_path": str(db_path),
        "status": "not_built",
        "chunk_count": 0,
    }
    if not db_path.exists():
        return status

    try:
        conn = sqlite3.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            row = conn.execute(
                "SELECT value FROM vector_metadata WHERE key = 'embedding_model'"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return status

    status["status"] = "ready" if active.enabled else "built_disabled"
    status["chunk_count"] = int(count)
    if row and row[0]:
        status["embedding_model"] = str(row[0])
    return status


def encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(payload: bytes) -> list[float]:
    if not payload:
        return []
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _create_embedding_model(config: VectorSearchConfig) -> HashingEmbeddingModel | SentenceTransformerEmbeddingModel:
    model_name = str(config.embedding_model).strip()
    if model_name and model_name != "hashing-ngram-v1":
        return SentenceTransformerEmbeddingModel(model_name)
    return HashingEmbeddingModel(dimensions=config.dimensions)


def _chunk_embedding_text(chunk: "KnowledgeChunk") -> str:
    metadata = chunk.metadata or {}
    domain = str(metadata.get("domain_label") or "")
    semantic_type = str(metadata.get("semantic_type") or "")
    return "\n".join(
        part
        for part in (
            chunk.product_name,
            domain,
            semantic_type,
            chunk.title,
            chunk.text,
        )
        if part
    )


def _extract_features(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    features: list[str] = []
    ascii_tokens = [token for token in _ASCII_TOKEN_RE.findall(normalized) if len(token) >= 2]
    features.extend(f"w:{token}" for token in ascii_tokens)
    for first, second in zip(ascii_tokens, ascii_tokens[1:]):
        features.append(f"bw:{first}_{second}")

    for cjk in _CJK_TOKEN_RE.findall(normalized):
        if len(cjk) <= 1:
            features.append(f"c:{cjk}")
            continue
        for size in (2, 3):
            for index in range(len(cjk) - size + 1):
                features.append(f"c{size}:{cjk[index:index + size]}")
    return features
