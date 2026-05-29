"""Lightweight vector index for hybrid retrieval.

The default implementation is dependency-free: it turns chunk text into a
normalized hashing n-gram vector and stores the vector as a SQLite BLOB.  This
is not a replacement for a neural embedding model, but it gives the project a
real vector retrieval channel that can later be swapped for BGE/E5 embeddings
without changing the retriever contract.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import os
import re
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from industry_agent.config import settings

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

# Milvus vector database (optional)
_MILVUS_TIMEOUT: float = 15.0  # seconds, per-operation timeout for Milvus calls

try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    MilvusClient = None  # type: ignore[assignment]
    DataType = None  # type: ignore[assignment]

if MilvusClient is not None:
    # Tell the embedded Milvus Lite (Go gRPC) server to accept pings more
    # frequently than its default 5-min minimum.  Without this the server
    # sends GOAWAY "too_many_pings" and drops the connection repeatedly.
    import os as _os
    _os.environ["GRPC_GO_KEEPALIVE_MIN_TIME"] = "30s"

if TYPE_CHECKING:
    from industry_agent.kb.models import KnowledgeChunk


DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("INDUSTRY_AGENT_EMBEDDING_MODEL", "BAAI/bge-m3"),
)
DEFAULT_VECTOR_DIMENSIONS = int(os.getenv("INDUSTRY_AGENT_VECTOR_DIMENSIONS", "1024"))
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
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        # get_embedding_dimension is the new name; fall back to old name for older sentence-transformers
        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension")
        self.dimensions = int(get_dim())

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
            model = self.model
            if hasattr(model, "embed_query"):
                query_vector = model.embed_query(query)
            else:
                query_vector = model.embed(query)
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


class MilvusVectorSearcher:
    """Vector searcher backed by Milvus (Lite or standalone)."""

    def __init__(
        self,
        db_path: Path = settings.processed_dir / "index.sqlite",
        *,
        config: VectorSearchConfig | None = None,
        milvus_uri: str | None = None,
        milvus_token: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        if MilvusClient is None:
            raise RuntimeError("pymilvus is not installed. Run: pip install pymilvus")

        self.db_path = db_path
        self.config = config or VectorSearchConfig(index_path=db_path)
        self.milvus_uri = milvus_uri or settings.milvus_uri
        self.milvus_token = milvus_token or settings.milvus_token
        self.collection_name = collection_name or settings.milvus_collection
        self._model: Any = None
        self._client: MilvusClient | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            if settings.dashscope_enabled and settings.dashscope_api_key:
                from industry_agent.rag.dashscope import DashScopeEmbeddingModel
                self._model = DashScopeEmbeddingModel(
                    api_key=settings.dashscope_api_key,
                    model=settings.dashscope_embedding_model,
                    dimensions=settings.dashscope_embedding_dimensions,
                    base_url=settings.dashscope_base_url,
                )
            else:
                self._model = _create_embedding_model(self.config)
        return self._model

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            kwargs = {"uri": self.milvus_uri, "timeout": _MILVUS_TIMEOUT}
            if self.milvus_token:
                kwargs["token"] = self.milvus_token
            self._client = MilvusClient(**kwargs)
        return self._client

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []

        try:
            model = self.model
            if hasattr(model, "embed_query"):
                query_vector = model.embed_query(query)
            else:
                query_vector = model.embed(query)
        except Exception:
            return []
        if not any(query_vector):
            return []

        # Check collection exists and load it (with retry)
        results = None
        for attempt in range(2):
            try:
                client = self.client
                if not client.has_collection(self.collection_name):
                    break
                client.load_collection(self.collection_name)
                results = client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=limit,
                    output_fields=["chunk_id"],
                    search_params={"metric_type": "IP", "params": {"nprobe": 16}},
                )
                break
            except Exception as exc:
                logger.warning("Milvus search error (attempt %d/2, %.1fs timeout): %s",
                               attempt + 1, _MILVUS_TIMEOUT, exc)
                if attempt == 1:
                    return []
                # Force reconnect on next iteration
                self._client = None
                time.sleep(0.5)

        if not results or not results[0]:
            return []

        # Get chunk_ids from Milvus results, filter by similarity threshold
        chunk_ids: list[str] = []
        score_map: dict[str, float] = {}
        _sim_threshold = float(os.getenv("MILVUS_SIMILARITY_THRESHOLD", "0.15"))
        for hit in results[0]:
            score = float(hit.get("distance", 0))
            if score < _sim_threshold:
                continue
            # chunk_id is in output_fields (merged to top-level), fallback to entity
            cid = str(hit.get("chunk_id") or hit.get("entity", {}).get("chunk_id", ""))
            if cid:
                chunk_ids.append(cid)
                score_map[cid] = score

        if not chunk_ids:
            return []

        # Look up full chunk rows from SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                placeholders = ",".join("?" for _ in chunk_ids)
                rows = conn.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return []

        # Build result rows preserving Milvus ranking
        row_map = {str(row["chunk_id"]): dict(row) for row in rows}
        scored: list[dict[str, Any]] = []
        for cid in chunk_ids:
            if cid not in row_map:
                continue
            record = row_map[cid]
            record["_vector_score"] = round(score_map.get(cid, 0), 6)
            record["_retrieval_channels"] = ["vector"]
            record.setdefault("fts_hit", 0)
            record.setdefault("fts_rank", None)
            scored.append(record)

        return scored[:limit]




def build_milvus_vector_index(
    chunks: list["KnowledgeChunk"],
    *,
    config: VectorSearchConfig | None = None,
    milvus_uri: str | None = None,
    milvus_token: str | None = None,
    collection_name: str | None = None,
    context_pad_chars: int = 100,
    drop_existing: bool = True,
) -> dict[str, Any]:
    """Build vector index in Milvus from KnowledgeChunks.

    Creates/recreates the Milvus collection, embeds all chunks with the
    configured embedding model, and inserts vectors into Milvus.
    """
    if MilvusClient is None:
        return {"enabled": False, "status": "pymilvus_not_installed", "chunk_count": 0}

    active = config or VectorSearchConfig()
    if not active.enabled:
        return {"enabled": False, "status": "disabled", "chunk_count": 0}

    if settings.dashscope_enabled and settings.dashscope_api_key:
        from industry_agent.rag.dashscope import DashScopeEmbeddingModel
        model = DashScopeEmbeddingModel(
            api_key=settings.dashscope_api_key,
            model=settings.dashscope_embedding_model,
            dimensions=settings.dashscope_embedding_dimensions,
            base_url=settings.dashscope_base_url,
        )
        dimensions = settings.dashscope_embedding_dimensions
    else:
        model = _create_embedding_model(active)
        dimensions = getattr(model, "dimensions", active.dimensions)
    uri = milvus_uri or settings.milvus_uri
    token = milvus_token or settings.milvus_token
    coll = collection_name or settings.milvus_collection

    client_kwargs = {"uri": uri, "timeout": _MILVUS_TIMEOUT}
    if token:
        client_kwargs["token"] = token
    client = MilvusClient(**client_kwargs)

    # Drop existing collection if requested
    if drop_existing and client.has_collection(coll):
        client.drop_collection(coll)

    # Create schema and index
    if not client.has_collection(coll):
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimensions)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="vector", metric_type="IP", index_type="IVF_FLAT", params={"nlist": 1024})
        client.create_collection(collection_name=coll, schema=schema, index_params=index_params)
        logger.info("Created Milvus collection '%s' (dim=%d, uri=%s)", coll, dimensions, uri)
    client.load_collection(coll)

    # Build adjacency map for context-padded embeddings
    pad_map: dict[str, tuple[str, str]] = {}
    if context_pad_chars > 0 and chunks:
        from collections import defaultdict
        by_manual: dict[str, list["KnowledgeChunk"]] = defaultdict(list)
        for c in chunks:
            by_manual[c.manual_id].append(c)
        for mid, group in by_manual.items():
            group.sort(key=lambda c: c.chunk_index)
            for i, c in enumerate(group):
                prev_text = group[i - 1].text if i > 0 else ""
                next_text = group[i + 1].text if i < len(group) - 1 else ""
                pad_map[c.chunk_id] = (prev_text, next_text)

    def _text_for(chunk: "KnowledgeChunk") -> str:
        prev, nxt = pad_map.get(chunk.chunk_id, ("", ""))
        return _chunk_embedding_text(chunk, prev_text=prev, next_text=nxt, pad_chars=context_pad_chars)

    # Embed and insert in batches
    batch_size = 64
    total = len(chunks)
    inserted = 0
    for start in range(0, total, batch_size):
        batch = chunks[start:start + batch_size]
        data = []
        for chunk in batch:
            vector = model.embed(_text_for(chunk))
            data.append({
                "chunk_id": chunk.chunk_id,
                "vector": vector,
            })
        client.insert(collection_name=coll, data=data)
        inserted += len(batch)
        logger.info("  Milvus insert progress: %d/%d", inserted, total)

    logger.info("Milvus vector index built: %d chunks in '%s'", total, coll)
    return {
        "enabled": True,
        "status": "built",
        "embedding_model": active.embedding_model,
        "dimensions": dimensions,
        "chunk_count": total,
        "collection": coll,
        "uri": uri,
    }


def build_chunk_vector_index(
    conn: sqlite3.Connection,
    chunks: list["KnowledgeChunk"],
    *,
    config: VectorSearchConfig | None = None,
    context_pad_chars: int = 100,
) -> dict[str, Any]:
    """Create and populate the chunk vector table inside the SQLite index.

    If ``context_pad_chars > 0``, each chunk's embedding text is padded with
    adjacent chunk content (tail of previous chunk, head of next chunk) for
    better semantic representation of short chunks.
    """

    active = config or VectorSearchConfig()
    if not active.enabled:
        logger.info("Vector index build skipped (disabled by config / env var)")
        return {"enabled": False, "status": "disabled", "chunk_count": 0}
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

    # Build adjacency map for context-padded embeddings
    pad_map: dict[str, tuple[str, str]] = {}
    if context_pad_chars > 0 and chunks:
        from collections import defaultdict
        by_manual: dict[str, list["KnowledgeChunk"]] = defaultdict(list)
        for c in chunks:
            by_manual[c.manual_id].append(c)
        for mid, group in by_manual.items():
            group.sort(key=lambda c: c.chunk_index)
            for i, c in enumerate(group):
                prev_text = group[i - 1].text if i > 0 else ""
                next_text = group[i + 1].text if i < len(group) - 1 else ""
                pad_map[c.chunk_id] = (prev_text, next_text)

    def _text_for(chunk: "KnowledgeChunk") -> str:
        prev, nxt = pad_map.get(chunk.chunk_id, ("", ""))
        return _chunk_embedding_text(chunk, prev_text=prev, next_text=nxt, pad_chars=context_pad_chars)

    # Batch embedding path (DashScope) for efficiency
    if hasattr(model, "embed_batch"):
        texts = [_text_for(chunk) for chunk in chunks]
        try:
            all_vectors = model.embed_batch(texts, text_type="document")
        except Exception as exc:
            logger.exception("DashScope batch embedding failed: %s", exc)
            all_vectors = []
        rows = [
            (
                chunk.chunk_id,
                active.embedding_model,
                dimensions,
                encode_vector(vec),
            )
            for chunk, vec in zip(chunks, all_vectors)
            if vec
        ]
    else:
        # Original per-chunk path (sentence-transformers / hashing)
        rows = [
            (
                chunk.chunk_id,
                active.embedding_model,
                dimensions,
                encode_vector(model.embed(_text_for(chunk))),
            )
            for chunk in chunks
        ]
    conn.executemany(
        """
        INSERT INTO chunk_vectors (chunk_id, embedding_model, dimensions, vector)
        VALUES (?, ?, ?, ?)
        """,
        rows,
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
    conn.commit()
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
    backend: str = "sqlite",
) -> dict[str, str | int | bool]:
    """Return deployment-facing vector retrieval status."""

    active = config or VectorSearchConfig(index_path=db_path)
    status: dict[str, str | int | bool] = {
        "enabled": active.enabled,
        "embedding_model": active.embedding_model,
        "dimensions": active.dimensions,
        "index_path": str(db_path),
        "backend": backend,
        "status": "not_built",
        "chunk_count": 0,
    }
    if backend == "milvus":
        status["milvus_uri"] = str(settings.milvus_uri)
        status["milvus_collection"] = settings.milvus_collection
        status["status"] = "ready" if active.enabled else "disabled"
        return status

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


def _create_embedding_model(
    config: VectorSearchConfig,
) -> HashingEmbeddingModel | SentenceTransformerEmbeddingModel | Any:
    # Always use the configured embedding model directly.
    # DashScope embedding is NOT used even when dashscope_enabled=True;
    # dashscope_enabled only controls reranker and LLM selection.
    model_name = str(config.embedding_model).strip()
    if model_name and model_name != "hashing-ngram-v1":
        return SentenceTransformerEmbeddingModel(model_name)
    return HashingEmbeddingModel(dimensions=config.dimensions)


def _chunk_embedding_text(chunk: "KnowledgeChunk", *, prev_text: str = "", next_text: str = "", pad_chars: int = 0) -> str:
    metadata = chunk.metadata or {}
    domain = str(metadata.get("domain_label") or "")
    semantic_type = str(metadata.get("semantic_type") or "")
    body = chunk.text or ""
    if pad_chars > 0 and body:
        if prev_text:
            body = prev_text[-pad_chars:] + " " + body
        if next_text:
            body = body + " " + next_text[:pad_chars]
    return "\n".join(
        part
        for part in (
            chunk.product_name,
            domain,
            semantic_type,
            chunk.title,
            body,
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
