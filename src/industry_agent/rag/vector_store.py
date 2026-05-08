"""FAISS-backed vector retriever for semantic search.

Provides VectorStoreRetriever with the same search() interface as
SQLiteRetriever, returning compatible dict results with _score fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from industry_agent.config import settings
from industry_agent.rag.embedding import EmbeddingManager


@dataclass
class VectorStoreConfig:
    index_path: Path = settings.vector_index_path
    embedding_model: str = settings.embedding_model


class VectorStoreRetriever:
    """Semantic retriever backed by a FAISS index.

    Each indexed chunk is stored with metadata so search() returns
    dicts compatible with SQLiteRetriever output.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        embedding_manager: EmbeddingManager | None = None,
    ) -> None:
        self.config = config or VectorStoreConfig()
        self.embedder = embedding_manager or EmbeddingManager(self.config.embedding_model)
        self._index: Any | None = None
        self._metadatas: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Save FAISS index and metadata to disk."""
        import faiss

        save_path = path or self.config.index_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            raise RuntimeError("No index to save. Build or load an index first.")

        faiss.write_index(self._index, str(save_path))

        meta_path = save_path.with_suffix(".meta.jsonl")
        with meta_path.open("w", encoding="utf-8") as f:
            for meta in self._metadatas:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def load(self, path: Path | None = None) -> None:
        """Load FAISS index and metadata from disk."""
        import faiss

        load_path = path or self.config.index_path
        if not load_path.exists():
            raise FileNotFoundError(f"Vector index not found: {load_path}")

        self._index = faiss.read_index(str(load_path))

        meta_path = load_path.with_suffix(".meta.jsonl")
        self._metadatas = []
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._metadatas.append(json.loads(line))

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build(
        self,
        chunks: list[dict[str, Any]],
        *,
        text_field: str = "text",
        batch_size: int = 64,
    ) -> None:
        """Build FAISS index from a list of chunk dicts.

        Each chunk must have at least a text_field. Other fields are stored
        as metadata and returned during search.
        """
        import faiss
        import numpy as np

        texts = [str(chunk.get(text_field, "")) for chunk in chunks]
        embeddings = self.embedder.encode(texts, batch_size=batch_size, show_progress=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vectors)
        index.add(embeddings)

        self._index = index
        self._metadatas = [dict(chunk) for chunk in chunks]

    def is_loaded(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Search the vector index for similar chunks.

        Returns a list of dicts with chunk metadata plus _score field,
        compatible with SQLiteRetriever.search() output format.
        """
        import numpy as np

        if not self.is_loaded():
            try:
                self.load()
            except FileNotFoundError:
                return []

        query_vec = self.embedder.encode_query(query)
        scores, indices = self._index.search(
            np.asarray(query_vec).reshape(1, -1), min(limit, self._index.ntotal)
        )

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = dict(self._metadatas[idx])
            row["_score"] = round(float(score) * 10, 3)  # scale to comparable range with SQLite scores
            row["_retrieval_source"] = "vector"
            results.append(row)
        return results
