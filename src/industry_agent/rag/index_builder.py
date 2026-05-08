"""Vector index builder — reads chunks.jsonl and builds a FAISS index.

Usage:
    python -m industry_agent.rag.index_builder
    # or programmatically:
    from industry_agent.rag.index_builder import build_vector_index
    build_vector_index()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from industry_agent.config import settings
from industry_agent.rag.embedding import EmbeddingManager
from industry_agent.rag.vector_store import VectorStoreRetriever, VectorStoreConfig


def load_chunks(chunks_path: Path | None = None) -> list[dict[str, Any]]:
    """Load chunks from the JSONL file produced by the KB builder."""
    path = chunks_path or settings.processed_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_vector_index(
    chunks_path: Path | None = None,
    index_path: Path | None = None,
    embedding_model: str | None = None,
    batch_size: int = 64,
) -> VectorStoreRetriever:
    """Build and save a FAISS vector index from chunks.jsonl.

    Args:
        chunks_path: Path to chunks.jsonl (default: data/processed/kb/chunks.jsonl)
        index_path: Where to save the FAISS index (default: data/processed/kb/vector.index)
        embedding_model: Embedding model name (default: from settings)
        batch_size: Encoding batch size

    Returns:
        The populated VectorStoreRetriever instance.
    """
    model_name = embedding_model or settings.embedding_model
    save_path = index_path or settings.vector_index_path

    print(f"Loading chunks from {chunks_path or settings.processed_dir / 'chunks.jsonl'}...")
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Initializing embedding model: {model_name}")
    embedder = EmbeddingManager(model_name)

    config = VectorStoreConfig(index_path=save_path, embedding_model=model_name)
    store = VectorStoreRetriever(config=config, embedding_manager=embedder)

    print("Building FAISS index...")
    store.build(chunks, batch_size=batch_size)

    print(f"Saving index to {save_path}...")
    store.save()

    print(f"Done. Index contains {store.size} vectors (dim={embedder.dimension}).")
    return store


if __name__ == "__main__":
    build_vector_index()
