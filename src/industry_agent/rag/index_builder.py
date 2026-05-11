"""Compatibility vector-index builder for the current SQLite knowledge index."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from industry_agent.config import settings
from industry_agent.kb.models import KnowledgeChunk
from industry_agent.rag.vector_store import VectorSearchConfig, build_chunk_vector_index


def load_chunks(chunks_path: Path | None = None) -> list[KnowledgeChunk]:
    path = chunks_path or settings.processed_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks: list[KnowledgeChunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            chunks.append(KnowledgeChunk(**record))
    return chunks


def build_vector_index(
    chunks_path: Path | None = None,
    db_path: Path | None = None,
    embedding_model: str | None = None,
    dimensions: int | None = None,
) -> dict[str, Any]:
    target_db = db_path or settings.processed_dir / "index.sqlite"
    if not target_db.exists():
        raise FileNotFoundError(f"SQLite index not found: {target_db}")

    chunks = load_chunks(chunks_path)
    config = VectorSearchConfig(
        embedding_model=(embedding_model or settings.embedding_model).strip(),
        dimensions=dimensions or VectorSearchConfig().dimensions,
        index_path=target_db,
    )
    conn = sqlite3.connect(target_db)
    try:
        summary = build_chunk_vector_index(conn, chunks, config=config)
        conn.commit()
    finally:
        conn.close()
    return {
        **summary,
        "db_path": str(target_db),
        "chunks_path": str(chunks_path or settings.processed_dir / "chunks.jsonl"),
    }


if __name__ == "__main__":  # pragma: no cover
    print(build_vector_index())
