#!/usr/bin/env python3
"""Rebuild the vector index using DashScope text-embedding-v4.

Requires ``INDUSTRY_AGENT_DASHSCOPE_ENABLED=1`` and ``DASHSCOPE_API_KEY`` set.

Usage::

    INDUSTRY_AGENT_DASHSCOPE_ENABLED=1 DASHSCOPE_API_KEY=sk-xxx \\
        python scripts/rebuild_vectors_dashscope.py

This will:
1. Load all text chunks from ``chunks.jsonl``
2. Batch-embed via DashScope text-embedding-v4 into SQLite chunk_vectors
3. Load image descriptions from ``image_chunks.jsonl``
4. Batch-embed image descriptions into chunk_vectors
5. Report summary
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industry_agent.config import settings
from industry_agent.kb.models import KnowledgeChunk
from industry_agent.rag.vector_store import (
    build_chunk_vector_index,
    encode_vector,
    VectorSearchConfig,
    _chunk_embedding_text,
    _create_embedding_model,
    decode_vector,
)


def load_text_chunks(path: Path) -> list[KnowledgeChunk]:
    """Load text chunks from chunks.jsonl."""
    chunks: list[KnowledgeChunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # Convert metadata dict if present
            if isinstance(record.get("metadata"), str):
                record["metadata"] = json.loads(record["metadata"])
            chunks.append(KnowledgeChunk(**record))
    return chunks


def load_image_chunks(
    path: Path,
) -> list[dict[str, str]]:
    """Load image descriptions from image_chunks.jsonl."""
    records: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records


def main() -> None:
    if not settings.dashscope_enabled:
        print("ERROR: INDUSTRY_AGENT_DASHSCOPE_ENABLED must be set to 1")
        sys.exit(1)
    if not settings.dashscope_api_key:
        print("ERROR: DASHSCOPE_API_KEY must be set")
        sys.exit(1)

    processed = settings.processed_dir
    chunks_path = processed / "chunks.jsonl"
    image_chunks_path = processed / "image_chunks.jsonl"
    db_path = processed / "index.sqlite"

    # ── Step 1: Load chunks ─────────────────────────────────────────
    print("Loading text chunks...")
    text_chunks = load_text_chunks(chunks_path)
    print(f"  {len(text_chunks)} text chunks loaded")

    image_records = load_image_chunks(image_chunks_path)
    print(f"  {len(image_records)} image descriptions loaded")

    # ── Step 2: Rebuild vector table for text chunks ─────────────────
    print("\nBuilding text chunk vectors via text-embedding-v4 ...")
    conn = sqlite3.connect(str(db_path))
    try:
        config = VectorSearchConfig(
            enabled=True,
            embedding_model=settings.dashscope_embedding_model,
            dimensions=settings.dashscope_embedding_dimensions,
            index_path=db_path,
        )
        t0 = time.time()
        result = build_chunk_vector_index(conn, text_chunks, config=config)
        t1 = time.time()
        print(f"  Text chunks: {result['chunk_count']} vectors")
        print(f"  Model: {result['embedding_model']} ({result['dimensions']}d)")
        print(f"  Time: {t1 - t0:.1f}s")

        # ── Step 3: Embed image chunks ───────────────────────────────
        print("\nBuilding image description vectors...")
        model = _create_embedding_model(config)
        dimensions = getattr(model, "dimensions", config.dimensions)
        image_rows: list[tuple[str, str, int, bytes]] = []
        batch: list[str] = []
        batch_ids: list[str] = []

        for rec in image_records:
            img_id = rec["image_id"]
            desc = rec.get("description", "").strip()
            if not desc:
                continue
            chunk_id = f"img:{img_id}"
            batch.append(desc)
            batch_ids.append(chunk_id)
            # Use model's max batch size
            batch_size = getattr(model, 'MAX_BATCH_SIZE', 10)
            if len(batch) >= batch_size:
                vectors = model.embed_batch(batch, text_type="document")
                for cid, vec in zip(batch_ids, vectors):
                    image_rows.append((cid, settings.dashscope_embedding_model, dimensions, encode_vector(vec)))
                batch.clear()
                batch_ids.clear()

        # Remaining
        if batch:
            vectors = model.embed_batch(batch, text_type="document")
            for cid, vec in zip(batch_ids, vectors):
                image_rows.append((cid, settings.dashscope_embedding_model, dimensions, encode_vector(vec)))

        t2 = time.time()
        if image_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding_model, dimensions, vector) VALUES (?, ?, ?, ?)",
                image_rows,
            )
        conn.commit()
        print(f"  Image chunks: {len(image_rows)} vectors")
        print(f"  Time: {t2 - t1:.1f}s")

        # ── Step 4: Summary ──────────────────────────────────────────
        total = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        has_negative = conn.execute(
            "SELECT COUNT(*) FROM chunk_vectors WHERE dimensions != ?",
            (settings.dashscope_embedding_dimensions,),
        ).fetchone()[0]
        print(f"\n=== Summary ===")
        print(f"Total vectors in chunk_vectors: {total}")
        print(f"Non-matching dimensions: {has_negative}")
        print(f"Embedding model: {settings.dashscope_embedding_model}")
        print(f"Dimensions: {settings.dashscope_embedding_dimensions}")
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
