"""Rebuild vector index and BM25 from the existing chunks.jsonl.

Use this when you've manually modified chunks.jsonl (e.g., title changes,
text cleanup, image binding) and want to refresh the search indexes WITHOUT
re-parsing the original manuals.

Usage:
    python scripts/rebuild_index_from_jsonl.py

This will:
  1. Sync SQLite chunks table + FTS5 from chunks.jsonl
  2. Rebuild chunk_vectors (embedding from modified text/title)
  3. Rebuild BM25 index (if rank_bm25 is installed)
  4. Keep all image_index and metadata intact
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from industry_agent.config import settings

DATA_DIR = settings.processed_dir
JSONL_PATH = DATA_DIR / "chunks.jsonl"
DB_PATH = DATA_DIR / "index.sqlite"


def step1_sync_sqlite() -> int:
    """Sync chunks + chunks_fts from chunks.jsonl."""
    print("=== Step 1: Sync SQLite + FTS5 ===")

    chunks: list[dict] = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  Loaded {len(chunks)} chunks from JSONL")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT chunk_id, title, text, image_ids, product_name, manual_id FROM chunks")
    db_map = {row["chunk_id"]: dict(row) for row in c.fetchall()}
    print(f"  Loaded {len(db_map)} chunks from SQLite")

    updated = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid not in db_map:
            continue
        changes = {}
        for field in ("title", "text", "image_ids", "product_name", "manual_id"):
            jv = str(chunk.get(field, "") or "")
            dv = str(db_map[cid].get(field, "") or "")
            if jv != dv:
                changes[field] = jv
        if not changes:
            continue

        set_clause = ", ".join(f"{k} = ?" for k in changes)
        vals = list(changes.values()) + [cid]
        c.execute(f"UPDATE chunks SET {set_clause} WHERE chunk_id = ?", vals)

        c.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
        c.execute(
            """INSERT INTO chunks_fts (chunk_id, manual_id, product_name, title, text)
               VALUES (?, ?, ?, ?, ?)""",
            (
                chunk["chunk_id"],
                chunk.get("manual_id", ""),
                chunk.get("product_name", ""),
                chunk.get("title", ""),
                chunk.get("text", ""),
            ),
        )
        updated += 1

    conn.commit()
    print(f"  Updated {updated} chunks")
    return updated


def step2_rebuild_vector_index() -> dict:
    """Rebuild chunk_vectors table from chunks.jsonl."""
    print("\n=== Step 2: Rebuild Vector Index ===")
    from industry_agent.rag.index_builder import build_vector_index

    result = build_vector_index(chunks_path=JSONL_PATH, db_path=DB_PATH)
    print(f"  Vector index: {result.get('status', '?')}, {result.get('chunk_count', 0)} chunks")
    return result


def step3_rebuild_bm25() -> str | None:
    """Rebuild BM25 index from SQLite data."""
    print("\n=== Step 3: Rebuild BM25 Index ===")
    try:
        from industry_agent.rag.bm25_retriever import BM25Retriever
    except ImportError:
        print("  BM25Retriever not available (install rank_bm25)")
        return "skipped (rank_bm25 not installed)"

    bm25_path = DATA_DIR / "bm25_index.pkl"

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, title, text, product_name FROM chunks"
    ).fetchall()
    conn.close()
    chunks = [dict(r) for r in rows]
    print(f"  Loaded {len(chunks)} chunks from DB for BM25")

    bm25 = BM25Retriever(chunks, index_path=bm25_path)
    if bm25.is_loaded:
        print(f"  BM25 index saved to {bm25_path}")
        return str(bm25_path)
    else:
        print("  BM25 index build failed")
        return "failed"


def main() -> int:
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found", file=sys.stderr)
        return 1
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found", file=sys.stderr)
        return 1

    step1_sync_sqlite()
    step2_rebuild_vector_index()
    step3_rebuild_bm25()

    print("\n=== Done ===")
    print("Restart the API service for changes to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
