"""Sync title changes from chunks.jsonl into index.sqlite.

Usage:
    python scripts/sync_titles.py

Reads the current chunks.jsonl and updates the SQLite database
(chunks + chunks_fts tables) to match.  Only touches rows whose
title actually changed — does NOT rebuild vectors / BM25 index.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb"
JSONL_PATH = DATA_DIR / "chunks.jsonl"
SQLITE_PATH = DATA_DIR / "index.sqlite"


def main() -> int:
    if not JSONL_PATH.exists():
        print(f"ERROR: {JSONL_PATH} not found", file=sys.stderr)
        return 1
    if not SQLITE_PATH.exists():
        print(f"ERROR: {SQLITE_PATH} not found", file=sys.stderr)
        return 1

    # Load jsonl chunks
    jsonl_chunks: dict[str, dict] = {}
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            jsonl_chunks[d["chunk_id"]] = d

    print(f"Loaded {len(jsonl_chunks)} chunks from JSONL")

    # Connect to SQLite
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get current db chunks
    c.execute("SELECT chunk_id, title, text, image_ids FROM chunks")
    db_rows = {row["chunk_id"]: dict(row) for row in c.fetchall()}
    print(f"Loaded {len(db_rows)} chunks from SQLite")

    # Find changed rows
    updated = 0
    for cid, jc in jsonl_chunks.items():
        db_row = db_rows.get(cid)
        if db_row is None:
            continue
        changes = {}
        for field in ("title", "text", "image_ids", "product_name", "manual_id"):
            jv = str(jc.get(field, "") or "")
            dv = str(db_row.get(field, "") or "")
            if jv != dv:
                changes[field] = jv
        if not changes:
            continue

        # Update chunks table
        set_clause = ", ".join(f"{k} = ?" for k in changes)
        vals = list(changes.values()) + [cid]
        c.execute(f"UPDATE chunks SET {set_clause} WHERE chunk_id = ?", vals)

        # Update chunks_fts (must DELETE + INSERT for FTS5)
        c.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
        c.execute(
            """INSERT INTO chunks_fts (chunk_id, manual_id, product_name, title, text)
               VALUES (?, ?, ?, ?, ?)""",
            (
                jc["chunk_id"],
                jc.get("manual_id", ""),
                jc.get("product_name", ""),
                jc.get("title", ""),
                jc.get("text", ""),
            ),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"\nDone. {updated} chunks updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
