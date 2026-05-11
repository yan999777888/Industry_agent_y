"""Persistence helpers for processed knowledge-base artifacts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from industry_agent.kb.models import ImageRecord, KnowledgeChunk
from industry_agent.rag.vector_store import build_chunk_vector_index


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sqlite_index(
    db_path: Path,
    chunks: list[KnowledgeChunk],
    images: list[ImageRecord],
    manual_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a lightweight SQLite index for the first RAG iteration."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        _create_base_tables(conn)
        _insert_records(conn, chunks, images, manual_records)
        fts_available = _create_fts_index(conn, chunks)
        vector_summary = build_chunk_vector_index(conn, chunks)
        conn.commit()
    finally:
        conn.close()

    return {
        "db_path": str(db_path),
        "fts5_available": fts_available,
        "chunk_count": len(chunks),
        "image_count": len(images),
        "manual_count": len(manual_records),
        "vector": vector_summary,
    }


def _create_base_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE manuals (
          manual_id TEXT PRIMARY KEY,
          product_name TEXT NOT NULL,
          source_path TEXT NOT NULL,
          char_count INTEGER NOT NULL,
          pic_count INTEGER NOT NULL,
          image_count INTEGER NOT NULL,
          chunk_count INTEGER NOT NULL,
          parse_mode TEXT NOT NULL
        );

        CREATE TABLE chunks (
          chunk_id TEXT PRIMARY KEY,
          manual_id TEXT NOT NULL,
          product_name TEXT NOT NULL,
          source_path TEXT NOT NULL,
          title TEXT NOT NULL,
          text TEXT NOT NULL,
          image_ids TEXT NOT NULL,
          section_index INTEGER NOT NULL,
          chunk_index INTEGER NOT NULL,
          char_count INTEGER NOT NULL,
          metadata TEXT NOT NULL
        );

        CREATE TABLE images (
          image_id TEXT PRIMARY KEY,
          file_name TEXT,
          path TEXT,
          exists_on_disk INTEGER NOT NULL,
          referenced_by TEXT NOT NULL,
          chunk_ids TEXT NOT NULL
        );
        """
    )


def _insert_records(
    conn: sqlite3.Connection,
    chunks: list[KnowledgeChunk],
    images: list[ImageRecord],
    manual_records: list[dict[str, Any]],
) -> None:
    conn.executemany(
        """
        INSERT INTO manuals (
          manual_id, product_name, source_path, char_count, pic_count,
          image_count, chunk_count, parse_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                record["manual_id"],
                record["product_name"],
                record["source_path"],
                record["char_count"],
                record["pic_count"],
                record["image_count"],
                record["chunk_count"],
                record["parse_mode"],
            )
            for record in manual_records
        ],
    )
    conn.executemany(
        """
        INSERT INTO chunks (
          chunk_id, manual_id, product_name, source_path, title, text,
          image_ids, section_index, chunk_index, char_count, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                chunk.chunk_id,
                chunk.manual_id,
                chunk.product_name,
                chunk.source_path,
                chunk.title,
                chunk.text,
                json.dumps(chunk.image_ids, ensure_ascii=False),
                chunk.section_index,
                chunk.chunk_index,
                chunk.char_count,
                json.dumps(chunk.metadata, ensure_ascii=False),
            )
            for chunk in chunks
        ],
    )
    conn.executemany(
        """
        INSERT INTO images (
          image_id, file_name, path, exists_on_disk, referenced_by, chunk_ids
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                image.image_id,
                image.file_name,
                image.path,
                1 if image.exists else 0,
                json.dumps(image.referenced_by, ensure_ascii=False),
                json.dumps(image.chunk_ids, ensure_ascii=False),
            )
            for image in images
        ],
    )


def _create_fts_index(conn: sqlite3.Connection, chunks: list[KnowledgeChunk]) -> bool:
    """Create FTS5 index with character-level tokenizer for Chinese text.

    SQLite FTS5 default tokenizer splits on whitespace/punctuation — useless
    for Chinese.  ``tokenize='unicode61 tokenchars "。，、；：？！""''《》（）【】"'``
    treats each Unicode codepoint as a token, which makes substring-style
    MATCH queries work for CJK characters.
    """
    create_sql_candidates = (
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
          chunk_id UNINDEXED,
          manual_id UNINDEXED,
          product_name,
          title,
          text,
          tokenize='unicode61 categories "L* N* Co"'
        )
        """,
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
          chunk_id UNINDEXED,
          manual_id UNINDEXED,
          product_name,
          title,
          text,
          tokenize='unicode61'
        )
        """,
    )
    for create_sql in create_sql_candidates:
        try:
            conn.execute(create_sql)
            break
        except sqlite3.OperationalError:
            continue
    else:
        return False

    conn.executemany(
        """
        INSERT INTO chunks_fts (chunk_id, manual_id, product_name, title, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                chunk.chunk_id,
                chunk.manual_id,
                chunk.product_name,
                chunk.title,
                chunk.text,
            )
            for chunk in chunks
        ],
    )
    return True
