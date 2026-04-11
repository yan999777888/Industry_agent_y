"""Minimal SQLite-backed retriever placeholder."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from industry_agent.config import settings


class SQLiteRetriever:
    """Small retriever for smoke-testing the generated index."""

    def __init__(self, db_path: Path = settings.processed_dir / "index.sqlite") -> None:
        self.db_path = db_path

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"index not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows: list[sqlite3.Row] = []
            try:
                rows = conn.execute(
                    """
                    SELECT c.*
                    FROM chunks_fts f
                    JOIN chunks c ON c.chunk_id = f.chunk_id
                    WHERE chunks_fts MATCH ?
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

            if not rows:
                rows = self._fallback_like_search(conn, query=query, limit=limit)
        finally:
            conn.close()

        return [dict(row) for row in rows]

    def _fallback_like_search(
        self,
        conn: sqlite3.Connection,
        *,
        query: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        terms = [query.strip()]
        terms.extend(part.strip() for part in query.split() if part.strip())
        terms = _unique_in_order([term for term in terms if term])
        where_clause = " OR ".join("(text LIKE ? OR title LIKE ? OR product_name LIKE ?)" for _ in terms)
        parameters: list[Any] = []
        for term in terms:
            like_term = f"%{term}%"
            parameters.extend([like_term, like_term, like_term])
        parameters.append(limit)
        return conn.execute(
            f"""
            SELECT *
            FROM chunks
            WHERE {where_clause}
            LIMIT ?
            """,
            parameters,
        ).fetchall()


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
