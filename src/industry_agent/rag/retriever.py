"""SQLite-backed retriever with Chinese keyword extraction.

SQLite FTS5 default tokenizer cannot segment Chinese text, so we use a
keyword-based LIKE search with simple heuristic extraction instead.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from industry_agent.config import settings

# ---------------------------------------------------------------------------
# Lightweight Chinese keyword extractor (no external deps)
# ---------------------------------------------------------------------------

# Common stop words / particles that carry no retrieval value
_STOPWORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "吗", "什么",
    "怎么", "怎样", "如何", "请问", "能", "可以", "吧", "呢", "啊",
    "那", "这个", "那个", "哪", "哪个", "多少", "为什么", "谁",
    "请", "帮", "告诉", "一下", "关于",
}

# Pattern: sequences of CJK chars, or sequences of ASCII word chars
_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+"     # CJK runs
    r"|[A-Za-z][A-Za-z0-9._-]*"                        # ASCII words (strict, no CJK)
    r"|[0-9]+(?:\.[0-9]+)*",                            # numbers / model numbers
)


def extract_keywords(query: str, *, min_len: int = 2) -> list[str]:
    """Extract search keywords from a user query.

    Strategy:
    1. Pull out contiguous CJK runs and ASCII tokens.
    2. Keep CJK runs <= 4 chars as whole terms.
    3. For longer CJK runs, emit overlapping bigrams *after* filtering
       out stop-word bigrams.
    4. Also try to recognize adjacent ASCII+CJK combos like "VR头显"
       by combining neighboring tokens.
    """
    raw_tokens = _TOKEN_RE.findall(query)
    keywords: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        if term and term not in seen and term not in _STOPWORDS and len(term) >= min_len:
            seen.add(term)
            keywords.append(term)

    # Pass 1: combine adjacent ASCII+CJK tokens (e.g. "VR" + "头显" -> "VR头显")
    merged_tokens: list[str] = []
    i = 0
    while i < len(raw_tokens):
        token = raw_tokens[i]
        # ASCII followed by CJK? merge them if combined <= 6 chars
        if (
            re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", token)
            and i + 1 < len(raw_tokens)
            and re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+", raw_tokens[i + 1])
            and len(token) + len(raw_tokens[i + 1]) <= 6
        ):
            merged = token + raw_tokens[i + 1]
            merged_tokens.append(merged)   # "VR头显"
            merged_tokens.append(token)     # "VR"
            merged_tokens.append(raw_tokens[i + 1])  # "头显"
            i += 2
            continue
        merged_tokens.append(token)
        i += 1

    # Pass 2: extract keywords from each token
    for token in merged_tokens:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*|[0-9]+(?:\.[0-9]+)*", token):
            _add(token.upper())
            _add(token)
            continue

        # CJK or mixed token
        if len(token) <= 4:
            _add(token)
        elif len(token) <= 6:
            _add(token)
            # Also emit sub-terms
            for size in (3, 2):
                for j in range(len(token) - size + 1):
                    _add(token[j : j + size])
        else:
            # Long CJK run: emit bigrams, skip stop-word-only bigrams
            for j in range(len(token) - 1):
                bigram = token[j : j + 2]
                _add(bigram)

    return keywords


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class SQLiteRetriever:
    """Keyword-based retriever backed by the SQLite knowledge index."""

    def __init__(self, db_path: Path = settings.processed_dir / "index.sqlite") -> None:
        self.db_path = db_path

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"index not found: {self.db_path}")

        keywords = extract_keywords(query)
        if not keywords:
            # Nothing useful extracted — fall back to raw query as a single LIKE term
            keywords = [query.strip()]

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = self._scored_search(conn, keywords=keywords, limit=limit)
        finally:
            conn.close()

        return [dict(row) for row in rows]

    # ------------------------------------------------------------------

    def _scored_search(
        self,
        conn: sqlite3.Connection,
        *,
        keywords: list[str],
        limit: int,
    ) -> list[sqlite3.Row]:
        """Search chunks by keywords and rank by number of keyword hits.

        Each keyword is matched with LIKE against text, title, and
        product_name.  The more keywords a chunk matches, the higher
        it ranks.  This is a simple but effective strategy for the
        current index size (~4k chunks).
        """
        if not keywords:
            return []

        # Build a scoring expression: +1 for each keyword that matches
        case_parts: list[str] = []
        parameters: list[str] = []
        for kw in keywords:
            like = f"%{kw}%"
            case_parts.append(
                "(CASE WHEN text LIKE ? THEN 1 ELSE 0 END"
                " + CASE WHEN title LIKE ? THEN 2 ELSE 0 END"
                " + CASE WHEN product_name LIKE ? THEN 5 ELSE 0 END)"
            )
            parameters.extend([like, like, like])

        score_expr = " + ".join(case_parts)

        # WHERE: at least one keyword must match
        where_parts = [
            "(text LIKE ? OR title LIKE ? OR product_name LIKE ?)"
            for _ in keywords
        ]
        where_clause = " OR ".join(where_parts)
        where_params: list[str] = []
        for kw in keywords:
            like = f"%{kw}%"
            where_params.extend([like, like, like])

        sql = f"""
            SELECT *, ({score_expr}) AS _score
            FROM chunks
            WHERE {where_clause}
            ORDER BY _score DESC, chunk_index ASC
            LIMIT ?
        """
        all_params = parameters + where_params + [limit]
        return conn.execute(sql, all_params).fetchall()
