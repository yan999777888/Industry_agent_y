"""BM25 keyword retriever for mixed Chinese-English text.

Uses ``rank_bm25`` with jieba-based tokenization.  Persists the
inverted index to disk (pickle) so startup rebuilds are O(1) after
the first run.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Regex matching either CJK characters or English word-like tokens.
_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]+")
_EN_WORD_RE = re.compile(r"[a-zA-Z]+[\w]*")
_DIGIT_RE = re.compile(r"\d+")


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25.

    - CJK runs are segmented with jieba.
    - English words are lower-cased and split on punctuation.
    - Digits are kept as tokens when they stand alone.
    """
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        # Skip whitespace/punctuation
        m = _CJK_RE.search(text, pos)
        if m and m.start() == pos:
            # Chinese segment — use jieba
            segment = m.group(0)
            for word in jieba.cut(segment):
                w = word.strip()
                if len(w) >= 1:
                    tokens.append(w)
            pos = m.end()
            continue

        m = _EN_WORD_RE.search(text, pos)
        if m and m.start() == pos:
            tokens.append(m.group(0).lower())
            pos = m.end()
            continue

        m = _DIGIT_RE.search(text, pos)
        if m and m.start() == pos:
            tokens.append(m.group(0))
            pos = m.end()
            continue

        pos += 1

    return tokens


class BM25Retriever:
    """BM25 keyword search over chunk corpus.

    Usage::

        retriever = BM25Retriever(chunks)   # build index
        results = retriever.search("如何安装烤箱门", top_k=10)
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]] | None = None,
        *,
        index_path: str | Path | None = None,
        b: float = 0.75,
        k1: float = 1.5,
    ) -> None:
        """Build or load BM25 index.

        Args:
            chunks: List of chunk dicts with ``chunk_id``, ``title``, ``text``.
                    Required when building fresh; may be omitted when loading.
            index_path: Path to persist/load the pickled index.
            b, k1: BM25 parameters (skipped when loading).
        """
        self.b = b
        self.k1 = k1
        self.index_path = Path(index_path) if index_path else None

        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []  # parallel to BM25 internal corpus
        self._chunk_map: dict[str, dict[str, Any]] = {}  # chunk_id -> original

        if chunks:
            self._build(chunks)
        elif self.index_path and self.index_path.exists():
            self._load()

    # ── Public API ──────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search with BM25 and return top-K chunks.

        Returns chunk dicts with ``_bm25_score`` set.
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # Pair (score, chunk_id) and sort descending
        ranked = sorted(
            [
                (float(scores[i]), cid)
                for i, cid in enumerate(self._chunk_ids)
                if scores[i] > 0
            ],
            key=lambda x: (-x[0], x[1]),
        )

        results: list[dict[str, Any]] = []
        for score, cid in ranked[:top_k]:
            chunk = dict(self._chunk_map[cid])  # copy
            chunk["_bm25_score"] = round(score, 4)
            chunk["_retrieval_channels"] = ["bm25"]
            results.append(chunk)

        return results

    @property
    def is_loaded(self) -> bool:
        return self._bm25 is not None

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_ids)

    # ── Index management ────────────────────────────────────────────────

    def _build(self, chunks: list[dict[str, Any]]) -> None:
        """Build BM25 index from chunk list."""
        logger.info("Building BM25 index from %d chunks ...", len(chunks))

        corpus: list[str] = []
        for c in chunks:
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
            corpus.append(f"{title} {text}")

        tokenized_corpus = [_tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus, b=self.b, k1=self.k1)
        self._chunk_ids = [str(c["chunk_id"]) for c in chunks]
        self._chunk_map = {str(c["chunk_id"]): c for c in chunks}

        logger.info("BM25 index built: %d docs", len(self._chunk_ids))

        if self.index_path:
            self._save()

    def _save(self) -> None:
        """Persist index to disk."""
        if self._bm25 is None:
            return
        if not self.index_path:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        data = {
            "chunk_ids": self._chunk_ids,
            "chunk_map": self._chunk_map,
            "b": self.b,
            "k1": self.k1,
            # BM25Okapi internal state: doc_freq, idf, doc_len, avg_doc_len, corpus_size
            "doc_freqs": self._bm25.doc_freqs,
            "idf": self._bm25.idf,
            "doc_len": self._bm25.doc_len,
            "avgdl": self._bm25.avgdl,
            "corpus_size": self._bm25.corpus_size,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        elapsed = time.time() - start
        logger.info("BM25 index saved to %s (%.2fs)", self.index_path, elapsed)

    def _load(self) -> None:
        """Load persisted index from disk."""
        if not self.index_path or not self.index_path.exists():
            logger.warning("BM25 index not found at %s", self.index_path)
            return

        start = time.time()
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self._chunk_ids = data["chunk_ids"]
        self._chunk_map = data["chunk_map"]
        self.b = data["b"]
        self.k1 = data["k1"]

        # Reconstruct BM25Okapi — use a dummy doc to avoid ZeroDivisionError
        # when _calc_idf runs on an empty/repeated-empty corpus.
        self._bm25 = BM25Okapi([["dummy"]], b=self.b, k1=self.k1)
        self._bm25.doc_freqs = data["doc_freqs"]
        self._bm25.idf = data["idf"]
        self._bm25.doc_len = data["doc_len"]
        self._bm25.avgdl = data["avgdl"]
        corpus_size = data.get("corpus_size", len(self._chunk_ids))
        self._bm25.corpus_size = corpus_size
        self._bm25.corpus = [[] for _ in range(corpus_size)]

        elapsed = time.time() - start
        logger.info(
            "BM25 index loaded from %s (%d docs, %.2fs)",
            self.index_path, len(self._chunk_ids), elapsed,
        )
