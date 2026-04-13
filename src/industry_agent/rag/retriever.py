"""Structured retriever for customer-service RAG."""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from industry_agent.config import settings

MODEL_RE = re.compile(r"[A-Za-z]{2,}\d+[A-Za-z0-9-]*")
NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff]+", flags=re.UNICODE)
CHINESE_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

STOP_TERMS = {
    "请问",
    "一下",
    "这个",
    "这些",
    "那个",
    "哪些",
    "怎么",
    "如何",
    "什么",
    "代表",
    "含义",
    "时候",
    "可以",
    "我的",
    "我们",
    "你们",
    "是否",
    "还有",
    "一下子",
}

DOMAIN_TERMS = [
    "指示灯",
    "闪烁",
    "标识",
    "充电",
    "充电器",
    "电池组",
    "表带",
    "尺寸",
    "更换",
    "安装",
    "维修",
    "故障",
    "清洁",
    "连接",
    "设置",
    "显示",
    "程序",
    "控制台",
    "佩戴",
    "模式",
    "温度",
    "延迟",
    "开机",
    "关机",
    "按键",
    "要求",
]


@dataclass
class SearchQuery:
    raw_query: str
    normalized_query: str
    product_terms: list[str]
    model_terms: list[str]
    keywords: list[str]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    chunk_id: str
    manual_id: str
    product_name: str
    title: str
    text: str
    image_ids: list[str]
    source_path: str
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResponse:
    query: SearchQuery
    results: list[SearchResult]

    def to_record(self) -> dict[str, Any]:
        return {
            "query": self.query.to_record(),
            "results": [item.to_record() for item in self.results],
        }


class SQLiteRetriever:
    """Hybrid retriever backed by the generated SQLite index."""

    def __init__(self, db_path: Path = settings.processed_dir / "index.sqlite") -> None:
        self.db_path = db_path
        self._chunks = self._load_chunks()
        self._product_aliases = self._build_product_aliases()

    def analyze_query(self, query: str) -> SearchQuery:
        normalized_query = _normalize_text(query)
        product_terms = sorted(
            {
                product_name
                for alias, product_name in self._product_aliases.items()
                if alias and alias in normalized_query
            },
            key=len,
            reverse=True,
        )
        model_terms = _unique_in_order(match.group(0).upper() for match in MODEL_RE.finditer(query))
        keywords = _extract_keywords(normalized_query, product_terms, model_terms)
        return SearchQuery(
            raw_query=query,
            normalized_query=normalized_query,
            product_terms=product_terms,
            model_terms=model_terms,
            keywords=keywords,
        )

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        response = self.retrieve(query, limit=limit)
        return [item.to_record() for item in response.results]

    def retrieve(self, query: str, *, limit: int = 5) -> SearchResponse:
        if not self.db_path.exists():
            raise FileNotFoundError(f"index not found: {self.db_path}")

        search_query = self.analyze_query(query)
        candidate_chunks = self._prefilter_chunks(search_query)
        scored_results: list[SearchResult] = []
        for chunk in candidate_chunks:
            score, score_breakdown = self._score_chunk(search_query, chunk)
            if score <= 0:
                continue
            scored_results.append(
                SearchResult(
                    chunk_id=str(chunk["chunk_id"]),
                    manual_id=str(chunk["manual_id"]),
                    product_name=str(chunk["product_name"]),
                    title=str(chunk["title"]),
                    text=str(chunk["text"]),
                    image_ids=_ensure_list(chunk["image_ids"]),
                    source_path=str(chunk["source_path"]),
                    score=round(score, 3),
                    score_breakdown=score_breakdown,
                )
            )

        scored_results.sort(
            key=lambda item: (
                item.score,
                len(item.image_ids),
                -len(item.title),
            ),
            reverse=True,
        )
        return SearchResponse(query=search_query, results=scored_results[:limit])

    def _load_chunks(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM chunks").fetchall()
        finally:
            conn.close()

        chunks: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["image_ids"] = _ensure_list(item["image_ids"])
            item["metadata"] = _ensure_dict(item["metadata"])
            item["_normalized_title"] = _normalize_text(str(item["title"]))
            item["_normalized_text"] = _normalize_text(str(item["text"]))
            item["_normalized_product"] = _normalize_text(str(item["product_name"]))
            chunks.append(item)
        return chunks

    def _build_product_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for chunk in self._chunks:
            product_name = str(chunk["product_name"])
            normalized_product = _normalize_text(product_name)
            if normalized_product:
                aliases[normalized_product] = product_name
            for alias in _expand_product_aliases(product_name):
                aliases[_normalize_text(alias)] = product_name
        return aliases

    def _prefilter_chunks(self, search_query: SearchQuery) -> list[dict[str, Any]]:
        product_scope = set(search_query.product_terms)
        if product_scope:
            scoped = [chunk for chunk in self._chunks if chunk["product_name"] in product_scope]
            if scoped:
                return scoped

        primary_terms = search_query.model_terms + search_query.keywords
        if not primary_terms:
            return [chunk for chunk in self._chunks if chunk["manual_id"] != "汇总英文手册"]

        candidates: list[dict[str, Any]] = []
        for chunk in self._chunks:
            haystack = f"{chunk['_normalized_title']} {chunk['_normalized_text']}"
            if any(_normalize_text(term) in haystack for term in primary_terms if term):
                candidates.append(chunk)
        return candidates or [chunk for chunk in self._chunks if chunk["manual_id"] != "汇总英文手册"]

    def _score_chunk(
        self,
        search_query: SearchQuery,
        chunk: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        normalized_title = chunk["_normalized_title"]
        normalized_text = chunk["_normalized_text"]
        normalized_product = chunk["_normalized_product"]

        score = 0.0
        breakdown: dict[str, float] = defaultdict(float)

        if chunk["manual_id"] == "汇总英文手册" and not search_query.product_terms:
            breakdown["summary_manual_penalty"] -= 4.0
            score -= 4.0

        if search_query.normalized_query and search_query.normalized_query in normalized_title:
            breakdown["full_query_in_title"] += 8.0
            score += 8.0
        elif search_query.normalized_query and search_query.normalized_query in normalized_text:
            breakdown["full_query_in_text"] += 5.0
            score += 5.0

        if search_query.product_terms:
            if chunk["product_name"] in search_query.product_terms:
                breakdown["product_match"] += 12.0
                score += 12.0
            elif any(_normalize_text(term) in normalized_product for term in search_query.product_terms):
                breakdown["product_partial_match"] += 7.0
                score += 7.0

        for model_term in search_query.model_terms:
            normalized_model = _normalize_text(model_term)
            if normalized_model in normalized_title:
                breakdown["model_title_match"] += 10.0
                score += 10.0
                if normalized_title.startswith(normalized_model):
                    breakdown["model_title_prefix_bonus"] += 3.0
                    score += 3.0
            elif normalized_model in normalized_text:
                breakdown["model_text_match"] += 6.0
                score += 6.0

        title_keyword_hits = 0
        text_keyword_hits = 0
        for keyword in search_query.keywords:
            normalized_keyword = _normalize_text(keyword)
            if not normalized_keyword or normalized_keyword in STOP_TERMS:
                continue
            if normalized_keyword in normalized_title:
                title_keyword_hits += 1
                breakdown["title_keyword_hits"] += 3.0
                score += 3.0
                if normalized_title.startswith(normalized_keyword):
                    breakdown["title_prefix_keyword_bonus"] += 2.0
                    score += 2.0
            elif normalized_keyword in normalized_text:
                text_keyword_hits += 1
                breakdown["text_keyword_hits"] += 1.5
                score += 1.5

        for intent_term in ("安装", "更换", "表带", "尺寸", "指示灯"):
            if intent_term in search_query.keywords:
                normalized_intent = _normalize_text(intent_term)
                if normalized_title.startswith(normalized_intent):
                    breakdown["intent_prefix_bonus"] += 4.0
                    score += 4.0
                elif f" {normalized_intent}" in f" {normalized_title}":
                    breakdown["intent_title_bonus"] += 2.0
                    score += 2.0

        if chunk["image_ids"]:
            if any(term in search_query.keywords for term in {"指示灯", "表带", "尺寸", "安装"}):
                breakdown["image_bonus"] += 1.2
                score += 1.2
            else:
                breakdown["image_bonus"] += 0.4
                score += 0.4

        if title_keyword_hits >= 2:
            breakdown["title_cohesion_bonus"] += 2.0
            score += 2.0
        if title_keyword_hits + text_keyword_hits >= 3:
            breakdown["coverage_bonus"] += 2.5
            score += 2.5

        return score, dict(breakdown)


def _extract_keywords(normalized_query: str, product_terms: list[str], model_terms: list[str]) -> list[str]:
    keywords: list[str] = []
    keywords.extend(product_terms)
    keywords.extend(model_terms)

    for term in DOMAIN_TERMS:
        if term in normalized_query:
            keywords.append(term)

    segments = CHINESE_SEGMENT_RE.findall(normalized_query)
    for segment in segments:
        if segment in product_terms or segment in STOP_TERMS:
            continue
        if 2 <= len(segment) <= 8:
            keywords.append(segment)
            continue
        for size in (4, 3, 2):
            for index in range(0, len(segment) - size + 1):
                piece = segment[index : index + size]
                if piece in STOP_TERMS:
                    continue
                if piece in DOMAIN_TERMS:
                    keywords.append(piece)

    return _unique_in_order(keyword for keyword in keywords if keyword and keyword not in STOP_TERMS)


def _expand_product_aliases(product_name: str) -> list[str]:
    aliases = [product_name]
    if product_name.endswith("器"):
        aliases.append(product_name[:-1])
    if product_name.endswith("机"):
        aliases.append(product_name[:-1])
    if product_name.startswith("蓝牙"):
        aliases.append(product_name[2:])
    return _unique_in_order(alias for alias in aliases if alias)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _unique_in_order(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered
