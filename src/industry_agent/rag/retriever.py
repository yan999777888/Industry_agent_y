"""SQLite-backed retriever with Chinese query analysis and reranking.

SQLite FTS5's default tokenizer cannot segment Chinese text, so the first
iteration uses lightweight keyword extraction plus Python-side scoring.  The
retriever intentionally returns ordinary dictionaries to keep compatibility
with the current AgentService.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industry_agent.config import settings

# ---------------------------------------------------------------------------
# Query analysis resources
# ---------------------------------------------------------------------------

_STOPWORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "吗", "什么",
    "怎么", "怎样", "如何", "请问", "能", "可以", "吧", "呢", "啊",
    "那", "这个", "那个", "哪", "哪个", "多少", "为什么", "谁",
    "请", "帮", "告诉", "一下", "关于", "需要", "是否", "哪些",
}

_DOMAIN_PHRASES: tuple[str, ...] = (
    "指示灯", "闪烁", "标识", "充电", "充电器", "电池组", "表带", "尺寸",
    "更换", "安装", "维修", "故障", "清洁", "连接", "设置", "显示",
    "程序", "控制台", "佩戴", "模式", "温度", "延迟", "开机", "关机",
    "按键", "默认密码", "安全注意事项", "注意事项", "售后", "保修",
)

_PRODUCT_ALIASES: dict[str, str] = {
    "vr头显": "VR头显",
    "头显": "VR头显",
    "ps vr": "VR头显",
    "人体工学椅": "人体工学椅",
    "椅子": "人体工学椅",
    "办公椅": "人体工学椅",
    "健身单车": "健身单车",
    "单车": "健身单车",
    "动感单车": "健身单车",
    "健身追踪器": "健身追踪器",
    "追踪器": "健身追踪器",
    "手表": "健身追踪器",
    "腕表": "健身追踪器",
    "表带": "健身追踪器",
    "儿童电动摩托车": "儿童电动摩托车",
    "电动摩托车": "儿童电动摩托车",
    "冰箱": "冰箱",
    "功能键盘": "功能键盘",
    "键盘": "功能键盘",
    "发电机": "发电机",
    "可编程温控器": "可编程温控器",
    "温控器": "可编程温控器",
    "吹风机": "吹风机",
    "摩托艇": "摩托艇",
    "水泵": "水泵",
    "洗碗机": "洗碗机",
    "烤箱": "烤箱",
    "电钻": "电钻",
    "冲击钻": "电钻",
    "起子": "电钻",
    "电动工具": "电钻",
    "相机": "相机",
    "空气净化器": "空气净化器",
    "净化器": "空气净化器",
    "空调": "空调",
    "蒸汽清洁机": "蒸汽清洁机",
    "清洁机": "蒸汽清洁机",
    "蓝牙激光鼠标": "蓝牙激光鼠标",
    "鼠标": "蓝牙激光鼠标",
}

_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+"
    r"|[A-Za-z][A-Za-z0-9._-]*"
    r"|[0-9]+(?:\.[0-9]+)*",
)
_MODEL_RE = re.compile(r"[A-Za-z]{2,}\d+[A-Za-z0-9._-]*")
_FTS_UNSAFE_RE = re.compile(r'["\'():*]+')


@dataclass(frozen=True)
class QueryAnalysis:
    raw_query: str
    keywords: list[str]
    products: list[str]
    models: list[str]


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze product scope, model numbers and useful search keywords."""

    normalized = _normalize(query)
    products = _unique(
        product
        for alias, product in _PRODUCT_ALIASES.items()
        if alias and alias in normalized
    )
    models = _unique(match.group(0).upper() for match in _MODEL_RE.finditer(query))
    keywords = extract_keywords(query)
    for phrase in _DOMAIN_PHRASES:
        if phrase in query:
            keywords.append(phrase)
    keywords.extend(products)
    keywords.extend(models)
    return QueryAnalysis(
        raw_query=query,
        keywords=_unique(keywords),
        products=products,
        models=models,
    )


def extract_keywords(query: str, *, min_len: int = 2) -> list[str]:
    """Extract Chinese and ASCII keywords from a user query."""

    raw_tokens = _TOKEN_RE.findall(query)
    keywords: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if term and term not in _STOPWORDS and len(term) >= min_len:
            keywords.append(term)

    merged_tokens = _merge_ascii_cjk_tokens(raw_tokens)
    for token in merged_tokens:
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*|[0-9]+(?:\.[0-9]+)*", token):
            add(token.upper())
            continue

        if len(token) <= 6:
            add(token)
            for size in (3, 2):
                for index in range(len(token) - size + 1):
                    add(token[index : index + size])
        else:
            for phrase in _DOMAIN_PHRASES:
                if phrase in token:
                    add(phrase)
            for index in range(len(token) - 1):
                add(token[index : index + 2])

    return _unique(keywords)


def _merge_ascii_cjk_tokens(tokens: list[str]) -> list[str]:
    merged_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", token)
            and index + 1 < len(tokens)
            and re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+", tokens[index + 1])
            and len(token) + len(tokens[index + 1]) <= 8
        ):
            merged_tokens.extend([token + tokens[index + 1], token, tokens[index + 1]])
            index += 2
            continue
        merged_tokens.append(token)
        index += 1
    return merged_tokens


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class SQLiteRetriever:
    """Keyword-based retriever backed by the SQLite knowledge index."""

    def __init__(self, db_path: Path = settings.processed_dir / "index.sqlite") -> None:
        self.db_path = db_path

    def search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Return reranked chunks for the query."""

        if not self.db_path.exists():
            raise FileNotFoundError(f"index not found: {self.db_path}")

        analysis = analyze_query(query)
        keywords = analysis.keywords or [query.strip()]
        fetch_limit = max(limit * 12, 50)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            like_candidates = self._candidate_search(
                conn,
                keywords=keywords,
                products=analysis.products,
                limit=fetch_limit,
            )
            fts_candidates = self._fts_candidate_search(
                conn,
                keywords=keywords,
                products=analysis.products,
                limit=fetch_limit,
            )
        finally:
            conn.close()

        candidate_rows = self._merge_candidate_rows(like_candidates, fts_candidates)
        scored = [
            self._score_row(dict(row), analysis)
            for row in candidate_rows
        ]
        scored = [row for row in scored if row["_score"] > 0]
        scored.sort(
            key=lambda item: (
                item["_score"],
                item["_product_match"],
                item["_title_hits"],
                len(_parse_json_list(item.get("image_ids"))),
            ),
            reverse=True,
        )
        return scored[:limit]

    # ------------------------------------------------------------------

    def _candidate_search(
        self,
        conn: sqlite3.Connection,
        *,
        keywords: list[str],
        products: list[str],
        limit: int,
    ) -> list[sqlite3.Row]:
        """Fetch broad candidates, then Python does the precise scoring."""

        where_parts: list[str] = []
        params: list[str] = []
        terms = _unique([*keywords, *products])
        for term in terms:
            like = f"%{term}%"
            where_parts.append("(text LIKE ? OR title LIKE ? OR product_name LIKE ?)")
            params.extend([like, like, like])

        if not where_parts:
            return []

        product_clause = ""
        product_params: list[str] = []
        if products:
            placeholders = ", ".join("?" for _ in products)
            product_clause = f"product_name IN ({placeholders}) AND "
            product_params.extend(products)

        sql = f"""
            SELECT *
            FROM chunks
            WHERE {product_clause}({' OR '.join(where_parts)})
            LIMIT ?
        """
        return conn.execute(sql, [*product_params, *params, limit]).fetchall()

    def _fts_candidate_search(
        self,
        conn: sqlite3.Connection,
        *,
        keywords: list[str],
        products: list[str],
        limit: int,
    ) -> list[sqlite3.Row]:
        """Fetch FTS5 candidates when the virtual table is available."""

        usable_terms = [
            _sanitize_fts_term(term)
            for term in _unique([*products, *keywords])
            if _sanitize_fts_term(term)
        ]
        if not usable_terms:
            return []

        match_terms = usable_terms[:10]
        match_query = " OR ".join(f'"{term}"' for term in match_terms)
        try:
            rows = conn.execute(
                """
                SELECT
                  chunks.*,
                  bm25(chunks_fts) AS fts_rank,
                  1 AS fts_hit
                FROM chunks_fts
                JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        if not products:
            return rows
        return [
            row
            for row in rows
            if str(row["product_name"]) in products
        ]

    def _merge_candidate_rows(
        self,
        like_rows: list[sqlite3.Row],
        fts_rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in like_rows:
            record = dict(row)
            record.setdefault("fts_rank", None)
            record.setdefault("fts_hit", 0)
            merged[str(record["chunk_id"])] = record

        for row in fts_rows:
            record = dict(row)
            chunk_id = str(record["chunk_id"])
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = record
                continue
            existing["fts_hit"] = max(int(existing.get("fts_hit", 0)), int(record.get("fts_hit", 0)))
            if record.get("fts_rank") is not None:
                existing["fts_rank"] = record.get("fts_rank")
        return list(merged.values())

    def _score_row(self, row: dict[str, Any], analysis: QueryAnalysis) -> dict[str, Any]:
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        product = str(row.get("product_name", ""))
        title_norm = _normalize(title)
        text_norm = _normalize(text)
        product_norm = _normalize(product)

        score = 0.0
        title_hits = 0
        text_hits = 0
        product_match = 0
        matched_keywords: list[str] = []

        if analysis.products:
            if product in analysis.products:
                product_match = 1
                score += 20.0
            else:
                score -= 12.0
        elif product == "汇总英文":
            score -= 8.0

        for model in analysis.models:
            model_norm = _normalize(model)
            if model_norm in title_norm:
                score += 12.0
                title_hits += 1
                matched_keywords.append(model)
            elif model_norm in text_norm:
                score += 7.0
                text_hits += 1
                matched_keywords.append(model)

        for keyword in analysis.keywords:
            if keyword in analysis.products:
                continue
            kw = _normalize(keyword)
            if not kw:
                continue
            if kw in product_norm:
                score += 1.0
                product_match = max(product_match, 1)
                matched_keywords.append(keyword)
            if kw in title_norm:
                score += 3.5
                title_hits += 1
                matched_keywords.append(keyword)
                if title_norm.startswith(kw):
                    score += 2.0
                    if keyword in _DOMAIN_PHRASES:
                        score += 8.0
                if keyword in _DOMAIN_PHRASES:
                    score += 4.0
            elif kw in text_norm:
                score += 1.2
                text_hits += 1
                matched_keywords.append(keyword)
                if keyword in _DOMAIN_PHRASES:
                    score += 1.5

        image_ids = _parse_json_list(row.get("image_ids"))
        if image_ids and any(term in analysis.keywords for term in ("指示灯", "表带", "尺寸", "安装", "更换")):
            score += 1.2

        if int(row.get("fts_hit", 0)):
            score += 5.0
            rank_bonus = _fts_rank_bonus(row.get("fts_rank"))
            score += rank_bonus

        if title_hits >= 2:
            score += 3.0
        if title_hits + text_hits >= 4:
            score += 2.0

        row["_score"] = round(score, 3)
        row["_matched_keywords"] = _unique(matched_keywords)
        row["_query_products"] = analysis.products
        row["_query_models"] = analysis.models
        row["_title_hits"] = title_hits
        row["_text_hits"] = text_hits
        row["_product_match"] = product_match
        row["_fts_rank"] = row.get("fts_rank")
        row["_fts_hit"] = int(row.get("fts_hit", 0))
        return row


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _sanitize_fts_term(term: str) -> str:
    cleaned = _FTS_UNSAFE_RE.sub(" ", str(term)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if len(cleaned) >= 2 else ""


def _fts_rank_bonus(value: Any) -> float:
    try:
        rank = float(value)
    except (TypeError, ValueError):
        return 0.0
    rank = abs(rank)
    return max(0.0, 4.0 - min(4.0, math.log1p(rank + 1e-6)))


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
