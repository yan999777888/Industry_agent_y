"""Utilities for splitting complex user questions into sub-questions."""

from __future__ import annotations

import re
from dataclasses import dataclass


_QUOTE_SEGMENT_RE = re.compile(r'"([^"\n]+)"')
_QUESTION_SPLIT_RE = re.compile(r"(?<=[？?])\s*")
_CLAUSE_SPLIT_RE = re.compile(r"[，,；;]\s*")
_LEADING_FILLER_RE = re.compile(
    r"^(请问|我想咨询一下|我想了解一下|我想问一下|帮我看一下|想问一下|我想知道|麻烦问一下)[，,\s]*"
)
_INTENT_TERMS = (
    "退货", "换货", "退款", "发票", "投诉", "运费", "物流", "维修", "安装",
    "设置", "清洁", "保修", "充电", "指示灯", "尺寸", "表带", "密码", "注意事项",
)
_QUESTION_CUE_RE = re.compile(
    r"(吗|呢|么|如何|怎么|怎样|多久|多少|哪里|哪儿|是什么|什么意思|代表什么|能不能|可不可以|是否|要不要|会不会|谁来|怎么办)"
)
_SERVICE_HINTS = (
    "退款", "退货", "换货", "退换货", "发票", "抬头", "税号", "物流", "快递", "运费",
    "发货", "补发", "售后", "赔偿", "订单", "差价", "补差价", "尺寸差价", "换大", "换小",
    "尺码", "改地址", "收货地址", "纸质版说明书", "电子版说明书", "电子版",
)
_MANUAL_HINTS = (
    "安装", "充电", "指示灯", "闪烁", "设置", "连接", "故障", "说明书", "清洁",
    "保养", "调节", "组装", "怎么充电", "如何安装", "如何设置", "如何清洁",
)


@dataclass(frozen=True)
class SubQuestion:
    """A normalized sub-question extracted from a complex user query."""

    sub_question_id: str
    text: str
    normalized_text: str
    intent: str
    depends_on_previous: bool = False


def split_complex_question(question: str) -> list[SubQuestion]:
    """Split a possibly multi-part user query into ordered sub-questions."""

    cleaned = _normalize_question_text(question)
    if not cleaned:
        return []

    segments = _extract_segments(cleaned)
    sub_questions: list[SubQuestion] = []
    for index, segment in enumerate(segments, start=1):
        normalized = _normalize_segment(segment)
        if not normalized:
            continue
        sub_questions.append(
            SubQuestion(
                sub_question_id=f"q{index}",
                text=segment.strip(),
                normalized_text=normalized,
                intent=_detect_intent(normalized),
                depends_on_previous=index > 1,
            )
        )

    if not sub_questions:
        normalized = _normalize_segment(cleaned)
        if normalized:
            return [
                SubQuestion(
                    sub_question_id="q1",
                    text=cleaned,
                    normalized_text=normalized,
                    intent=_detect_intent(normalized),
                    depends_on_previous=False,
                )
            ]
    return sub_questions


def _extract_segments(text: str) -> list[str]:
    quoted_segments = [item.strip() for item in _QUOTE_SEGMENT_RE.findall(text) if item.strip()]
    if len(quoted_segments) >= 2 and all(_looks_like_complete_quoted_question(item) for item in quoted_segments):
        return quoted_segments

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    line_segments: list[str] = []
    for line in normalized.splitlines():
        line = line.strip(" ,，")
        if not line:
            continue
        for segment in _split_by_question_mark(line):
            line_segments.extend(_split_by_clause_boundaries(segment))

    return [segment for segment in line_segments if segment.strip()]


def _split_by_question_mark(text: str) -> list[str]:
    parts = _QUESTION_SPLIT_RE.split(text)
    segments: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        buffer += part
        if buffer.endswith("？") or buffer.endswith("?"):
            segments.append(buffer.strip(" ,，"))
            buffer = ""
    if buffer.strip(" ,，"):
        segments.append(buffer.strip(" ,，"))
    return segments


def _normalize_question_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text.replace('"\n"', "\n").strip()


def _normalize_segment(text: str) -> str:
    text = text.strip().strip('"').strip(" ,，")
    text = _LEADING_FILLER_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_complete_quoted_question(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    if len(normalized) < 4:
        return False
    return normalized.endswith(("？", "?"))


def _detect_intent(text: str) -> str:
    for term in _INTENT_TERMS:
        if term in text:
            return term
    return "general"


def _split_by_clause_boundaries(text: str) -> list[str]:
    clauses = [item.strip(" ,，") for item in _CLAUSE_SPLIT_RE.split(text) if item.strip(" ,，")]
    if len(clauses) < 2:
        return [text.strip()]

    segments: list[str] = []
    pending = clauses[0]
    for clause in clauses[1:]:
        if _should_split_clause(pending, clause):
            segments.append(_ensure_question_tail(pending))
            pending = clause
            continue
        pending = f"{pending}，{clause}"
    segments.append(_ensure_question_tail(pending))
    return [segment for segment in segments if segment.strip()]


def _should_split_clause(left: str, right: str) -> bool:
    left_question = _looks_like_question_fragment(left)
    right_question = _looks_like_question_fragment(right)
    return left_question and right_question


def _looks_like_question_fragment(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text.strip())
    if len(normalized) < 4:
        return False
    if normalized.endswith(("？", "?")):
        return True
    return bool(_QUESTION_CUE_RE.search(normalized))


def _ensure_question_tail(text: str) -> str:
    normalized = text.strip(" ,，")
    if not normalized:
        return ""
    if normalized.endswith(("？", "?", "。", ".")):
        return normalized
    if _QUESTION_CUE_RE.search(normalized):
        return normalized + "？"
    return normalized


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)
