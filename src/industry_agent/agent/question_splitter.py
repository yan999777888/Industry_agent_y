"""Utilities for splitting complex user questions into sub-questions."""

from __future__ import annotations

import re
from dataclasses import dataclass


_QUOTE_SEGMENT_RE = re.compile(r'"([^"\n]+)"')
_QUESTION_SPLIT_RE = re.compile(r"(?<=[？?])\s*")
_LEADING_FILLER_RE = re.compile(
    r"^(请问|我想咨询一下|我想了解一下|我想问一下|帮我看一下|想问一下|我想知道|麻烦问一下)[，,\s]*"
)
_INTENT_TERMS = (
    "退货", "换货", "退款", "发票", "投诉", "运费", "物流", "维修", "安装",
    "设置", "清洁", "保修", "充电", "指示灯", "尺寸", "表带", "密码", "注意事项",
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
    if len(quoted_segments) >= 2:
        return quoted_segments

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    line_segments: list[str] = []
    for line in normalized.splitlines():
        line = line.strip(" ,，")
        if not line:
            continue
        line_segments.extend(_split_by_question_mark(line))

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


def _detect_intent(text: str) -> str:
    for term in _INTENT_TERMS:
        if term in text:
            return term
    return "general"
