"""Normalize answer style for manual QA and customer-service replies.

Simplified: no forced section structure.  Preserve natural LLM output.
"""

from __future__ import annotations

import re

_CAUTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"注意",
        r"请勿",
        r"不要",
        r"避免",
        r"必须",
        r"确保",
        r"建议",
        r"以.*为准",
    )
)


def format_manual_answer(answer: str, *, image_ids: list[str], compact: bool = False) -> str:
    """Clean up a manual-QA answer but do NOT force section structure.

    - If the answer contains natural section headers (结论： etc.), keep them.
    - If the answer is plain prose, return it as-is after basic cleanup.
    """
    text = _strip_markdown(answer)

    # Fallback — pass through as-is
    if "根据现有资料无法回答此问题" in text or "根据现有资料无法准确回答此问题" in text:
        return _format_manual_fallback()

    # Detect whether the LLM naturally used section headers
    has_sections = bool(re.search(r"(结论|操作/说明|注意事项|操作说明|操作|说明|注意)\s*[:：]\s*", text))

    if has_sections:
        # Preserve the structure the LLM chose
        return _clean_sectioned_answer(text)
    else:
        # Plain natural prose — just basic clean up, no section forcing
        return _clean_plain_answer(text)


def format_customer_service_answer(answer: str) -> str:
    text = _strip_markdown(answer).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(您好|你好|Hello)[，,!！]?\s*", "", text).strip()

    lines: list[str] = []
    blank_pending = False
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        # Skip structured headers
        if line in ("结论：", "结论:", "目标：", "目标:"):
            continue
        # Strip "- " bullet markers
        if line.startswith("- ") or line.startswith("- "):
            line = line[2:].strip()
        # Strip numbered markers like "1. ", "2、", "1) "
        line = re.sub(r"^\d+[\.、)）]\s*", "", line).strip()
        if not line:
            if lines:
                blank_pending = True
            continue
        if blank_pending and lines:
            lines.append("")
        blank_pending = False
        lines.append(line)

    formatted = "\n".join(lines).strip()
    if formatted and not formatted.endswith(("。", "！", "？")):
        formatted += "。"
    return formatted


def format_multi_question_answer(sub_answers: list[tuple[str, str]]) -> str:
    """Merge per-sub-question answers into one natural reply."""
    texts = [answer.strip() for _, answer in sub_answers if answer.strip()]
    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_manual_fallback() -> str:
    return "根据现有资料无法准确回答此问题。请补充产品名称、型号、故障现象或图片后再试。"


def _strip_markdown(text: str) -> str:
    cleaned = text.replace("**", "").strip()
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _clean_sectioned_answer(text: str) -> str:
    """Light cleanup on a section-structured answer — don't rearrange sections."""
    # Strip empty bullet lines
    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Remove "无" lines in "相关图片：" sections
        if stripped == "- 无":
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _clean_plain_answer(text: str) -> str:
    """Basic cleanup for plain-text answers — no section forcing."""
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Remove trailing "相关图片：" placeholder
    cleaned = re.sub(r"\n*相关图片[：:]\s*无?\s*$", "", cleaned)
    return cleaned or text
