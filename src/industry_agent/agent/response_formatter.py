"""Normalize answer style for manual QA and customer-service replies."""

from __future__ import annotations

import re


_SECTION_ORDER: tuple[str, ...] = ("结论", "操作/说明", "注意事项", "相关图片")
_SECTION_PATTERN = re.compile(
    r"(结论|操作/说明|操作说明|操作|说明|注意事项|注意|相关图片)\s*[:：]\s*",
)
_MULTI_SECTION_PATTERN = re.compile(
    r"(结论|操作/说明|操作说明|操作|说明|处理步骤|时效/费用|时效费用|补充说明|注意事项|注意|相关图片)\s*[:：]\s*",
)
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
    text = _strip_markdown(answer)
    if "根据现有资料无法回答此问题" in text or "根据现有资料无法准确回答此问题" in text:
        return _format_manual_fallback()

    sections = _parse_sections(_normalize_section_labels(text))
    if not sections:
        sections = _build_sections_from_plain_text(text)

    sections = _fill_missing_sections(sections)
    if compact:
        return _render_manual_compact(sections, image_ids=image_ids)
    return _render_manual_sections(sections, image_ids=image_ids)


def format_customer_service_answer(answer: str) -> str:
    text = _strip_markdown(answer).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(您好|你好)[，,]?\s*", "", text).strip()

    lines: list[str] = []
    blank_pending = False
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
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
    """Merge per-sub-question answers into one natural structured reply."""

    merged = {
        "结论": [],
        "操作/说明": [],
        "处理步骤": [],
        "时效/费用": [],
        "注意事项": [],
        "补充说明": [],
        "相关图片": [],
    }
    sectioned = False
    for _question, answer in sub_answers:
        sections = _parse_multi_answer_sections(answer)
        if sections:
            sectioned = True
            for label, lines in sections.items():
                merged[label].extend(lines)
            continue
        fallback_sections = _build_multi_sections_from_plain_text(answer)
        for label, lines in fallback_sections.items():
            merged[label].extend(lines)

    for label, lines in merged.items():
        if label == "相关图片":
            merged[label] = _dedupe_raw_items(lines)
        else:
            merged[label] = _dedupe_lines(lines)

    if not sectioned and not any(merged.values()):
        plain_blocks = [_normalize_answer_block(answer) for _, answer in sub_answers]
        return "\n\n".join(block for block in plain_blocks if block).strip()
    return _render_multi_question_sections(merged)


def _format_manual_fallback() -> str:
    return (
        "结论：\n"
        "- 根据现有资料无法准确回答此问题。\n\n"
        "操作/说明：\n"
        "- 请补充产品名称、型号、故障现象或上传更清晰的图片后再试。\n\n"
        "注意事项：\n"
        "- 当前回答仅基于知识库中的说明书资料，请以实际产品和原文为准。\n\n"
        "相关图片：\n"
        "- 无"
    )


def _strip_markdown(text: str) -> str:
    cleaned = text.replace("**", "").strip()
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _normalize_answer_block(text: str) -> str:
    cleaned = _strip_markdown(text)
    cleaned = re.sub(r"^回答\s*[:：]\s*", "", cleaned.strip())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    lines = []
    previous = ""
    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        if not line:
            if previous:
                lines.append("")
                previous = ""
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines).strip() or "根据现有资料无法准确回答此问题。"


def _normalize_section_labels(text: str) -> str:
    replacements = {
        "操作说明：": "操作/说明：",
        "操作：": "操作/说明：",
        "说明：": "操作/说明：",
        "注意：": "注意事项：",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _normalize_multi_section_labels(text: str) -> str:
    replacements = {
        "操作说明：": "操作/说明：",
        "操作：": "操作/说明：",
        "说明：": "操作/说明：",
        "注意：": "注意事项：",
        "时效费用：": "时效/费用：",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _parse_sections(text: str) -> dict[str, list[str]]:
    matches = list(_SECTION_PATTERN.finditer(text))
    if not matches:
        return {}

    sections: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        label = _canonical_label(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if label == "相关图片":
            continue
        sections[label] = _clean_section_lines(content)
    return sections


def _parse_multi_answer_sections(text: str) -> dict[str, list[str]]:
    normalized_text = _normalize_multi_section_labels(_normalize_answer_block(text))
    matches = list(_MULTI_SECTION_PATTERN.finditer(normalized_text))
    if not matches:
        return {}

    sections: dict[str, list[str]] = {}
    for index, match in enumerate(matches):
        label = _canonical_multi_label(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        content = normalized_text[start:end].strip()
        if not content:
            continue
        if label == "相关图片":
            sections[label] = _clean_image_section_lines(content)
            continue
        sections[label] = _clean_section_lines(content)
    return sections


def _canonical_label(label: str) -> str:
    if label in {"操作说明", "操作", "说明"}:
        return "操作/说明"
    if label == "注意":
        return "注意事项"
    return label


def _canonical_multi_label(label: str) -> str:
    if label in {"操作说明", "操作", "说明"}:
        return "操作/说明"
    if label == "时效费用":
        return "时效/费用"
    if label == "注意":
        return "注意事项"
    return label


def _clean_section_lines(content: str) -> list[str]:
    lines: list[str] = []
    for raw_line in re.split(r"[\n\r]+", content):
        line = raw_line.strip()
        line = re.sub(r"^[\-\*\d\.\、\s]+", "", line).strip()
        if not line:
            continue
        lines.extend(_split_sentences(line))
    return _dedupe_lines(lines)


def _clean_image_section_lines(content: str) -> list[str]:
    lines: list[str] = []
    for raw_line in re.split(r"[\n\r]+", content):
        line = raw_line.strip()
        line = re.sub(r"^[\-\*\d\.\、\s]+", "", line).strip()
        if not line or line == "无":
            continue
        parts = [part.strip() for part in re.split(r"[、,，\s]+", line) if part.strip()]
        lines.extend(parts)
    return _dedupe_raw_items(lines)


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"[；;]+", "。", text.strip())
    parts = re.split(r"(?<=[。！？])\s*", normalized)
    results = [part.strip() for part in parts if part.strip()]
    return results or ([normalized] if normalized else [])


def _build_sections_from_plain_text(text: str) -> dict[str, list[str]]:
    sentences = _dedupe_lines(_split_sentences(text))
    if not sentences:
        sentences = ["请参考说明书相关章节。"]

    conclusion = [sentences[0]]
    cautions = [sentence for sentence in sentences if any(pattern.search(sentence) for pattern in _CAUTION_PATTERNS)]
    operation = [
        sentence
        for sentence in sentences[1:4]
        if sentence not in cautions
    ] or [
        sentence
        for sentence in sentences[1:]
        if sentence not in cautions
    ] or [sentences[0]]
    if not cautions:
        cautions = ["请以实际产品型号和说明书原文为准。"]
    return {
        "结论": conclusion,
        "操作/说明": _dedupe_lines(operation),
        "注意事项": _dedupe_lines(cautions),
    }


def _fill_missing_sections(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {label: _dedupe_lines(lines) for label, lines in sections.items()}

    conclusion = result.get("结论", [])
    operation = result.get("操作/说明", [])
    cautions = result.get("注意事项", [])

    if not conclusion:
        if operation:
            conclusion = [operation[0]]
        elif cautions:
            conclusion = [cautions[0]]
        else:
            conclusion = ["请参考说明书相关章节。"]

    if not operation:
        operation = [conclusion[0]]

    if not cautions:
        extracted = [
            line
            for line in [*operation, *conclusion]
            if any(pattern.search(line) for pattern in _CAUTION_PATTERNS)
        ]
        cautions = extracted or ["说明书未明确提及额外注意事项，请以实际产品和原文为准。"]

    if operation == conclusion:
        operation = [line for line in operation if line not in conclusion] or operation
    if cautions == conclusion:
        cautions = ["说明书未明确提及额外注意事项，请以实际产品和原文为准。"]

    result["结论"] = _dedupe_lines(conclusion[:2])
    result["操作/说明"] = _dedupe_lines(operation[:3])
    result["注意事项"] = _dedupe_lines(cautions[:3])
    return result


def _render_manual_sections(sections: dict[str, list[str]], *, image_ids: list[str]) -> str:
    blocks: list[str] = []
    for label in ("结论", "操作/说明", "注意事项"):
        lines = _dedupe_lines(sections.get(label, []))
        if not lines:
            continue
        bullet_lines = "\n".join(f"- {line}" for line in lines)
        blocks.append(f"{label}：\n{bullet_lines}")

    image_line = "、".join(image_ids[:4]) if image_ids else "无"
    blocks.append(f"相关图片：\n- {image_line}")
    return "\n\n".join(blocks).strip()


def _render_manual_compact(sections: dict[str, list[str]], *, image_ids: list[str]) -> str:
    lines: list[str] = []
    conclusion = _dedupe_lines(sections.get("结论", []))[:1]
    operation = _dedupe_lines(sections.get("操作/说明", []))[:2]
    cautions = [
        line
        for line in _dedupe_lines(sections.get("注意事项", []))[:2]
        if line not in {
            "请以实际产品型号和说明书原文为准。",
            "说明书未明确提及额外注意事项，请以实际产品和原文为准。",
        }
    ]

    lines.extend(conclusion)
    for line in operation:
        if line not in lines:
            lines.append(line)
    for line in cautions:
        if line not in lines:
            lines.append(line)

    body = " ".join(line.strip() for line in lines if line.strip()).strip()
    if image_ids:
        image_line = "、".join(image_ids[:3])
        if body:
            return f"{body}\n相关图片：{image_line}"
        return f"相关图片：{image_line}"
    return body or "请参考说明书相关章节。"


def _build_multi_sections_from_plain_text(text: str) -> dict[str, list[str]]:
    cleaned = _normalize_answer_block(text)
    sentences = _dedupe_lines(_split_sentences(cleaned))
    if not sentences:
        return {}

    cautions = [sentence for sentence in sentences if any(pattern.search(sentence) for pattern in _CAUTION_PATTERNS)]
    non_cautions = [sentence for sentence in sentences if sentence not in cautions]
    sections: dict[str, list[str]] = {"结论": [], "操作/说明": [], "注意事项": []}
    if non_cautions:
        sections["结论"] = non_cautions[:1]
        sections["操作/说明"] = non_cautions[1:3]
    if not sections["操作/说明"] and len(non_cautions) > 1:
        sections["操作/说明"] = non_cautions[1:]
    sections["注意事项"] = cautions[:3]
    return {label: lines for label, lines in sections.items() if lines}


def _render_multi_question_sections(sections: dict[str, list[str]]) -> str:
    blocks: list[str] = []

    conclusions = _dedupe_lines(sections.get("结论", []))[:3]
    if conclusions:
        blocks.append("结论：\n" + "\n".join(f"- {line}" for line in conclusions))

    operation = _dedupe_lines(sections.get("操作/说明", []))[:4]
    if operation:
        blocks.append("操作/说明：\n" + "\n".join(f"- {line}" for line in operation))

    steps = _dedupe_lines(sections.get("处理步骤", []))[:4]
    if steps:
        blocks.append(
            "处理步骤：\n" + "\n".join(f"{index}. {line}" for index, line in enumerate(steps, start=1))
        )

    timing_fee = _dedupe_lines(sections.get("时效/费用", []))[:4]
    if timing_fee:
        blocks.append("时效/费用：\n" + "\n".join(f"- {line}" for line in timing_fee))

    cautions = _dedupe_lines(sections.get("注意事项", []))[:4]
    if cautions:
        blocks.append("注意事项：\n" + "\n".join(f"- {line}" for line in cautions))

    support = _dedupe_lines(sections.get("补充说明", []))[:3]
    if support:
        blocks.append("补充说明：\n" + "\n".join(f"- {line}" for line in support))

    image_ids = _dedupe_raw_items(sections.get("相关图片", []))[:4]
    if image_ids:
        blocks.append("相关图片：\n- " + "、".join(image_ids))

    return "\n\n".join(blocks).strip()


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if not line.endswith(("。", "！", "？")) and len(line) > 4:
            line = f"{line}。"
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return result


def _dedupe_raw_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw_item in items:
        item = raw_item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
