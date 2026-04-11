"""Chunk manual text into RAG-friendly knowledge units."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from industry_agent.kb.models import KnowledgeChunk, ManualDocument

PIC_MARKER_RE = re.compile(r"\[\[PIC:([^\]]+)\]\]")
SECTION_RE = re.compile(r"(?m)^#\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n+")


def chunk_manual(
    manual: ManualDocument,
    marked_text: str,
    *,
    project_root: Path,
    max_chars: int = 1200,
) -> list[KnowledgeChunk]:
    """Create ordered chunks from a marked manual text."""

    chunks: list[KnowledgeChunk] = []
    for section_index, section_text in enumerate(_split_sections(marked_text)):
        title = _derive_title(section_text)
        for part in _split_to_size(section_text, max_chars=max_chars):
            clean_text = _strip_markers(part)
            if not clean_text:
                continue

            image_ids = _unique_in_order(PIC_MARKER_RE.findall(part))
            chunk_index = len(chunks)
            chunk_id = _make_chunk_id(manual.manual_id, chunk_index, clean_text)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    manual_id=manual.manual_id,
                    product_name=manual.product_name,
                    source_path=str(manual.source_path.relative_to(project_root)),
                    title=title,
                    text=clean_text,
                    image_ids=image_ids,
                    section_index=section_index,
                    chunk_index=chunk_index,
                    char_count=len(clean_text),
                    metadata={
                        "has_image": bool(image_ids),
                    },
                )
            )
    return chunks


def _split_sections(text: str) -> list[str]:
    starts = [match.start() for match in SECTION_RE.finditer(text)]
    if not starts:
        return [text.strip()] if text.strip() else []

    sections: list[str] = []
    if starts[0] > 0:
        preamble = text[: starts[0]].strip()
        if preamble:
            sections.append(preamble)

    starts.append(len(text))
    for current, next_start in zip(starts, starts[1:]):
        section = text[current:next_start].strip()
        if section:
            sections.append(section)
    return sections


def _split_to_size(section_text: str, *, max_chars: int) -> list[str]:
    units = _sentence_units(section_text)
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for unit in units:
        if len(unit) > max_chars:
            if buffer:
                chunks.append("\n".join(buffer).strip())
                buffer = []
                buffer_len = 0
            chunks.extend(_hard_split(unit, max_chars=max_chars))
            continue

        next_len = buffer_len + len(unit) + 1
        if buffer and next_len > max_chars:
            chunks.append("\n".join(buffer).strip())
            buffer = [unit]
            buffer_len = len(unit)
        else:
            buffer.append(unit)
            buffer_len = next_len

    if buffer:
        chunks.append("\n".join(buffer).strip())
    return [chunk for chunk in chunks if chunk.strip()]


def _sentence_units(text: str) -> list[str]:
    units: list[str] = []
    position = 0
    for match in PIC_MARKER_RE.finditer(text):
        units.extend(_plain_sentence_units(text[position : match.start()]))
        units.append(match.group(0))
        position = match.end()
    units.extend(_plain_sentence_units(text[position:]))
    return [unit.strip() for unit in units if unit.strip()]


def _plain_sentence_units(text: str) -> list[str]:
    rough_units = SENTENCE_SPLIT_RE.split(text)
    return [unit.strip() for unit in rough_units if unit.strip()]


def _hard_split(text: str, *, max_chars: int) -> list[str]:
    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars)]


def _derive_title(section_text: str) -> str:
    text = _strip_markers(section_text).lstrip("# ").strip()
    first_line = text.splitlines()[0].strip() if text else "未命名章节"
    first_sentence = re.split(r"[。！？!?；;]", first_line, maxsplit=1)[0].strip()
    title = first_sentence or first_line or "未命名章节"
    return title[:80]


def _strip_markers(text: str) -> str:
    text = PIC_MARKER_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _make_chunk_id(manual_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{manual_id}:{chunk_index}:{text[:120]}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:12]}"
