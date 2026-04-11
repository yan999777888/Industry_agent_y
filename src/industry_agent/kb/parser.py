"""Manual parsing and text normalization."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from industry_agent.kb.models import ManualDocument

PIC_TOKEN = "<PIC>"
PIC_RE = re.compile(r"<PIC>", flags=re.IGNORECASE)
TAIL_IMAGE_LIST_RE = re.compile(
    r',\s*(\[(?:"[^"]+"\s*,\s*)*"[^"]*"\s*\])\s*\]\s*$',
    flags=re.DOTALL,
)


def load_manual(path: Path) -> ManualDocument:
    """Load one manual file.

    Most files are valid JSON/Python literals shaped as [text, image_ids].
    A few contain raw quotes or newlines inside the text field, so we keep a
    fallback parser that recovers the final image-id list from the file tail.
    """

    raw = path.read_text(encoding="utf-8")
    text, image_ids, parse_mode = _parse_structured_manual(raw)
    cleaned_text = normalize_manual_text(text)
    return ManualDocument(
        manual_id=path.stem,
        product_name=_product_name_from_path(path),
        source_path=path,
        text=cleaned_text,
        image_ids=image_ids,
        pic_count=len(PIC_RE.findall(cleaned_text)),
        parse_mode=parse_mode,
    )


def _parse_structured_manual(raw: str) -> tuple[str, list[str], str]:
    for parser_name, parser in (("json", json.loads), ("literal", ast.literal_eval)):
        try:
            data = parser(raw)
        except Exception:
            continue
        text, image_ids = _validate_manual_payload(data)
        return text, image_ids, parser_name

    match = TAIL_IMAGE_LIST_RE.search(raw)
    if not match:
        raise ValueError("cannot parse manual payload or recover tail image list")

    image_ids = [str(item) for item in json.loads(match.group(1))]
    text = raw[: match.start()].strip()
    if text.startswith('["'):
        text = text[2:]
    elif text.startswith("["):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    return _decode_common_escapes(text), image_ids, "tail-recovery"


def _validate_manual_payload(data: object) -> tuple[str, list[str]]:
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("manual payload must be a list shaped as [text, image_ids]")
    text = str(data[0])
    raw_image_ids = data[1]
    if not isinstance(raw_image_ids, list):
        raise ValueError("manual image_ids must be a list")
    return text, [str(item) for item in raw_image_ids]


def _decode_common_escapes(text: str) -> str:
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\/", "/")
    )


def normalize_manual_text(text: str) -> str:
    """Normalize manual text while preserving headings and picture markers."""

    text = _decode_common_escapes(text)
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*<PIC>\s*", f"\n{PIC_TOKEN}\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\n)#\s+", "\n# ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def attach_image_markers(text: str, image_ids: list[str]) -> tuple[str, list[str], int]:
    """Replace each <PIC> with an ordered image marker used during chunking.

    When a manual has more picture placeholders than image ids, the extra
    placeholders are kept as a generic missing marker so they do not pollute
    the image index with synthetic ids.
    """

    parts = PIC_RE.split(text)
    marked_parts: list[str] = []
    attached_ids: list[str] = []
    unmatched_pic_count = 0

    for index, part in enumerate(parts):
        marked_parts.append(part)
        if index >= len(parts) - 1:
            continue
        if index < len(image_ids):
            image_id = image_ids[index]
            attached_ids.append(image_id)
            marked_parts.append(f"\n[[PIC:{image_id}]]\n")
        else:
            unmatched_pic_count += 1
            marked_parts.append("\n[[PIC_MISSING]]\n")

    return "".join(marked_parts), attached_ids, unmatched_pic_count


def _product_name_from_path(path: Path) -> str:
    name = path.stem
    return name[: -len("手册")] if name.endswith("手册") else name
