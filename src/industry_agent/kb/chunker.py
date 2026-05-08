"""Chunk manual text into RAG-friendly knowledge units."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from industry_agent.kb.models import KnowledgeChunk, ManualDocument

PIC_MARKER_RE = re.compile(r"\[\[PIC:([^\]]+)\]\]")
PIC_MISSING_RE = re.compile(r"\[\[PIC_MISSING\]\]")
SECTION_RE = re.compile(r"(?m)^#\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n+")
ENGLISH_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "boat": (
        "boat", "sailing", "anchor light", "bimini top", "bilge pump", "engine compartment",
        "watercraft", "jet thrust", "stern", "bow", "hull", "battery switch", "wet storage",
    ),
    "ereader": (
        "e reader", "e-reader", "e-book reader", "ebook", "voice recording", "photo viewer",
        "photo mode", "browser history", "main menu",
    ),
    "vacuum": (
        "vacuum", "home base", "full bin", "side brush", "caster wheel", "dual-mode virtual wall",
        "dust bin", "room ba", "cleaning head module", "roomba", "dirt detect",
    ),
    "motherboard": (
        "motherboard", "tpm connector", "pci express", "cpu", "system memory", "raid",
        "rear panel connectors", "onboard led", "bios", "sata", "usb 3.1", "intel lan",
        "serial port", "apm configuration", "erp ready", "pxe option", "configuration options",
    ),
    "airfryer": (
        "air fryer", "airfryer", "nutriu", "favorite recipe", "remote cooking", "basket",
        "hot air", "rapid air", "keep warm", "smart chef", "air fry",
    ),
    "pressure_cooker": (
        "pressure cooker", "quick release", "float valve", "steam release", "anti-block shield",
        "condensation collector", "sealing ring",
    ),
    "microwave": (
        "microwave", "over-the-range", "auto defrost", "grease filter", "charcoal filter",
        "oven light", "light timer",
    ),
    "snowmobile": (
        "snowmobile", "throttle cable", "v-belt", "spark plug", "brake lever", "ski",
        "vk540", "suspension", "spring preload", "fresh snow", "fuel tank", "rider",
    ),
    "landline": (
        "landline", "base station", "handset", "answering machine", "phonebook",
    ),
    "camera": (
        "camera", "lens", "shutter", "viewfinder", "autofocus", "battery grip", "eos",
        "mode dial", "cf card", "image playback", "battery charger", "lcd panel",
        "white balance", "aperture", "exposure", "metering", "flash", "iso speed",
        "picture style", "shooting", "image-recording", "date/time battery",
    ),
    "lawn_mower": (
        "lawn mower", "mower deck", "blade-control switch", "height-of-cut", "grass deflector",
        "spark-plug", "parking brake", "cutting blade", "anti-scalp", "pto",
    ),
    "coffee_machine": (
        "coffee", "espresso", "lungo", "capsule", "descaling", "water tank", "coffee preparation",
        "drip tray", "milk frother",
    ),
    "fax": (
        "fax", "telephone line cord", "phone line", "mfc-", "ink cartridge", "document feeder",
        "scanner glass", "brother", "telephone wall jack",
    ),
    "toothbrush": (
        "toothbrush", "brush head", "brushing", "pressure sensor", "brush pacer",
        "senseiq", "gum", "bristles", "toothpaste",
    ),
    "grill": (
        "grill", "grilling", "burner", "cooking surface", "grease tray", "spider alert",
        "propane", "bristle brush",
    ),
    "earphone": (
        "earphone", "earphones", "earbud", "earbuds", "charging case", "bluetooth",
        "noise canceling", "pairing", "wearing the headset",
    ),
    "television": (
        "television", "tv", "hdmi", "remote control", "channel", "picture mode",
        "screen", "antenna", "audio output",
    ),
    "washing_machine": (
        "washer", "washing machine", "washtub", "wash timer", "spin timer", "drain filter",
        "overflow filter", "drain hose", "rinse", "cycle selector", "water supply",
    ),
}
PROCEDURE_HINTS: tuple[str, ...] = (
    "install", "remove", "replace", "clean", "set ", "setting", "connect", "adjust",
    "operate", "use ", "using", "charge", "recharge", "pair", "assemble", "mount",
    "detach", "insert", "plug", "unplug", "press", "select", "turn on", "turn off",
    "安装", "拆卸", "拆下", "更换", "清洁", "设置", "连接", "调节", "操作",
    "使用", "充电", "佩戴", "扣紧", "打开", "关闭", "插入", "选择",
)
SAFETY_HINTS: tuple[str, ...] = (
    "warning", "caution", "danger", "safety", "safeguards", "risk of", "do not",
    "never", "avoid", "hazard", "注意", "警告", "危险", "小心", "安全",
)
TROUBLESHOOTING_HINTS: tuple[str, ...] = (
    "troubleshooting", "problem", "error", "fault", "fails", "not working", "does not",
    "cannot", "can't", "flashing", "blinking", "indicator", "beep", "alarm",
    "故障", "错误", "报错", "无法", "不能", "不工作", "闪烁", "指示灯", "蜂鸣",
)
PARTS_HINTS: tuple[str, ...] = (
    "nomenclature", "overview", "parts", "included", "accessories", "package contents",
    "item check list", "components", "部件", "零件", "清单", "包装", "组成", "配件",
)
SPECIFICATION_HINTS: tuple[str, ...] = (
    "specifications", "technical data", "dimensions", "weight", "battery life", "capacity",
    "temperature range", "default", "factory setting", "rating", "model", "规格",
    "参数", "尺寸", "重量", "容量", "默认", "出厂", "型号", "密码",
)


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
        if _is_toc_like(section_text):
            continue
        title = _derive_title(section_text)
        for part in _split_to_size(section_text, max_chars=max_chars):
            clean_text = _strip_markers(part)
            if not clean_text:
                continue

            image_ids = _unique_in_order(PIC_MARKER_RE.findall(part))
            if _is_low_value_fragment(clean_text, image_ids=image_ids):
                continue
            chunk_index = len(chunks)
            chunk_id = _make_chunk_id(manual.manual_id, chunk_index, clean_text)
            metadata = _build_chunk_metadata(
                manual=manual,
                title=title,
                text=clean_text,
                image_ids=image_ids,
            )
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
                    metadata=metadata,
                )
            )
    if manual.product_name == "汇总英文":
        _backfill_english_domain_labels(chunks)
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
    text = PIC_MISSING_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_chunk_metadata(
    *,
    manual: ManualDocument,
    title: str,
    text: str,
    image_ids: list[str],
) -> dict[str, object]:
    domain_label = _detect_english_domain(f"{title}\n{text}") if manual.product_name == "汇总英文" else ""
    has_toc_noise = _is_toc_like(text)
    has_ocr_noise = bool(re.search(r"\\u[0-9a-fA-F]{4}|\\(?:mathsf|mathrm|pmb)|[a-z]{18,}", text))
    semantic_type = _detect_semantic_type(title=title, text=text, is_toc=has_toc_noise)
    clean_score = 1.0
    if has_toc_noise:
        clean_score -= 0.4
    if has_ocr_noise:
        clean_score -= 0.2
    if not domain_label and manual.product_name == "汇总英文":
        clean_score -= 0.1
    return {
        "has_image": bool(image_ids),
        "domain_label": domain_label,
        "is_toc": has_toc_noise,
        "has_ocr_noise": has_ocr_noise,
        "semantic_type": semantic_type,
        "is_procedure": semantic_type == "procedure",
        "is_warning_only": semantic_type == "safety_warning" and _step_count(text) < 2,
        "clean_score": round(max(clean_score, 0.1), 2),
    }


def _is_toc_like(text: str) -> bool:
    cleaned = _strip_markers_for_noise_check(text)
    dot_lines = 0
    page_like_lines = 0
    page_index_hits = 0
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in lines:
        if re.search(r"\.{5,}|…{3,}", line):
            dot_lines += 1
        if re.search(r"(?:page\s*)?\d{1,3}\s*$", line, flags=re.IGNORECASE):
            page_like_lines += 1
        page_index_hits += len(re.findall(r"[A-Za-z][A-Za-z /,'()-]{2,}\.?\s*\d{1,3}(?=\s|$)", line))
    if len(cleaned) < 80:
        return page_index_hits >= 3
    if dot_lines >= 2:
        return True
    if len(lines) >= 6 and page_like_lines / len(lines) >= 0.65:
        return True
    return False


def _strip_markers_for_noise_check(text: str) -> str:
    text = PIC_MARKER_RE.sub("", text)
    text = PIC_MISSING_RE.sub("", text)
    return text.strip()


def _is_low_value_fragment(text: str, *, image_ids: list[str]) -> bool:
    if image_ids:
        return False
    normalized = re.sub(r"\s+", " ", text.lstrip("# ").strip().lower())
    if not normalized:
        return True
    if normalized in {"warranty", "note", "notes", "notice"}:
        return True
    if len(normalized) < 24 and not re.search(r"\d|[。！？!?；;]", normalized):
        return True
    if len(normalized.split()) <= 4 and normalized.endswith(("warranty", "notice")):
        return True
    return False


def _detect_english_domain(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower())
    scores: dict[str, int] = {}
    for domain, hints in ENGLISH_DOMAIN_HINTS.items():
        score = 0
        for hint in hints:
            if _english_hint_occurs(normalized, hint):
                score += 2 if " " in hint else 1
        if score:
            scores[domain] = score
    if not scores:
        return ""
    return max(scores.items(), key=lambda item: item[1])[0]


def _backfill_english_domain_labels(chunks: list[KnowledgeChunk]) -> None:
    """Fill generic summary chunks with neighboring product-domain context."""

    labels = [str(chunk.metadata.get("domain_label") or "") for chunk in chunks]
    next_label: list[str] = [""] * len(chunks)
    next_distance: list[int] = [10**9] * len(chunks)
    seen_label = ""
    seen_index = 10**9
    for index in range(len(chunks) - 1, -1, -1):
        if labels[index]:
            seen_label = labels[index]
            seen_index = index
        next_label[index] = seen_label
        next_distance[index] = seen_index - index if seen_label else 10**9

    previous_label = ""
    previous_index = -10**9
    for index, chunk in enumerate(chunks):
        if labels[index]:
            previous_label = labels[index]
            previous_index = index
            continue

        previous_distance = index - previous_index if previous_label else 10**9
        candidate = ""
        if previous_label and previous_label == next_label[index] and previous_distance <= 12 and next_distance[index] <= 12:
            candidate = previous_label
        elif previous_label and previous_distance <= 5:
            candidate = previous_label
        elif next_label[index] and next_distance[index] <= 5:
            candidate = next_label[index]

        if candidate:
            chunk.metadata["domain_label"] = candidate
            chunk.metadata["domain_inferred"] = True
            if isinstance(chunk.metadata.get("clean_score"), (int, float)):
                chunk.metadata["clean_score"] = round(min(float(chunk.metadata["clean_score"]) + 0.05, 1.0), 2)


def _english_hint_occurs(normalized_text: str, hint: str) -> bool:
    normalized_hint = re.sub(r"\s+", " ", hint.lower()).strip()
    if not normalized_hint:
        return False
    if " " in normalized_hint or "-" in normalized_hint:
        return normalized_hint in normalized_text
    return bool(re.search(rf"\b{re.escape(normalized_hint)}\b", normalized_text))


def _detect_semantic_type(*, title: str, text: str, is_toc: bool) -> str:
    if is_toc:
        return "toc"

    combined = f"{title}\n{text}"
    normalized = re.sub(r"\s+", " ", combined.lower())
    title_norm = re.sub(r"\s+", " ", title.lower())
    steps = _step_count(text)

    has_procedure = _contains_any(normalized, PROCEDURE_HINTS) or steps >= 2
    has_safety_title = _contains_any(title_norm, SAFETY_HINTS)
    has_safety = has_safety_title or _contains_any(normalized, SAFETY_HINTS)
    if has_safety_title and not (has_procedure and steps >= 2):
        return "safety_warning"
    if _contains_any(normalized, TROUBLESHOOTING_HINTS):
        return "troubleshooting"
    if _contains_any(normalized, PARTS_HINTS):
        return "parts_list"
    if _contains_any(normalized, SPECIFICATION_HINTS):
        return "specification"
    if has_procedure:
        return "procedure"
    if has_safety:
        return "safety_warning"
    return "general"


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint.lower() in text for hint in hints)


def _step_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if re.match(r"^\s*(?:\d+[\).、]|[A-Z][\).]|[a-z][\).]|[①-⑳])\s+", line):
            count += 1
    count += len(re.findall(r"(?:^|\s)(?:\d+[\).、])\s+[A-Z\u4e00-\u9fff]", text))
    return count


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
