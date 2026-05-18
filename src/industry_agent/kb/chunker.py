"""Chunk manual text into RAG-friendly knowledge units."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from industry_agent.kb.models import KnowledgeChunk, ManualDocument

PIC_MARKER_RE = re.compile(r"\[\[PIC:([^\]]+)\]\]")
PIC_MISSING_RE = re.compile(r"\[\[PIC_MISSING\]\]")
SECTION_RE = re.compile(r"(?m)^#\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n+")
STEP_LINE_RE = re.compile(r"^\s*(?:\d+[\).、]|[A-Z][\).]|[a-z][\).]|[①-⑳]|步骤\s*\d+|step\s*\d+)\s*", flags=re.IGNORECASE)
BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•·]|[①-⑳])\s+")
KEY_VALUE_LINE_RE = re.compile(r"^\s*[\w\u4e00-\u9fff /().-]{2,40}\s*[:：]\s*.+$")
UPPER_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9 /_-]{2,40}$")
LAYOUT_CODE_RE = re.compile(r"^[A-Z]{2}\d{5}$")
ISOLATED_MARKER_RE = re.compile(r"^(?:[①-⑳]+|\d{1,3})$")
TOC_LINE_DOT_RE = re.compile(r"\.{3,}|…{2,}")
TOC_LINE_PAGE_RE = re.compile(r"(?:\.|\s)(\d{1,3})(?=\s|$)")
TITLE_TRAILING_PAGE_RE = re.compile(r"(?:\.{2,}|\s+)(\d{1,3}(?:-\d{1,3})?)$")
TITLE_LAYOUT_CODE_RE = re.compile(r"\b[A-Z]{2}\d{5}\b")
TITLE_SPLIT_MARK_RE = re.compile(r"\s+[●·•]\s+")
MAX_OVERLAP_CHARS = 220
MAX_OVERLAP_UNITS = 2
ENGLISH_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "boat": (
        "boat", "anchor", "anchoring", "anchor light", "bimini top", "bilge pump", "jet thrust", "stern", "bow",
        "hull", "battery switch", "wet storage", "swim platform", "livewell", "no-wake",
        "helm", "aerator switch", "navigation and anchor lights",
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
        "vk540", "suspension", "spring preload", "fresh snow", "v-belt holder",
        "crossing a slope", "riding uphill", "riding downhill",
    ),
    "landline": (
        "landline", "base station", "handset", "answering machine", "phonebook",
    ),
    "camera": (
        "camera", "viewfinder", "autofocus", "battery grip", "eos", "mode dial",
        "cf card", "image playback", "battery charger", "lcd panel",
        "white balance", "aperture", "exposure", "metering", "flash", "iso speed",
        "picture style", "shooting", "image-recording", "date/time battery",
    ),
    "lawn_mower": (
        "lawn mower", "mower deck", "blade-control switch", "height-of-cut", "grass deflector",
        "cutting blade", "anti-scalp", "pto", "roll bar", "electric deck lift",
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
        "television", "hdmi", "captions", "on-screen text", "outdoor antenna",
        "signal reception", "dvd player", "supplier's declaration of conformity",
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
GENERIC_ENGLISH_SECTION_PREFIXES: tuple[str, ...] = (
    "select ", "set ", "press ", "display ", "go ", "turn ", "start ", "stop ",
    "check ", "view ", "delete ", "protect ", "rotate ", "connect ", "jump ",
    "print ", "remove ", "insert ", "open ", "close ", "attach ", "install ",
)
ENGLISH_GENERIC_DOMAIN_TITLES: tuple[str, ...] = (
    "warning", "caution", "important", "note", "notes", "tip", "general safety",
    "maintenance safety", "technical specifications", "front view", "contents",
    "table of contents", "chapter 1", "chapter 2", "appendices", "speedometer",
    "fuel meter", "oil tank filler cap", "remote control levers",
)
ENGLISH_GENERIC_DOMAIN_PHRASES: tuple[str, ...] = (
    "warning", "caution", "important", "tip", "note", "battery", "fuel", "button",
    "lever", "switch", "screen", "menu", "view", "care", "cleaning", "replacement",
    "installation", "instructions", "general safety",
)


@dataclass
class SectionPlan:
    section_index: int
    section_text: str
    title: str
    semantic_type: str
    explicit_domain_label: str = ""
    domain_label: str = ""
    domain_inferred: bool = False
    domain_segment_index: int = -1


def chunk_manual(
    manual: ManualDocument,
    marked_text: str,
    *,
    project_root: Path,
    max_chars: int = 1200,
) -> list[KnowledgeChunk]:
    """Create ordered chunks from a marked manual text."""

    chunks: list[KnowledgeChunk] = []
    section_plans = _prepare_section_plans(manual, marked_text)
    for plan in section_plans:
        for part in _split_to_size(plan.section_text, max_chars=max_chars, semantic_type=plan.semantic_type):
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
                title=plan.title,
                text=clean_text,
                image_ids=image_ids,
                section_semantic_type=plan.semantic_type,
                section_domain_label=plan.domain_label,
                section_domain_inferred=plan.domain_inferred,
                domain_segment_index=plan.domain_segment_index,
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    manual_id=manual.manual_id,
                    product_name=manual.product_name,
                    source_path=str(manual.source_path.relative_to(project_root)),
                    title=plan.title,
                    text=clean_text,
                    image_ids=image_ids,
                    section_index=plan.section_index,
                    chunk_index=chunk_index,
                    char_count=len(clean_text),
                    metadata=metadata,
                )
            )
    return chunks


def _prepare_section_plans(manual: ManualDocument, marked_text: str) -> list[SectionPlan]:
    plans: list[SectionPlan] = []
    is_english = manual.manual_id.startswith("汇总英文手册")
    for section_index, section_text in enumerate(_split_sections(marked_text)):
        if _is_toc_like(section_text):
            continue
        title = _derive_title(section_text)
        clean_text = _strip_markers(section_text)
        semantic_type = _detect_semantic_type(
            title=title,
            text=clean_text,
            is_toc=False,
        )
        explicit_domain_label = _detect_english_domain(f"{title}\n{clean_text}") if is_english else ""
        plans.append(
            SectionPlan(
                section_index=section_index,
                section_text=section_text,
                title=title,
                semantic_type=semantic_type,
                explicit_domain_label=explicit_domain_label,
                domain_label=explicit_domain_label,
            )
        )
    if is_english:
        _infer_english_section_domains(plans)
        _smooth_english_section_domains(plans)
        _annotate_english_section_segments(plans)
    return plans


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


def _split_to_size(section_text: str, *, max_chars: int, semantic_type: str) -> list[str]:
    units = _merge_picture_neighborhood(_section_units(section_text, semantic_type=semantic_type))
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
            overlap = _overlap_units(buffer, semantic_type=semantic_type, max_chars=max_chars)
            buffer = _fit_units_within_limit([*overlap, unit], max_chars=max_chars)
            buffer_len = _joined_length(buffer)
        else:
            buffer.append(unit)
            buffer_len = _joined_length(buffer)

    if buffer:
        chunks.append("\n".join(buffer).strip())
    return [chunk for chunk in chunks if chunk.strip()]


def _section_units(section_text: str, *, semantic_type: str) -> list[str]:
    lines = [line.rstrip() for line in section_text.splitlines()]
    if semantic_type in {"procedure", "troubleshooting"}:
        return _procedure_like_units(lines)
    if semantic_type in {"parts_list", "specification", "safety_warning"}:
        return _line_group_units(lines)
    return _sentence_units(section_text)


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


def _procedure_like_units(lines: list[str]) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                units.append("\n".join(current).strip())
                current = []
            continue
        if stripped.startswith("# "):
            if current:
                units.append("\n".join(current).strip())
                current = []
            units.append(stripped)
            continue
        if _is_step_like_line(stripped):
            if current:
                units.append("\n".join(current).strip())
            current = [stripped]
            continue
        if _is_picture_marker_only(stripped):
            if current:
                current.append(stripped)
            elif units:
                units[-1] = f"{units[-1]}\n{stripped}".strip()
            else:
                current = [stripped]
            continue
        if current:
            current.append(stripped)
        else:
            current = [stripped]
    if current:
        units.append("\n".join(current).strip())
    return [unit for unit in units if unit.strip()]


def _line_group_units(lines: list[str]) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                units.append("\n".join(current).strip())
                current = []
                current_len = 0
            continue
        if stripped.startswith("# "):
            if current:
                units.append("\n".join(current).strip())
                current = []
                current_len = 0
            units.append(stripped)
            continue
        starts_new_group = (
            _is_step_like_line(stripped)
            or bool(BULLET_LINE_RE.match(stripped))
            or bool(KEY_VALUE_LINE_RE.match(stripped))
            or bool(UPPER_LABEL_RE.fullmatch(stripped))
        )
        if current and (starts_new_group or current_len + len(stripped) > 260):
            units.append("\n".join(current).strip())
            current = [stripped]
            current_len = len(stripped)
            continue
        current.append(stripped)
        current_len += len(stripped)
    if current:
        units.append("\n".join(current).strip())
    return [unit for unit in units if unit.strip()]


def _merge_picture_neighborhood(units: list[str]) -> list[str]:
    if not units:
        return []
    merged: list[str] = []
    index = 0
    while index < len(units):
        unit = units[index].strip()
        if not _is_picture_marker_only(unit):
            merged.append(unit)
            index += 1
            continue

        previous = merged.pop() if merged else ""
        next_unit = units[index + 1].strip() if index + 1 < len(units) else ""
        combined_parts = [part for part in (previous, unit, next_unit) if part]
        combined = "\n".join(combined_parts).strip()
        if combined and len(combined) <= 320:
            merged.append(combined)
            index += 2 if next_unit else 1
            continue
        if previous:
            merged.append(f"{previous}\n{unit}".strip())
        elif next_unit:
            merged.append(f"{unit}\n{next_unit}".strip())
            index += 2
            continue
        else:
            merged.append(unit)
        index += 1
    return [unit for unit in merged if unit.strip()]


def _overlap_units(buffer: list[str], *, semantic_type: str, max_chars: int) -> list[str]:
    overlap: list[str] = []
    overlap_len = 0
    for unit in reversed(buffer):
        stripped = unit.strip()
        if not stripped:
            continue
        unit_len = len(stripped)
        if overlap and (
            len(overlap) >= MAX_OVERLAP_UNITS
            or overlap_len + unit_len > min(MAX_OVERLAP_CHARS, max_chars // 3)
        ):
            break
        if semantic_type in {"procedure", "troubleshooting"} and not (
            _is_step_like_line(stripped.splitlines()[0]) or _is_picture_marker_only(stripped)
        ):
            if overlap:
                break
        overlap.insert(0, stripped)
        overlap_len += unit_len
    return overlap


def _joined_length(units: list[str]) -> int:
    if not units:
        return 0
    return sum(len(unit) for unit in units) + max(len(units) - 1, 0)


def _fit_units_within_limit(units: list[str], *, max_chars: int) -> list[str]:
    if not units:
        return []
    fitted = [unit for unit in units if unit.strip()]
    while len(fitted) > 1 and _joined_length(fitted) > max_chars:
        del fitted[0]
    if fitted and len(fitted) == 1 and len(fitted[0]) > max_chars:
        return _hard_split(fitted[0], max_chars=max_chars)
    return fitted


def _hard_split(text: str, *, max_chars: int) -> list[str]:
    return [text[index : index + max_chars].strip() for index in range(0, len(text), max_chars)]


def _derive_title(section_text: str) -> str:
    text = _strip_markers(section_text).lstrip("# ").strip()
    first_line = _normalize_title_line(text.splitlines()[0].strip()) if text else "未命名章节"
    first_sentence = re.split(r"[。！？!?；;]", first_line, maxsplit=1)[0].strip()
    title = first_sentence or first_line or "未命名章节"
    return title[:80]


def _normalize_title_line(line: str) -> str:
    normalized = line.strip().lstrip("#").strip()
    normalized = TITLE_LAYOUT_CODE_RE.sub("", normalized)
    normalized = TOC_LINE_DOT_RE.sub(" ", normalized)
    normalized = TITLE_TRAILING_PAGE_RE.sub("", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    normalized = normalized.strip(" -.:：")
    normalized = _shorten_title_line(normalized)
    return normalized or "未命名章节"


def _shorten_title_line(line: str) -> str:
    if not line:
        return ""
    split_match = TITLE_SPLIT_MARK_RE.search(line)
    if split_match:
        candidate = line[: split_match.start()].strip(" -.:：")
        if 2 <= len(candidate) <= 60:
            return candidate
    colon_candidate = _title_prefix_before_colon(line)
    if colon_candidate:
        return colon_candidate
    if len(line) > 60:
        english_candidate = _leading_english_phrase(line)
        if english_candidate:
            return english_candidate
    return line


def _title_prefix_before_colon(line: str) -> str:
    for mark in (":", "："):
        if mark not in line:
            continue
        prefix, suffix = line.split(mark, 1)
        prefix = prefix.strip(" -.:：")
        suffix = suffix.strip()
        if 3 <= len(prefix) <= 60 and suffix:
            prefix_words = len(prefix.split())
            if prefix_words <= 10 or len(prefix) <= 28:
                return prefix
    return ""


def _leading_english_phrase(line: str) -> str:
    words = line.split()
    if len(words) < 4:
        return ""
    stop_tokens = {
        "the", "this", "these", "those", "when", "if", "once", "all", "do", "never",
        "follow", "allows", "allow", "contains", "provides", "indicates", "shows",
    }
    candidate_words: list[str] = []
    for index, word in enumerate(words):
        lower = word.lower().strip(".,;:!?")
        if index >= 2 and lower in stop_tokens:
            break
        candidate_words.append(word)
        if len(candidate_words) >= 8:
            break
    candidate = " ".join(candidate_words).strip(" -.:：")
    if 3 <= len(candidate) <= 60 and candidate != line:
        return candidate
    return ""


def _strip_markers(text: str) -> str:
    text = PIC_MARKER_RE.sub("", text)
    text = PIC_MISSING_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _strip_layout_artifacts(text.strip())


def _build_chunk_metadata(
    *,
    manual: ManualDocument,
    title: str,
    text: str,
    image_ids: list[str],
    section_semantic_type: str,
    section_domain_label: str = "",
    section_domain_inferred: bool = False,
    domain_segment_index: int = -1,
) -> dict[str, object]:
    chunk_domain_label = _detect_english_domain(f"{title}\n{text}") if manual.manual_id.startswith("汇总英文手册") else ""
    domain_label = chunk_domain_label or section_domain_label
    has_toc_noise = _is_toc_like(text)
    has_ocr_noise = bool(re.search(r"\\u[0-9a-fA-F]{4}|\\(?:mathsf|mathrm|pmb)|[a-z]{18,}", text))
    semantic_type = _detect_semantic_type(title=title, text=text, is_toc=has_toc_noise)
    step_count = _step_count(text)
    line_count = len([line for line in text.splitlines() if line.strip()])
    key_value_pairs = len(re.findall(r"(?m)^\s*[\w\u4e00-\u9fff /().-]{2,40}\s*[:：]\s*.+$", text))
    clean_score = 1.0
    if has_toc_noise:
        clean_score -= 0.4
    if has_ocr_noise:
        clean_score -= 0.2
    if not domain_label and manual.manual_id.startswith("汇总英文手册"):
        clean_score -= 0.1
    if line_count <= 1 and step_count == 0 and not image_ids and len(text) < 40:
        clean_score -= 0.15
    return {
        "has_image": bool(image_ids),
        "image_count": len(image_ids),
        "domain_label": domain_label,
        "section_domain_label": section_domain_label,
        "domain_inferred": bool(section_domain_inferred and not chunk_domain_label),
        "domain_segment_index": domain_segment_index if domain_segment_index >= 0 else None,
        "domain_segment_label": section_domain_label if domain_segment_index >= 0 and section_domain_label else "",
        "sub_manual_id": _make_sub_manual_id(
            manual_id=manual.manual_id,
            domain_label=section_domain_label,
            domain_segment_index=domain_segment_index,
        ),
        "is_toc": has_toc_noise,
        "has_ocr_noise": has_ocr_noise,
        "semantic_type": semantic_type,
        "section_semantic_type": section_semantic_type,
        "chunk_type": semantic_type,
        "is_procedure": semantic_type == "procedure",
        "is_warning_only": semantic_type == "safety_warning" and step_count < 2,
        "step_count": step_count,
        "line_count": line_count,
        "key_value_pairs": key_value_pairs,
        "title_length": len(title),
        "has_overlap_context": step_count > 0 or bool(image_ids),
        "clean_score": round(max(clean_score, 0.1), 2),
    }


def _is_toc_like(text: str) -> bool:
    cleaned = _strip_markers_for_noise_check(text)
    dot_lines = 0
    page_like_lines = 0
    page_index_hits = 0
    toc_like_lines = 0
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in lines:
        if TOC_LINE_DOT_RE.search(line):
            dot_lines += 1
        if re.search(r"(?:page\s*)?\d{1,3}\s*$", line, flags=re.IGNORECASE):
            page_like_lines += 1
        page_index_hits += len(re.findall(r"[A-Za-z][A-Za-z /,'()-]{2,}\.?\s*\d{1,3}(?=\s|$)", line))
        if _looks_like_toc_line(line):
            toc_like_lines += 1
    if len(lines) == 1 and lines:
        page_tokens = TOC_LINE_PAGE_RE.findall(lines[0])
        if _looks_like_toc_line(lines[0]) and (len(page_tokens) >= 2 or page_index_hits >= 2):
            return True
    if len(cleaned) < 80:
        return page_index_hits >= 3
    if toc_like_lines >= 2 and toc_like_lines >= max(2, len(lines) // 2):
        return True
    if dot_lines >= 2:
        return True
    if len(lines) >= 6 and page_like_lines / len(lines) >= 0.65:
        return True
    return False


def _strip_markers_for_noise_check(text: str) -> str:
    text = PIC_MARKER_RE.sub("", text)
    text = PIC_MISSING_RE.sub("", text)
    return _strip_layout_artifacts(text.strip())


def _strip_layout_artifacts(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        line = TITLE_LAYOUT_CODE_RE.sub("", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if LAYOUT_CODE_RE.fullmatch(line):
            continue
        if line == "#":
            continue
        if ISOLATED_MARKER_RE.fullmatch(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _looks_like_toc_line(line: str) -> bool:
    normalized = line.strip().lstrip("#").strip()
    if not normalized:
        return False
    title_words = re.findall(r"[A-Za-z][A-Za-z /,'()-]{2,}", normalized)
    page_tokens = TOC_LINE_PAGE_RE.findall(normalized)
    if len(page_tokens) >= 3 and len(title_words) >= 3:
        return True
    if len(normalized) > 200:
        return False
    if (
        len(title_words) >= 2
        and sum(1 for word in title_words if TITLE_TRAILING_PAGE_RE.search(word.strip())) >= 2
    ):
        return True
    if TOC_LINE_DOT_RE.search(normalized):
        return True
    english_words = re.findall(r"[A-Za-z][A-Za-z'-]+", normalized)
    if len(page_tokens) >= 2 and len(title_words) >= 3:
        return True
    if normalized.endswith(tuple(str(i) for i in range(10))) and len(english_words) >= 2:
        if not re.search(r"[。！？!?：:]", normalized):
            return True
    return False


def _is_low_value_fragment(text: str, *, image_ids: list[str]) -> bool:
    if image_ids:
        return False
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(stripped_lines) == 1 and stripped_lines[0].startswith("#"):
        heading_body = stripped_lines[0].lstrip("# ").strip()
        if (
            len(heading_body) <= 20
            and not any(mark in heading_body for mark in ("。", "！", "？", ".", "!", "?", "：", ":"))
        ):
            return True
    normalized = re.sub(r"\s+", " ", text.lstrip("# ").strip().lower())
    if not normalized:
        return True
    if normalized in {"warranty", "note", "notes", "notice"}:
        return True
    if normalized in {"contents", "table of contents", "index"}:
        return True
    if len(normalized) < 24 and not re.search(r"\d|[。！？!?；;]", normalized):
        return True
    if len(normalized) < 40 and len(normalized.split()) <= 6 and not re.search(r"[:：]|[。！？!?；;]", normalized):
        return True
    if len(normalized.split()) <= 4 and normalized.endswith(("warranty", "notice")):
        return True
    if re.fullmatch(r"(?:page\s*)?\d{1,3}", normalized):
        return True
    return False


def _detect_english_domain(text: str) -> str:
    scored_domains = _score_english_domains(text)
    if not scored_domains:
        return ""
    top_domain, top_score = scored_domains[0]
    second_score = scored_domains[1][1] if len(scored_domains) > 1 else 0
    title = _strip_markers(text).splitlines()[0].lstrip("# ").strip() if text.strip() else ""
    title_is_generic = _is_generic_english_domain_title(title)
    if top_score < 3:
        return ""
    if title_is_generic and top_score < 5:
        return ""
    if top_score - second_score < 2 and top_score < 6:
        return ""
    return top_domain


def _score_english_domains(text: str) -> list[tuple[str, int]]:
    normalized = re.sub(r"\s+", " ", text.lower())
    scores: list[tuple[str, int]] = []
    for domain, hints in ENGLISH_DOMAIN_HINTS.items():
        score = 0
        for hint in hints:
            hit_count = _english_hint_count(normalized, hint)
            if hit_count <= 0:
                continue
            score += min(hit_count, 3) * (3 if " " in hint or len(hint) >= 9 else 1)
        if score > 0:
            scores.append((domain, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def _is_generic_english_domain_title(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(title).lower()).strip(" -.:：")
    if not normalized:
        return True
    if normalized in ENGLISH_GENERIC_DOMAIN_TITLES:
        return True
    words = [word for word in re.findall(r"[a-z]+", normalized) if word]
    informative_words = [word for word in words if word not in ENGLISH_GENERIC_DOMAIN_PHRASES]
    if len(words) <= 3 and not informative_words:
        return True
    return False


def _infer_english_section_domains(plans: list[SectionPlan]) -> None:
    """Fill generic english-summary sections with neighboring product-domain context."""

    labels = [plan.explicit_domain_label for plan in plans]
    next_label: list[str] = [""] * len(plans)
    next_distance: list[int] = [10**9] * len(plans)
    seen_label = ""
    seen_index = 10**9
    for index in range(len(plans) - 1, -1, -1):
        if labels[index]:
            seen_label = labels[index]
            seen_index = index
        next_label[index] = seen_label
        next_distance[index] = seen_index - index if seen_label else 10**9

    previous_label = ""
    previous_index = -10**9
    for index, plan in enumerate(plans):
        if labels[index]:
            previous_label = labels[index]
            previous_index = index
            continue

        previous_distance = index - previous_index if previous_label else 10**9
        candidate = ""
        if previous_label and previous_label == next_label[index] and previous_distance <= 18 and next_distance[index] <= 18:
            candidate = previous_label
        elif previous_label and previous_distance <= 8:
            candidate = previous_label
        elif next_label[index] and next_distance[index] <= 8:
            candidate = next_label[index]

        if candidate:
            plan.domain_label = candidate
            plan.domain_inferred = True


def _annotate_english_section_segments(plans: list[SectionPlan]) -> None:
    current_label = ""
    segment_index = -1
    for plan in plans:
        label = plan.domain_label
        if not label:
            continue
        if label != current_label:
            current_label = label
            segment_index += 1
        plan.domain_segment_index = segment_index


def _smooth_english_section_domains(plans: list[SectionPlan]) -> None:
    labels = [plan.domain_label for plan in plans]
    run_start = 0
    while run_start < len(plans):
        run_label = labels[run_start]
        run_end = run_start + 1
        while run_end < len(plans) and labels[run_end] == run_label:
            run_end += 1

        if run_label and (run_end - run_start) <= 3:
            previous_label = labels[run_start - 1] if run_start > 0 else ""
            next_label = labels[run_end] if run_end < len(plans) else ""
            if previous_label and previous_label == next_label:
                run_plans = plans[run_start:run_end]
                if all(_is_generic_english_section_plan(plan) for plan in run_plans):
                    for plan in run_plans:
                        if plan.domain_label != previous_label:
                            plan.domain_label = previous_label
                            plan.domain_inferred = True
        run_start = run_end


def _is_generic_english_section_plan(plan: SectionPlan) -> bool:
    title_norm = re.sub(r"\s+", " ", plan.title.lower()).strip()
    if not title_norm:
        return False
    if title_norm.startswith(GENERIC_ENGLISH_SECTION_PREFIXES):
        return True
    if title_norm in {"menu setting", "note", "warning", "caution"}:
        return True
    if len(title_norm) <= 24 and plan.semantic_type in {"general", "procedure", "specification"}:
        return True
    return False


def _english_hint_occurs(normalized_text: str, hint: str) -> bool:
    return _english_hint_count(normalized_text, hint) > 0


def _english_hint_count(normalized_text: str, hint: str) -> int:
    normalized_hint = re.sub(r"\s+", " ", hint.lower()).strip()
    if not normalized_hint:
        return 0
    if " " in normalized_hint or "-" in normalized_hint:
        return normalized_text.count(normalized_hint)
    return len(re.findall(rf"\b{re.escape(normalized_hint)}\b", normalized_text))


def _make_sub_manual_id(*, manual_id: str, domain_label: str, domain_segment_index: int) -> str:
    if not domain_label or domain_segment_index < 0:
        return manual_id
    return f"{manual_id}:{domain_label}:{domain_segment_index}"


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
    line_based_count = 0
    for line in text.splitlines():
        if _is_step_like_line(line):
            line_based_count += 1
    if line_based_count > 0:
        return line_based_count
    return len(re.findall(r"(?:^|\s)(?:\d+[\).、])\s+[A-Z\u4e00-\u9fff]", text))


def _is_step_like_line(text: str) -> bool:
    return bool(STEP_LINE_RE.match(text))


def _is_picture_marker_only(text: str) -> bool:
    stripped = text.strip()
    return bool(re.fullmatch(r"\[\[PIC:[^\]]+\]\]", stripped) or PIC_MISSING_RE.fullmatch(stripped))


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
