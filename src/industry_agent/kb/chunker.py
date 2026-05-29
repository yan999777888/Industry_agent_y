"""Chunk manual text into RAG-friendly knowledge units."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from industry_agent.kb.models import KnowledgeChunk, ManualDocument

# Optional: local embedding model for semantic chunking
# Disabled by default: requires sentence-transformers which may not be available
_SEMANTIC_CHUNKING = os.getenv("KB_SEMANTIC_CHUNKING", "0").strip().lower() in ("1", "true", "yes")
_SEMANTIC_MODEL_NAME = os.getenv("KB_EMBEDDING_MODEL", "BAAI/bge-m3")
_SEMANTIC_SIM_THRESHOLD = float(os.getenv("KB_SEMANTIC_THRESHOLD", "0.55"))
_SEMANTIC_MODEL = None  # lazy-loaded

# English-specific chunking constants
_EN_MAX_CHARS = int(os.getenv("KB_EN_MAX_CHARS", "1600"))
_EN_MIN_CHARS = int(os.getenv("KB_EN_MIN_CHARS", "200"))
_EN_MIN_MEANINGFUL_CHARS = int(os.getenv("KB_EN_MIN_MEANINGFUL_CHARS", "40"))
_EN_WORD_OVERLAP_THRESHOLD = 0.25
_EN_OCR_FIX = os.getenv("KB_EN_OCR_FIX", "1").strip().lower() in ("1", "true", "yes")

_OCR_FIX_PROMPT = """Fix ALL OCR/scanning errors in this English text. The text was extracted from a PDF manual by OCR and contains damaged words. Your job is to restore it to proper English. This text will be used to build a knowledge base (RAG) for product technical support, so every sentence must be clean, complete, and factually accurate.

Common error patterns to fix:
- Split words: "be en"→"been", "the ir"→"their", "a ny"→"any", "in a re as"→"in areas"
- Merged words: "topof"→"top of", "Thereareno"→"There are no"
- Character swaps: "fo"→"to", "fh"→"th", "ou f"→"out"
- Nonsense strings: "i Ad a pf local iz a fi on"→ try to reconstruct what makes sense in context
- Preserve product names, model numbers, technical terms — only fix words that are clearly damaged

Example input:
"Before using vacuum, pickup objects like clothing loose papers, pull cords fo be supervised fo ensure f hey do n of play with vacuum. Thereareno user-serviceable parts in side. Clean i Ad a pf local iz a fi on camera with a cloth."

Example output:
"Before using vacuum, pick up objects like clothing, loose papers, pull cords to be supervised to ensure they do not play with vacuum. There are no user-serviceable parts inside. Clean with a cloth."

Rules:
- Fix every word that looks damaged — no word should be left garbled
- Keep the same structure (headings, line breaks, numbering) unchanged
- Output ONLY the corrected text, no explanation

Text:
{text}

Corrected:"""


def _fix_english_ocr_llm(text: str) -> str:
    """Clean English OCR errors using the configured LLM."""
    text = text.strip()
    if not text or len(text) < 30:
        return text
    try:
        import httpx
        api_key = os.getenv("KB_OCR_API_KEY", "sk-afb92d9130384509885c6de4a50ddf9a")
        base_url = os.getenv("KB_OCR_BASE_URL", "https://api.deepseek.com/v1")
        model = os.getenv("KB_OCR_MODEL", "deepseek-v4-flash")
        if not api_key:
            return text
        # Only fix if there are clear OCR indicators (random spaces in short words)
        short_broken = len(re.findall(r'\b[a-z]{1,2}\s[a-z]{1,4}\s[a-z]{1,2}\b', text, re.IGNORECASE))
        if short_broken < 3:
            return text
        logger.warning("OCR_FIX: sending %d chars to %s", len(text[:4000]), model)
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": _OCR_FIX_PROMPT.format(text=text[:4000])}],
                "temperature": 0.05, "max_tokens": 4096,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        # Remove common prefixes the LLM might add
        for prefix in ("Corrected:", "Corrected text:", "Fixed text:"):
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        if len(result) >= len(text) * 0.5:
            return result
    except Exception as exc:
        logger.warning("OCR fix failed: %s", exc)
    return text

logger = logging.getLogger(__name__)

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
    min_chars: int = 150,
) -> list[KnowledgeChunk]:
    """Create ordered chunks from a marked manual text."""

    chunks: list[KnowledgeChunk] = []
    is_english = manual.manual_id.startswith("汇总英文手册")
    section_plans = _prepare_section_plans(manual, marked_text)
    if is_english:
        splitter = _split_english  # English: word-overlap semantic boundaries
    elif _SEMANTIC_CHUNKING:
        splitter = _split_semantic  # Embedding-based semantic boundaries
    else:
        splitter = _split_to_size  # Size-based greedy packing
    for plan in section_plans:
        if is_english and _EN_OCR_FIX:
            plan.section_text = _fix_english_ocr_llm(plan.section_text)
        if is_english:
            parts = splitter(plan.section_text, max_chars=_EN_MAX_CHARS, min_chars=_EN_MIN_CHARS)
        else:
            parts = splitter(plan.section_text, max_chars=max_chars, semantic_type=plan.semantic_type, min_chars=min_chars)
        for part in parts:
            clean_text = _embed_image_anchors(part, is_english=is_english)
            if not clean_text:
                continue

            image_ids = _unique_in_order(PIC_MARKER_RE.findall(part))
            if _is_low_value_fragment(clean_text, image_ids=image_ids):
                continue
            # Filter truly empty chunks (no text, no images)
            if not image_ids and not re.sub(r'\s', '', clean_text):
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
            # English: use domain_label as actual product name
            _product = manual.product_name
            if is_english and plan.domain_label:
                _domain_map = {
                    'boat': 'boat', 'camera': 'camera', 'motherboard': 'motherboard',
                    'snowmobile': 'snowmobile', 'lawn_mower': 'lawn mower', 'landline': 'landline phone',
                    'microwave': 'microwave', 'grill': 'grill', 'airfryer': 'air fryer',
                    'television': 'television', 'ereader': 'e-reader', 'pressure_cooker': 'pressure cooker',
                    'fax': 'fax machine', 'toothbrush': 'toothbrush', 'washing_machine': 'washing machine',
                    'coffee_machine': 'coffee machine', 'vacuum': 'vacuum cleaner', 'earphone': 'earphone',
                }
                _product = _domain_map.get(plan.domain_label, manual.product_name)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    manual_id=manual.manual_id,
                    product_name=_product,
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
    # English: generate child chunks (sentences) for fine-grained vector retrieval.
    # Each child points to its parent via metadata.parent_chunk_id.
    if is_english:
        _parent_index = len(chunks)
        for pi in range(_parent_index):
            parent = chunks[pi]
            text_clean = re.sub(r'\[IMG_\d+_[a-zA-Z0-9_\-]+\]', '', parent.text).strip()
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text_clean) if s.strip()]
            if len(sents) < 2:
                continue
            for si, sent in enumerate(sents):
                if len(sent) < 30:
                    continue
                child_id = f"{parent.chunk_id}_c{si}"
                child_meta = dict(parent.metadata)
                child_meta["parent_chunk_id"] = parent.chunk_id
                child_meta["is_child"] = True
                child_meta["parent_text"] = parent.text
                child_meta["parent_image_ids"] = parent.image_ids
                chunks.append(KnowledgeChunk(
                    chunk_id=child_id,
                    manual_id=parent.manual_id,
                    product_name=parent.product_name,
                    source_path=parent.source_path,
                    title=parent.title,
                    text=sent,
                    image_ids=[],
                    section_index=parent.section_index,
                    chunk_index=len(chunks),
                    char_count=len(sent),
                    metadata=child_meta,
                ))

    # Merge pure-image chunks into the preceding chunk
    merged: list[KnowledgeChunk] = []
    for c in chunks:
        text_only = re.sub(r'\[IMG_\d+_[a-zA-Z0-9_\-]+\]', '', c.text).strip()
        if not text_only and c.image_ids and merged:
            prev = merged[-1]
            for img in c.image_ids:
                if img not in prev.image_ids:
                    prev.image_ids.append(img)
                    prev.text += f'\n[IMG_{prev.image_ids.index(img)}_{img}]'
                    prev.char_count = len(prev.text)
                    prev.metadata.setdefault("image_count", 0)
                    prev.metadata["image_count"] = len(prev.image_ids)
        else:
            merged.append(c)
    return merged


def _prepare_section_plans(manual: ManualDocument, marked_text: str) -> list[SectionPlan]:
    plans: list[SectionPlan] = []
    is_english = manual.manual_id.startswith("汇总英文手册")
    for section_index, section_text in enumerate(_split_sections(marked_text)):
        if _is_toc_like(section_text):
            continue
        title = _derive_title(section_text)
        # Strip the first # heading line from body — title is already captured
        body_text = _strip_heading_line(section_text)
        clean_text = _embed_image_anchors(body_text)
        semantic_type = _detect_semantic_type(
            title=title,
            text=clean_text,
            is_toc=False,
        )
        explicit_domain_label = _detect_english_domain(f"{title}\n{clean_text}") if is_english else ""
        plans.append(
            SectionPlan(
                section_index=section_index,
                section_text=body_text,
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


def _add_english_sentences(text: str, out: list[str]) -> None:
    """Split English prose into sentences, respecting abbreviations."""
    if not text.strip():
        return
    _abbr = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Rd|Blvd|vs|etc|e\.g|i\.e|fig|Fig|approx|No)'
    pattern = re.compile(
        rf'(?:(?:{_abbr})|[^.!?\n])+'
        rf'(?:[.!?]|$)(?:\s|(?=\n|$))',
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        sent = m.group().strip()
        if sent:
            out.append(sent)


def _split_english(section_text: str, *, max_chars: int = _EN_MAX_CHARS, min_chars: int = _EN_MIN_CHARS) -> list[str]:
    """Split English text into topically coherent chunks using sentence boundaries and size limits."""
    # Split into proper English sentences
    sentences: list[str] = []
    position = 0
    for m in PIC_MARKER_RE.finditer(section_text):
        _add_english_sentences(section_text[position:m.start()], sentences)
        sentences.append(m.group(0))
        position = m.end()
    _add_english_sentences(section_text[position:], sentences)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    # Compute word overlap as topic-shift indicator (soft signal)
    def _word_terms(t: str) -> set[str]:
        words = re.findall(r'[a-zA-Z0-9]{3,}', t.lower())
        return set(words) | {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}

    s_terms = [_word_terms(s) for s in sentences]
    overlaps: list[float] = []
    for i in range(len(sentences) - 1):
        a, b = s_terms[i], s_terms[i + 1]
        overlaps.append(len(a & b) / max(len(a | b), 1.0) if a and b else 0.0)
    # Mark boundary only if overlap is extremely low AND buffer is substantial
    boundaries: set[int] = set()
    for i, ov in enumerate(overlaps):
        if ov < 0.08 and i > 0:
            boundaries.add(i)

    # Greedy pack with sentence-boundary backtracking: never cut mid-sentence
    def _flush_backtrack(buf: list[str]) -> tuple[str, list[str]]:
        """Flush buf at last sentence boundary; return (chunk_text, carryover)."""
        if len(buf) <= 1:
            return " ".join(buf), []
        text = " ".join(buf)
        # Find all sentence boundaries after the halfway point
        mid = len(text) // 2
        best = -1
        for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z"[{\(])|(?<=\n)\s*', text):
            if m.end() >= mid:
                best = m.end()
        if best > 0:
            return text[:best].strip(), [text[best:].strip()]
        return text, []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for i, sent in enumerate(sentences):
        s_len = len(sent)
        if s_len > max_chars:
            if buf:
                flushed, carry = _flush_backtrack(buf)
                if flushed: chunks.append(flushed)
                buf = carry if carry else []
                buf_len = sum(len(s) for s in buf) if carry else 0
            chunks.append(sent)
            continue
        next_len = buf_len + s_len + 1
        force_split = (i - 1) in boundaries and buf_len >= min_chars and next_len > max_chars * 0.7
        if force_split:
            flushed, carry = _flush_backtrack(buf)
            if flushed: chunks.append(flushed)
            buf = carry if carry else []
            buf_len = sum(len(s) for s in buf) if carry else 0
        if buf_len + s_len + 1 > max_chars:
            flushed, carry = _flush_backtrack(buf)
            if flushed: chunks.append(flushed)
            buf = carry if carry else []
            buf_len = sum(len(s) for s in buf) if carry else 0
        buf.append(sent)
        buf_len += s_len
    if buf:
        flushed, _ = _flush_backtrack(buf)
        if flushed: chunks.append(flushed)

    # Precise filter: remove chunks with almost no alpha-numeric content
    result: list[str] = []
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        text_only = re.sub(r'\[IMG_\d+_[a-zA-Z0-9_\-]+\]', '', stripped).strip()
        alpha = re.sub(r'[^a-zA-Z0-9]', '', text_only)
        if len(alpha) < _EN_MIN_MEANINGFUL_CHARS:
            continue
        result.append(stripped)
    return result


def _split_to_size(section_text: str, *, max_chars: int, semantic_type: str, min_chars: int = 150, is_english: bool = False) -> list[str]:
    units = _merge_picture_neighborhood(_section_units(section_text, semantic_type=semantic_type, is_english=is_english))
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
    chunks = [chunk for chunk in chunks if chunk.strip()]
    # Merge adjacent undersized chunks so no fragment falls below min_chars
    if min_chars > 0:
        return _merge_undersized(chunks, min_chars=min_chars, max_chars=max_chars)
    return chunks


def _load_embedding_model() -> object | None:
    """Lazy-load sentence-transformers for semantic chunking."""
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None and _SEMANTIC_CHUNKING:
        try:
            from sentence_transformers import SentenceTransformer
            _SEMANTIC_MODEL = SentenceTransformer(_SEMANTIC_MODEL_NAME)
            logger.warning("SEMANTIC_CHUNK: loaded model=%s", _SEMANTIC_MODEL_NAME)
        except Exception as exc:
            logger.warning("SEMANTIC_CHUNK: failed to load model: %s", exc)
    return _SEMANTIC_MODEL


def _compute_unit_embeddings(units: list[str]) -> list[np.ndarray]:
    """Batch-compute embeddings for text units using the local model."""
    model = _load_embedding_model()
    if model is None:
        return []
    cleaned = [u[:512] for u in units]
    embeddings = model.encode(cleaned, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return [np.array(e, dtype=np.float32) for e in embeddings]


def _split_semantic(section_text: str, *, max_chars: int, semantic_type: str, min_chars: int = 150, is_english: bool = False) -> list[str]:
    """Split section text into chunks at semantic boundaries (topic shifts)."""
    units = _merge_picture_neighborhood(_section_units(section_text, semantic_type=semantic_type, is_english=is_english))
    if not units:
        return []
    model = _load_embedding_model()
    if model is None:
        return _split_to_size(section_text, max_chars=max_chars, semantic_type=semantic_type, min_chars=min_chars)
    embeddings = _compute_unit_embeddings(units)
    sim_gaps: list[float] = []
    for i in range(len(units) - 1):
        if i < len(embeddings) and i + 1 < len(embeddings):
            sim_gaps.append(float(np.dot(embeddings[i], embeddings[i + 1])))
        else:
            sim_gaps.append(1.0)
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for idx, unit in enumerate(units):
        if len(unit) > max_chars:
            if buffer:
                chunks.append("\n".join(buffer).strip())
                buffer = []; buffer_len = 0
            chunks.extend(_hard_split(unit, max_chars=max_chars))
            continue
        next_len = buffer_len + len(unit) + 1
        should_split = (
            buffer_len >= min_chars
            and idx > 0 and idx - 1 < len(sim_gaps)
            and sim_gaps[idx - 1] < _SEMANTIC_SIM_THRESHOLD
            and next_len <= max_chars * 1.2
        )
        if should_split:
            chunks.append("\n".join(buffer).strip())
            buffer = []; buffer_len = 0
        if next_len > max_chars:
            chunks.append("\n".join(buffer).strip())
            overlap = _overlap_units(buffer, semantic_type=semantic_type, max_chars=max_chars)
            buffer = _fit_units_within_limit([*overlap, unit], max_chars=max_chars)
            buffer_len = _joined_length(buffer)
        else:
            buffer.append(unit)
            buffer_len = _joined_length(buffer)
    if buffer:
        chunks.append("\n".join(buffer).strip())
    chunks = [c for c in chunks if c.strip()]
    if min_chars > 0:
        chunks = _merge_undersized(chunks, min_chars=min_chars, max_chars=max_chars)
    logger.warning("SEMANTIC_CHUNK: %d units → %d chunks (thresh=%.2f)", len(units), len(chunks), _SEMANTIC_SIM_THRESHOLD)
    return chunks


_EN_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n+")

def _section_units(section_text: str, *, semantic_type: str, is_english: bool = False) -> list[str]:
    lines = [line.rstrip() for line in section_text.splitlines()]
    if semantic_type in {"procedure", "troubleshooting"}:
        return _procedure_like_units(lines)
    if semantic_type in {"parts_list", "specification", "safety_warning"}:
        return _line_group_units(lines)
    if is_english:
        return _sentence_units(section_text, split_re=_EN_SENTENCE_SPLIT_RE)
    return _sentence_units(section_text)


def _sentence_units(text: str, split_re: re.Pattern | None = None) -> list[str]:
    splitter = split_re or SENTENCE_SPLIT_RE
    units: list[str] = []
    position = 0
    for match in PIC_MARKER_RE.finditer(text):
        units.extend(_plain_sentence_units(text[position : match.start()], split_re=splitter))
        units.append(match.group(0))
        position = match.end()
    units.extend(_plain_sentence_units(text[position:], split_re=splitter))
    return [unit.strip() for unit in units if unit.strip()]


def _plain_sentence_units(text: str, split_re: re.Pattern | None = None) -> list[str]:
    splitter = split_re or SENTENCE_SPLIT_RE
    rough_units = splitter.split(text)
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


def _merge_undersized(chunks: list[str], *, min_chars: int, max_chars: int = 1200) -> list[str]:
    """Merge adjacent chunks below min_chars into neighbors.

    Prevents tiny fragments that are hard to retrieve via keyword/vector search.
    PIC markers are preserved through the text merge.
    """
    if not chunks:
        return []
    merged: list[str] = []
    pending = chunks[0]
    for chunk in chunks[1:]:
        if len(pending) < min_chars:
            # Merge pending into current chunk
            pending = f"{pending}\n{chunk}"
        elif len(chunk) < min_chars and len(pending) + len(chunk) < max_chars * 1.2:
            # Chunk is short and merging won't make pending too large
            pending = f"{pending}\n{chunk}"
        else:
            merged.append(pending)
            pending = chunk
    if pending:
        if merged and len(pending) < min_chars and len(merged[-1]) + len(pending) < max_chars * 1.5:
            merged[-1] = f"{merged[-1]}\n{pending}"
        else:
            merged.append(pending)
    return merged


def _merge_two_chunks(a: KnowledgeChunk, b: KnowledgeChunk) -> KnowledgeChunk:
    """Merge two adjacent chunks into one, keeping first chunk's identity fields."""
    combined_text = f"{a.text}\n{b.text}".strip()
    combined_image_ids = _unique_in_order(a.image_ids + b.image_ids)
    return KnowledgeChunk(
        chunk_id=a.chunk_id,
        manual_id=a.manual_id,
        product_name=a.product_name,
        source_path=a.source_path,
        title=a.title,
        text=combined_text,
        image_ids=combined_image_ids,
        section_index=a.section_index,
        chunk_index=a.chunk_index,
        char_count=len(combined_text),
        metadata=a.metadata,
    )


def _merge_chunks_post_process(chunks: list[KnowledgeChunk], *, min_chars: int, max_chars: int = 1200) -> list[KnowledgeChunk]:
    """Merge adjacent short chunks across section boundaries (post-processing pass)."""
    if min_chars <= 0 or not chunks:
        return chunks
    merged: list[KnowledgeChunk] = []
    carry = chunks[0]
    for chunk in chunks[1:]:
        if carry.char_count < min_chars:
            carry = _merge_two_chunks(carry, chunk)
        elif chunk.char_count < min_chars and carry.char_count + chunk.char_count <= int(max_chars * 1.3):
            carry = _merge_two_chunks(carry, chunk)
        else:
            merged.append(carry)
            carry = chunk
    if carry:
        if merged and carry.char_count < min_chars and merged[-1].char_count + carry.char_count <= int(max_chars * 1.3):
            merged[-1] = _merge_two_chunks(merged[-1], carry)
        else:
            merged.append(carry)

    # Renumber chunk_index and regenerate chunk_id
    for i, chunk in enumerate(merged):
        chunk.chunk_index = i
        chunk.chunk_id = _make_chunk_id(chunk.manual_id, i, chunk.text)
    return merged


def _strip_heading_line(section_text: str) -> str:
    """Remove the first # heading prefix from section text (title already captured)."""
    lines = section_text.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        # Only strip the # prefix from the first line, keep rest of the content
        first = lines[0].lstrip()
        first = re.sub(r"^#+\s*", "", first, count=1).strip()
        lines[0] = first
        return "\n".join(lines).strip()
    return section_text.strip()


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


def _embed_image_anchors(text: str, *, is_english: bool = False) -> str:
    """Replace [[PIC:id]] with numbered [IMG_X_id] anchors in the text."""
    counter = 0
    def _replacer(m: re.Match) -> str:
        nonlocal counter
        img_id = m.group(1)
        anchor = f"[IMG_{counter}_{img_id}]"
        counter += 1
        return anchor

    text = PIC_MARKER_RE.sub(_replacer, text)
    text = PIC_MISSING_RE.sub("", text)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if is_english:
        text = _fix_english_spacing(text)
    return _strip_layout_artifacts(text.strip())


_EN_WORDS = {"the","to","at","in","on","of","for","and","or","is","it","as",
             "by","with","from","an","a","be","no","up","if","we","he","she",
             "are","was","were","been","has","had","not","but","all","can","may",
             "will","this","that","have","their","there","which","what","when",
             "where","how","each","would","could","should","than","then","into",
             "your","our","its","you","they","them","these","those","some","any",
             "more","most","much","many","about","after","before","between",
             "during","without","because","also","just","very","well","such"}

def _fix_english_spacing(text: str) -> str:
    """Insert missing spaces between merged English words (OCR artifact)."""
    for word in sorted(_EN_WORDS, key=len, reverse=True):
        text = re.sub(rf'(?<!\w)({word})([a-z]{{2,}})(?!\w)', r'\1 \2', text)
        text = re.sub(rf'(?<!\w)({word})([A-Z])', r'\1 \2', text)
    return text


def _strip_markers(text: str) -> str:
    """Remove all [[PIC:id]] markers (for title/noise checking)."""
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

    # Track which type hints matched — used for ambiguity detection below
    _match_spec = _contains_any(normalized, SPECIFICATION_HINTS)
    _match_parts = _contains_any(normalized, PARTS_HINTS)

    if _match_spec:
        result = "specification"
    elif _match_parts:
        result = "parts_list"
    elif has_procedure:
        result = "procedure"
    elif has_safety:
        result = "safety_warning"
    else:
        return "general"

    # Keyword heuristics for parts_list are unreliable — generic words like
    # "包装", "配件", "组成" frequently appear in non-parts-list contexts
    # (e.g. waste disposal text, cleaning instructions).  Use LLM to verify
    # EVERY chunk that keyword wants to label as parts_list.
    if result == "parts_list":
        llm_result = _llm_classify_chunk_type(title, text)
        if llm_result:
            return llm_result

    return result


def _llm_classify_chunk_type(title: str, text: str) -> str | None:
    """Use LLM to resolve ambiguous chunk type classifications.

    Called when keyword heuristics match multiple types (e.g. both
    specification and parts_list hints).  Returns a type string on
    success, or *None* to fall back to the keyword-based result.
    """
    try:
        from industry_agent.llm.client import LLMClient

        # Keep the call cheap — 300 chars is enough context
        text_preview = text[:300].strip()
        prompt = (
            "You are classifying a product manual section. "
            "Reply with EXACTLY ONE word chosen from:\n"
            "- specification (technical specs, dimensions, parameters, weight, capacity)\n"
            "- parts_list (package contents, included items, components, accessories)\n"
            "- procedure (step-by-step instructions, how to install/remove/clean)\n"
            "- troubleshooting (error diagnosis, problem solving, fault finding)\n"
            "- safety_warning (warnings, cautions, danger, safety precautions)\n"
            "- general (none of the above)\n\n"
            f"Title: {title}\n"
            f"Text: {text_preview}\n\n"
            "Classification:"
        )

        client = LLMClient()
        result = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )

        result = result.strip().lower()
        valid = {
            "specification", "parts_list", "procedure",
            "troubleshooting", "safety_warning", "general",
        }
        for word in result.split():  # take first valid word
            word = word.strip(",.!;:\"'")
            if word in valid:
                return word
        return None
    except Exception:
        return None


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
