"""Agent orchestration: retrieve context -> build prompt -> call LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from industry_agent.agent.context_manager import ContextManager, TurnContext
from industry_agent.agent.customer_service_policy import CustomerServicePolicy
from industry_agent.agent.image_understanding import ImageUnderstandingResult, ImageUnderstander
from industry_agent.agent.question_splitter import SubQuestion, split_complex_question
from industry_agent.agent.question_router import QuestionRouter, RouteDecision
from industry_agent.agent.response_formatter import (
    format_customer_service_answer,
    format_manual_answer,
    format_multi_question_answer,
)
from industry_agent.agent.session_store import InMemorySessionStore, SessionState
from industry_agent.config import settings
from industry_agent.llm.client import LLMClient
from industry_agent.rag.retriever import SQLiteRetriever, analyze_query

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRIEVAL_LIMIT = 10        # chunks to retrieve before evidence filtering
FINAL_CONTEXT_CHUNKS = 4    # chunks passed into the LLM
MAX_CONTEXT_CHARS = 4000    # truncate context to fit model window
MAX_HISTORY_TURNS = 5       # keep last N turns per session
MIN_TOP_SCORE = 4.0         # below this, do not ask LLM to hallucinate
MIN_KEEP_SCORE = 3.0        # chunks below this score are discarded
MULTIMODAL_RETRIEVAL_LIMIT = 6

SYSTEM_TEMPLATE = """\
你是一个专业的产品客服智能体。请严格遵守以下规则：

1. **只基于下方【参考资料】回答**，不得编造。
2. 尽量从参考资料中提取对用户有帮助的内容，详细、完整地回答。只有参考资料完全不含相关信息时才说"根据现有资料无法回答此问题"。
3. 如果参考资料是英文的，或用户用英文提问，请用英文回答。
4. **图文结合**：参考资料中出现配图ID时，在关键步骤处插入 **最多3个** <PIC> 标记。不要堆砌大量<PIC>。
5. 回答结构要清晰：先给结论，再分步骤说明操作方法，最后列注意事项。用编号列表组织步骤。
6. 直接输出最终答案，不要输出思考过程、不要输出"结论："等标签。
7. **严禁重复**：同一句话、同一段内容绝对不能重复出现。每个句子只说一次。
8. 不要重复用户的问题。
9. 不要输出 <IMG src=""> 这样的空标签。
10. 不要输出 "请以实际产品型号和说明书原文为准" 之类的免责声明（除非资料确实不足）。

【参考资料】
{context}
"""

SUBQUESTION_MERGE_TEMPLATE = """\
请将下面多个子问题的回答合并成一个最终客服回复。要求：

1. 按“问题1 / 问题2 / 问题3”依次输出。
2. 每个问题都先直接回答，再补充必要说明。
3. 不要编造没有出现过的事实。
4. 如果某个子问题资料不足，就保留“根据现有资料无法回答此问题”。
5. 直接输出最终答案，不要输出思考过程。

【原始问题】
{original_question}

【子问题回答】
{sub_answers}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChatRequest:
    question: str
    images: list[str] | None = None
    session_id: str | None = None


@dataclass
class ChatResponse:
    answer: str
    image_ids: list[str]
    images: list[dict[str, str | bool]]
    sources: list[str]
    references: list[dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions (defined before the class that uses them)
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ANSWER_START_RE = re.compile(
    r"^([\u4e00-\u9fff]|#{1,3}\s|根据|您好|以下|关于|\d+[\.、])"
)
_NON_WORD_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_SMALLTALK_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "greeting",
        re.compile(r"^(你好|您好|hello|hi|hey|哈喽|嗨|早上好|中午好|下午好|晚上好)$", flags=re.IGNORECASE),
        "你好，我是工业产品客服智能体。你可以告诉我产品名称、型号、故障现象，或直接上传图片，我会尽量基于说明书资料帮你查询。",
    ),
    (
        "thanks",
        re.compile(r"^(谢谢|thanks|thankyou|thankyouverymuch|多谢|谢了)$", flags=re.IGNORECASE),
        "不客气。如果你愿意，可以继续告诉我具体的产品名称、型号、问题现象或上传图片，我来继续帮你查。",
    ),
    (
        "farewell",
        re.compile(r"^(再见|拜拜|bye|goodbye|回头见)$", flags=re.IGNORECASE),
        "再见。如果之后还有产品使用、安装、故障或配件相关问题，随时可以再来问我。",
    ),
)
_SERVICE_FOLLOW_UP_TERMS: tuple[str, ...] = (
    "那", "还", "还有", "需要", "准备", "材料", "多久", "几天", "怎么办",
    "可以吗", "能不能", "怎么申请", "怎么处理", "流程", "凭证", "证明",
    "谁承担", "联系谁", "审核", "下一步", "然后呢",
)


def _strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks that some models (qwen3) emit.

    Handles both:
      - <think>...</think> XML blocks
      - "Thinking Process:\n..." free-form prefix (stops at first Chinese or
        markdown answer section)
    """
    # 1. Strip <think>...</think> blocks
    text = _THINK_TAG_RE.sub("", text).strip()

    # 2. Strip "Thinking Process:" free-form prefix
    if text.startswith("Thinking"):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and _ANSWER_START_RE.match(stripped):
                preceding = "\n".join(lines[:i])
                if len(preceding) > 100:
                    text = "\n".join(lines[i:]).strip()
                    break

    return text


def _normalize_for_smalltalk(text: str) -> str:
    return _NON_WORD_RE.sub("", text.strip().lower())


def _match_smalltalk_reply(question: str) -> tuple[str, str] | None:
    normalized = _normalize_for_smalltalk(question)
    if not normalized:
        return None
    for intent, pattern, reply in _SMALLTALK_PATTERNS:
        if pattern.fullmatch(normalized):
            return intent, reply
    return None


def _filter_evidence(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep high-confidence, same-product evidence for cleaner prompts."""
    if not chunks:
        return []

    top = chunks[0]
    top_score = float(top.get("_score", 0.0))
    if top_score < MIN_TOP_SCORE:
        return []

    top_product = top.get("product_name", "")
    filtered: list[dict[str, Any]] = []
    for chunk in chunks:
        score = float(chunk.get("_score", 0.0))
        if score < MIN_KEEP_SCORE:
            continue
        if top_product and chunk.get("product_name") != top_product:
            continue
        filtered.append(chunk)
        if len(filtered) >= FINAL_CONTEXT_CHUNKS:
            break
    return filtered


def _text_overlap_count(text: str, terms: list[str]) -> int:
    normalized = re.sub(r"\s+", "", str(text).lower())
    count = 0
    for term in terms:
        token = re.sub(r"\s+", "", str(term).lower())
        if len(token) < 2:
            continue
        if token in normalized:
            count += 1
    return count


def _clean_cs_answer(text: str) -> str:
    """Clean up customer service LLM answers — remove repetition and garbage."""
    # Remove empty IMG tags
    text = re.sub(r'<IMG\s+src=""[^>]*/>', '', text)
    # Remove repeated identical sentences (keep first occurrence)
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        key = re.sub(r'\s+', '', s)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    result = ' '.join(unique)
    # Remove generic prefixes
    result = re.sub(r'^这类问题更适合按通用客服流程处理。?\s*', '', result)
    result = re.sub(r'\s{2,}', ' ', result).strip()
    return result if result else text.strip()


def _deduplicate_answer(text: str) -> str:
    """Remove duplicate sentences, empty IMG tags, and cap PIC markers from LLM output."""
    # Remove empty IMG tags
    text = re.sub(r'<IMG\s+src=""[^>]*/>', '', text)
    # Remove "请以实际产品型号和说明书原文为准" disclaimer
    text = re.sub(r'请以实际产品型号和说明书原文为准[。]?\s*', '', text)
    # Cap PIC markers at 3
    pic_count = text.count('<PIC>')
    if pic_count > 3:
        # Keep only first 3 PIC markers
        parts = text.split('<PIC>')
        text = '<PIC>'.join(parts[:4]) + ''.join(parts[4:]).replace('<PIC>', '')

    # Split into sentences and deduplicate
    sentences = re.split(r'(?<=[。！？.!?])\s+|(?<=[。！？.!?])$', text)
    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Normalize for comparison: strip whitespace and PIC markers
        key = re.sub(r'\s+', '', s)
        key = re.sub(r'<PIC>', '', key)
        key = re.sub(r'</?PIC>', '', key)
        if not key or len(key) < 4:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    result = ' '.join(unique)
    # Remove "（相关配图：...）" inline references
    result = re.sub(r'（相关配图：[^）]*）', '', result)
    result = re.sub(r'\s{2,}', ' ', result).strip()
    return result if result else text.strip()


def _is_ascii_heavy(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    return len(letters) >= 8 and len(letters) > len(cjk)


def _is_manual_fallback_answer(answer: str) -> bool:
    return "根据现有资料无法准确回答此问题" in answer or "根据现有资料无法回答此问题" in answer


def _build_extractive_manual_answer(
    *,
    query: str,
    evidence_chunks: list[dict[str, Any]],
    image_ids: list[str],
) -> str:
    """Build a conservative evidence-only answer when the LLM refuses English evidence."""

    if not evidence_chunks:
        return format_manual_answer("根据现有资料无法回答此问题。", image_ids=[])

    analysis = analyze_query(query)
    query_terms = _unique(
        [
            *analysis.phrases[:6],
            *analysis.keywords[:12],
            *analysis.models,
        ]
    )
    instructional_query = _looks_like_instructional_query(query)
    title_candidates: list[tuple[float, str]] = []
    sentence_candidates: list[tuple[float, str]] = []
    for chunk in evidence_chunks[:3]:
        title = _clean_evidence_text(str(chunk.get("title", "")))
        text = _clean_evidence_text(str(chunk.get("text", "")))
        if title and not _looks_like_toc_noise(title):
            title_score = _text_overlap_count(title, query_terms) + 2.0
            if instructional_query:
                title_score += _instructional_sentence_bonus(title)
            title_candidates.append((title_score, title))
        for sentence in _split_evidence_sentences(text):
            if _looks_like_toc_noise(sentence):
                continue
            score = _text_overlap_count(sentence, query_terms)
            if score <= 0 and len(sentence) < 80:
                continue
            if instructional_query:
                score += _instructional_sentence_bonus(sentence)
                score -= _low_value_instruction_sentence_penalty(sentence)
            sentence_candidates.append((float(score), sentence))

    title_candidates.sort(key=lambda item: (item[0], len(item[1]) <= 160), reverse=True)
    sentence_candidates.sort(key=lambda item: (item[0], len(item[1]) <= 220), reverse=True)
    selected: list[str] = []
    best_title = title_candidates[0][1] if title_candidates else ""
    for _, sentence in sentence_candidates:
        if instructional_query and re.search(r":\s*$", sentence):
            continue
        if (
            instructional_query
            and selected
            and _low_value_instruction_sentence_penalty(sentence) >= 3.0
        ):
            continue
        if best_title and _is_duplicate_evidence_sentence(sentence, [best_title]):
            if (
                (
                    len(sentence) > len(best_title) + 20
                    and _normalize_sentence_key(best_title) in _normalize_sentence_key(sentence)
                )
                or (
                    _looks_like_truncated_heading(best_title)
                    and _instructional_sentence_bonus(sentence) > 0
                )
            ):
                best_title = _clean_submission_style_sentence(sentence)
            continue
        if _is_duplicate_evidence_sentence(sentence, selected):
            continue
        selected.append(_clean_submission_style_sentence(sentence))
        if len(selected) >= 3:
            break

    if not selected:
        first = evidence_chunks[0]
        fallback_text = _clean_evidence_text(f"{first.get('title', '')} {first.get('text', '')}")[:320]
        selected = [_clean_submission_style_sentence(fallback_text)]

    conclusion = _compose_extractive_conclusion(
        title=best_title,
        sentence=selected[0] if selected else "",
        query=query,
    )
    details = [
        sentence
        for sentence in selected[1:]
        if not _is_duplicate_evidence_sentence(sentence, [conclusion])
    ]
    if not details and selected:
        details = [selected[0]] if not _is_duplicate_evidence_sentence(selected[0], [conclusion]) else []
    caution = ""
    if not _is_ascii_heavy(query):
        caution = "涉及安全操作时请以原说明书为准。"

    lines = [conclusion]
    if details:
        lines.append("")
        for i, item in enumerate(details, 1):
            lines.append(f"{i}. {item}")
    if caution:
        lines.extend(["", caution])
    # Insert <PIC> markers for image-text complementarity (cap at 3)
    if image_ids:
        lines.append("")
        lines.append("相关配图如下：" + "<PIC>" * min(len(image_ids), 3))
    return "\n".join(lines).strip()


def _clean_evidence_text(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"\[\[PIC[^\]]*\]\]", " ", cleaned)
    cleaned = cleaned.replace("#", " ")
    cleaned = re.sub(r"\\u[0-9a-fA-F]{4}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -|")


def _clean_submission_style_sentence(text: str) -> str:
    cleaned = _clean_evidence_text(text)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" -|")


def _split_evidence_sentences(text: str) -> list[str]:
    cleaned = _clean_evidence_text(text)
    if not cleaned:
        return []
    raw_parts = re.split(r"(?<=[。！？.!?;:])\s+|[\n\r]+", cleaned)
    sentences: list[str] = []
    for part in raw_parts:
        sentence = part.strip(" -|")
        if 18 <= len(sentence) <= 320:
            sentences.append(sentence)
    if not sentences and cleaned:
        sentences.append(cleaned[:320])
    return sentences


def _looks_like_toc_noise(text: str) -> bool:
    normalized = str(text)
    if normalized.count("...") >= 1:
        return True
    if len(re.findall(r"\bpage\b", normalized, flags=re.IGNORECASE)) >= 2:
        return True
    if len(re.findall(r"\b\d+\b", normalized)) >= 5 and len(normalized) < 180:
        return True
    return False


def _looks_like_truncated_heading(text: str) -> bool:
    cleaned = str(text).strip()
    if not cleaned or cleaned.endswith((".", "。", "!", "?", ":", ";")):
        return False
    tail = cleaned.rsplit(" ", 1)[-1].lower()
    return tail in {"support", "with", "to", "and", "or", "of", "for", "in", "the", "a"}


def _normalize_sentence_key(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "", str(text).lower())
    return normalized[:240]


def _is_duplicate_evidence_sentence(candidate: str, existing_items: list[str]) -> bool:
    candidate_key = _normalize_sentence_key(candidate)
    if not candidate_key:
        return True
    for item in existing_items:
        existing_key = _normalize_sentence_key(item)
        if not existing_key:
            continue
        if candidate_key == existing_key:
            return True
        if candidate_key in existing_key or existing_key in candidate_key:
            return True
    return False


def _compose_extractive_conclusion(*, title: str, sentence: str, query: str) -> str:
    clean_title = _clean_submission_style_sentence(title)
    clean_sentence = _clean_submission_style_sentence(sentence)
    if not clean_title:
        return clean_sentence
    if not clean_sentence:
        return clean_title
    if _is_duplicate_evidence_sentence(clean_title, [clean_sentence]):
        return clean_sentence if len(clean_sentence) >= len(clean_title) else clean_title
    if _looks_like_instructional_query(query) and _instructional_sentence_bonus(clean_title) > 0:
        return clean_title
    if _is_ascii_heavy(query) and clean_sentence.lower().startswith(
        ("this ", "these ", "it ", "they ", "the device ", "the labels ")
    ):
        return f"{clean_title}: {clean_sentence}"
    return clean_sentence


def _looks_like_instructional_query(text: str) -> bool:
    normalized = str(text).lower()
    return bool(re.search(r"\b(how|operate|operation|use|using|press|select|turn|open|close)\b", normalized))


def _instructional_sentence_bonus(text: str) -> float:
    normalized = str(text).lower()
    bonus = 0.0
    for pattern in (
        r"\bgo to\b",
        r"\bpress\b",
        r"\bselect\b",
        r"\bturn\b",
        r"\bopen\b",
        r"\bclose\b",
        r"\bmove\b",
        r"\benter\b",
        r"\binstall\b",
        r"\bremove\b",
    ):
        if re.search(pattern, normalized):
            bonus += 2.0
    return min(bonus, 6.0)


def _low_value_instruction_sentence_penalty(text: str) -> float:
    normalized = str(text).strip().lower()
    if normalized.startswith(("note:", "note ", "and it ", "it also ", "this device support")):
        return 3.0
    return 0.0


def _title_key(title: str) -> str:
    return re.sub(r"\s+", "", str(title).strip().lower())


def _rank_evidence_chunks(
    chunks: list[dict[str, Any]],
    *,
    query: str,
    image_terms: list[str] | None = None,
    image_features: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    analysis = analyze_query(query) if query.strip() else None
    image_terms = [term for term in (image_terms or []) if len(str(term).strip()) >= 2]
    image_features = image_features or {}
    component_terms = [term for term in image_features.get("component_terms", []) if len(str(term).strip()) >= 2]
    status_terms = [term for term in image_features.get("status_terms", []) if len(str(term).strip()) >= 2]
    issue_terms = [term for term in image_features.get("issue_terms", []) if len(str(term).strip()) >= 2]
    ranked: list[dict[str, Any]] = []
    for chunk in chunks:
        row = dict(chunk)
        base_score = float(row.get("_score", 0.0))
        title = str(row.get("title", ""))
        text = str(row.get("text", ""))
        image_ids = _parse_json_list(row.get("image_ids"))

        query_terms = []
        if analysis is not None:
            query_terms = [
                *analysis.products,
                *analysis.models,
                *analysis.phrases[:4],
                *analysis.expanded_keywords[:4],
                *analysis.keywords[:8],
            ]

        query_overlap = _text_overlap_count(f"{title}\n{text}", _unique(query_terms))
        image_overlap = _text_overlap_count(f"{title}\n{text}", image_terms)
        title_component_overlap = _text_overlap_count(title, component_terms)
        text_component_overlap = _text_overlap_count(text, component_terms)
        title_status_overlap = _text_overlap_count(title, status_terms)
        text_status_overlap = _text_overlap_count(text, status_terms)
        issue_overlap = _text_overlap_count(f"{title}\n{text}", issue_terms)

        evidence_score = base_score
        evidence_score += min(query_overlap * 1.1, 5.0)
        evidence_score += min(image_overlap * 1.8, 5.4)
        evidence_score += min(title_component_overlap * 1.8 + text_component_overlap * 0.9, 4.8)
        evidence_score += min(title_status_overlap * 1.2 + text_status_overlap * 1.5, 4.6)
        evidence_score += min(issue_overlap * 1.6, 3.2)
        if image_ids and image_overlap > 0:
            evidence_score += 1.0
        if (title_component_overlap + text_component_overlap) > 0 and (title_status_overlap + text_status_overlap) > 0:
            evidence_score += 2.0
        if issue_overlap > 0 and image_ids:
            evidence_score += 0.8
        if int(row.get("_variant_hits", 0)) >= 2:
            evidence_score += 1.5
        if image_overlap == 0 and image_terms and query_overlap <= 1 and not image_ids:
            evidence_score -= 1.2

        row["_evidence_score"] = round(evidence_score, 3)
        row["_query_overlap"] = query_overlap
        row["_image_overlap"] = image_overlap
        row["_image_component_overlap"] = title_component_overlap + text_component_overlap
        row["_image_status_overlap"] = title_status_overlap + text_status_overlap
        row["_image_issue_overlap"] = issue_overlap
        ranked.append(row)

    ranked.sort(
        key=lambda item: (
            float(item.get("_evidence_score", 0.0)),
            int(item.get("_image_component_overlap", 0)) + int(item.get("_image_status_overlap", 0)),
            int(item.get("_image_overlap", 0)),
            int(item.get("_query_overlap", 0)),
            len(_parse_json_list(item.get("image_ids"))),
        ),
        reverse=True,
    )
    return ranked


def _filter_evidence_for_query(
    chunks: list[dict[str, Any]],
    *,
    query: str,
    image_terms: list[str] | None = None,
    image_features: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Keep high-confidence evidence with light multimodal reranking and diversity."""
    ranked = _rank_evidence_chunks(
        chunks,
        query=query,
        image_terms=image_terms,
        image_features=image_features,
    )
    if not ranked:
        return []

    top = ranked[0]
    top_score = float(top.get("_evidence_score", 0.0))
    if top_score < MIN_TOP_SCORE:
        return []

    top_product = top.get("product_name", "")
    filtered: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for chunk in ranked:
        score = float(chunk.get("_evidence_score", 0.0))
        if score < MIN_KEEP_SCORE:
            continue
        if top_product and chunk.get("product_name") != top_product:
            continue
        title_key = _title_key(str(chunk.get("title", "")))
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        filtered.append(chunk)
        if len(filtered) >= FINAL_CONTEXT_CHUNKS:
            break
    return filtered


def _confidence_from_chunks(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.15
    top_score = float(chunks[0].get("_score", 0.0))
    second_score = float(chunks[1].get("_score", 0.0)) if len(chunks) > 1 else 0.0
    confidence = 0.35 + min(top_score / 50.0, 0.45) + min(max(top_score - second_score, 0.0) / 40.0, 0.15)
    return round(min(confidence, 0.95), 2)


def _merge_confidence(confidences: list[float]) -> float:
    if not confidences:
        return 0.15
    return round(sum(confidences) / len(confidences), 2)


def _assemble_context(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[str], list[str], list[dict[str, str]]]:
    """Build context string, collect image IDs, sources, and references."""
    parts: list[str] = []
    all_image_ids: list[str] = []
    seen_images: set[str] = set()
    sources: list[str] = []
    seen_sources: set[str] = set()
    references: list[dict[str, str]] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks, start=1):
        product = chunk.get("product_name", "")
        title = chunk.get("title", "")
        text = chunk.get("text", "")

        # Parse image_ids (stored as JSON string in SQLite)
        raw_img = chunk.get("image_ids", "[]")
        img_ids = _parse_json_list(raw_img)

        # Collect unique image IDs
        for img_id in img_ids:
            if img_id and img_id not in seen_images:
                seen_images.add(img_id)
                all_image_ids.append(img_id)

        # Collect unique sources
        if product and product not in seen_sources:
            seen_sources.add(product)
            sources.append(product)

        # Build context part
        score = chunk.get("_score", "")
        header = f"[参考{idx}] 产品：{product} | 章节：{title} | 检索分：{score}"
        body = text.strip()
        if img_ids:
            body += f"\n（相关配图：{', '.join(img_ids)}）"
        part = f"{header}\n{body}"

        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                parts.append(part[:remaining] + "……")
            break
        parts.append(part)
        total_chars += len(part)

        # Reference snippet for response metadata
        references.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "title": title,
            "text_snippet": text[:100],
            "product_name": product,
            "score": str(chunk.get("_score", "")),
        })

    context = "\n\n".join(parts)
    return context, all_image_ids, sources, references


def _parse_json_list(value: Any) -> list[str]:
    """Safely parse a JSON-encoded list or return as-is if already a list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _load_image_index() -> dict[str, dict[str, str | bool]]:
    """Load image metadata generated by the knowledge-base builder."""
    image_index_path = settings.processed_dir / "images.jsonl"
    records: dict[str, dict[str, str | bool]] = {}
    if not image_index_path.exists():
        return records
    with image_index_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            image_id = str(item.get("image_id", ""))
            if not image_id:
                continue
            records[image_id] = {
                "image_id": image_id,
                "file_name": item.get("file_name") or "",
                "path": item.get("path") or "",
                "exists": bool(item.get("exists", False)),
            }
    return records


def _image_details(image_ids: list[str], image_index: dict[str, dict[str, str | bool]]) -> list[dict[str, str | bool]]:
    details: list[dict[str, str | bool]] = []
    for image_id in image_ids:
        if image_id in image_index:
            details.append(image_index[image_id])
            continue
        fallback_path = settings.image_dir / f"{image_id}.jpg"
        details.append({
            "image_id": image_id,
            "file_name": fallback_path.name,
            "path": str(fallback_path.relative_to(settings.project_root)) if fallback_path.exists() else "",
            "exists": fallback_path.exists(),
        })
    return details


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _merge_images(image_groups: list[list[dict[str, str | bool]]]) -> list[dict[str, str | bool]]:
    merged: list[dict[str, str | bool]] = []
    seen: set[str] = set()
    for group in image_groups:
        for image in group:
            image_id = str(image.get("image_id", ""))
            if not image_id or image_id in seen:
                continue
            seen.add(image_id)
            merged.append(image)
    return merged


def _merge_retrieval_candidates(
    candidate_groups: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for variant_name, rows in candidate_groups:
        for rank, row in enumerate(rows):
            record = dict(row)
            chunk_id = str(record.get("chunk_id", ""))
            if not chunk_id:
                continue
            existing = merged.get(chunk_id)
            variant_bonus = max(0.0, 2.6 - rank * 0.25)
            if variant_name != "text_only":
                variant_bonus += 1.0
            if existing is None:
                record["_retrieval_variants"] = [variant_name]
                record["_variant_hits"] = 1
                record["_fusion_score"] = round(float(record.get("_score", 0.0)) + variant_bonus, 3)
                merged[chunk_id] = record
                continue

            existing_variants = list(existing.get("_retrieval_variants", []))
            if variant_name not in existing_variants:
                existing_variants.append(variant_name)
            existing["_retrieval_variants"] = existing_variants
            existing["_variant_hits"] = len(existing_variants)
            existing["_fusion_score"] = round(
                max(float(existing.get("_fusion_score", 0.0)), float(record.get("_score", 0.0)) + variant_bonus)
                + (1.2 if len(existing_variants) >= 2 else 0.0),
                3,
            )
            if float(record.get("_score", 0.0)) > float(existing.get("_score", 0.0)):
                for key, value in record.items():
                    if key.startswith("_"):
                        continue
                    existing[key] = value
                existing["_score"] = record.get("_score", existing.get("_score", 0.0))

    merged_rows = list(merged.values())
    merged_rows.sort(
        key=lambda item: (
            float(item.get("_fusion_score", 0.0)),
            int(item.get("_variant_hits", 0)),
            float(item.get("_score", 0.0)),
            len(_parse_json_list(item.get("image_ids"))),
        ),
        reverse=True,
    )
    return merged_rows


def _build_visual_focus_terms(image_features: dict[str, list[str]] | None) -> list[str]:
    image_features = image_features or {}
    return _unique(
        [
            *image_features.get("component_terms", []),
            *image_features.get("status_terms", []),
            *image_features.get("issue_terms", []),
        ]
    )[:MULTIMODAL_RETRIEVAL_LIMIT]


# ---------------------------------------------------------------------------
# Agent service
# ---------------------------------------------------------------------------

class AgentService:
    """Retrieve -> assemble context -> call LLM -> return answer."""

    def __init__(
        self,
        retriever: SQLiteRetriever | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.retriever = retriever or SQLiteRetriever()
        self.llm_client = llm_client or LLMClient()
        self.session_store = InMemorySessionStore(max_history_turns=MAX_HISTORY_TURNS)
        self.context_manager = ContextManager(max_history_turns=MAX_HISTORY_TURNS)
        self.question_router = QuestionRouter()
        self.customer_service_policy = CustomerServicePolicy()
        self.image_understander = ImageUnderstander(llm_client=self.llm_client)
        self.image_index = _load_image_index()

    def _build_subquestion_query(
        self,
        sub_question: SubQuestion,
        original_question: str,
        turn_context: TurnContext,
    ) -> str:
        """Build a retrieval query for one sub-question with session context."""

        return self.context_manager.build_subquestion_query(
            sub_question=sub_question,
            original_question=original_question,
            turn_context=turn_context,
        )

    def generate_response(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        image_input: str | None = None,
        dialog_summary: str | None = None,
        image_context: str | None = None,
        image_terms: list[str] | None = None,
        image_features: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        # 1. Retrieve
        candidate_groups: list[tuple[str, list[dict[str, Any]]]] = [
            ("text_only", self.retriever.search(query, limit=RETRIEVAL_LIMIT))
        ]
        multimodal_query = query
        if image_terms:
            multimodal_query = f"{query} {' '.join(image_terms[:MULTIMODAL_RETRIEVAL_LIMIT])}".strip()
            if multimodal_query != query:
                candidate_groups.append(
                    ("multimodal_fused", self.retriever.search(multimodal_query, limit=RETRIEVAL_LIMIT))
                )
        visual_focus_terms = _build_visual_focus_terms(image_features)
        if visual_focus_terms:
            visual_focus_query = f"{query} {' '.join(visual_focus_terms)}".strip()
            if visual_focus_query != query and visual_focus_query != multimodal_query:
                candidate_groups.append(
                    ("visual_focus", self.retriever.search(visual_focus_query, limit=RETRIEVAL_LIMIT))
                )

        chunks = _merge_retrieval_candidates(candidate_groups)
        evidence_chunks = _filter_evidence_for_query(
            chunks,
            query=query,
            image_terms=image_terms,
            image_features=image_features,
        )

        if not evidence_chunks:
            return {
                "answer": "根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。",
                "image_ids": [],
                "images": [],
                "sources": [],
                "references": [],
                "confidence": 0.15,
                "retrieval_debug": {
                    "retrieved_count": len(chunks),
                    "top_score": chunks[0].get("_score", 0) if chunks else 0,
                    "top_title": chunks[0].get("title", "") if chunks else "",
                    "query_variants": [name for name, _ in candidate_groups],
                    "image_terms": image_terms or [],
                    "image_features": image_features or {},
                    "reason": "low_confidence_or_no_evidence",
                },
            }

        # 2. Assemble context / collect metadata
        context, image_ids, sources, references = _assemble_context(evidence_chunks)
        images = _image_details(image_ids, self.image_index)
        confidence = _confidence_from_chunks(evidence_chunks)

        # 3. Build messages
        system_msg = SYSTEM_TEMPLATE.format(context=context if context else "（未找到相关资料）")
        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
        if dialog_summary:
            messages.append({"role": "system", "content": f"【会话上下文】\n{dialog_summary}"})
        if image_context:
            messages.append({"role": "system", "content": f"【用户上传图片信息】\n{image_context}"})

        # Append conversation history (if any)
        if history:
            messages.extend(history[-MAX_HISTORY_TURNS * 2 :])

        messages.append({"role": "user", "content": query})

        # 4. Call LLM
        answer = self._call_llm(messages)
        answer = _deduplicate_answer(answer)
        answer = format_manual_answer(answer, image_ids=image_ids)
        if _is_ascii_heavy(query) and _is_manual_fallback_answer(answer):
            answer = _build_extractive_manual_answer(
                query=query,
                evidence_chunks=evidence_chunks,
                image_ids=image_ids,
            )

        return {
            "answer": answer,
            "image_ids": image_ids,
            "images": images,
            "sources": sources,
            "references": references,
            "confidence": confidence,
            "retrieval_debug": {
                "retrieved_count": len(chunks),
                "evidence_count": len(evidence_chunks),
                "top_score": evidence_chunks[0].get("_score", 0) if evidence_chunks else 0,
                "top_evidence_score": evidence_chunks[0].get("_evidence_score", 0) if evidence_chunks else 0,
                "top_title": evidence_chunks[0].get("title", "") if evidence_chunks else "",
                "top_product": evidence_chunks[0].get("product_name", "") if evidence_chunks else "",
                "query_variants": [name for name, _ in candidate_groups],
                "image_terms": image_terms or [],
                "image_features": image_features or {},
                "candidate_titles": [str(chunk.get("title", "")) for chunk in chunks[:5]],
                "selected_titles": [str(chunk.get("title", "")) for chunk in evidence_chunks],
            },
        }

    def _merge_subquestion_answers(
        self,
        *,
        original_question: str,
        sub_questions: list[SubQuestion],
        sub_results: list[dict[str, Any]],
    ) -> str:
        """Merge several per-sub-question answers into one final reply."""

        if len(sub_results) == 1:
            return str(sub_results[0]["answer"])

        return format_multi_question_answer(
            [
                (sub_question.normalized_text, str(result["answer"]))
                for sub_question, result in zip(sub_questions, sub_results)
            ]
        )

    def _generate_customer_service_response(
        self,
        *,
        question: str,
        route_decision: RouteDecision,
        context_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        policy_response = self.customer_service_policy.answer(
            question,
            context_topics=context_topics,
        )

        # Send policy context through LLM for natural language generation
        # instead of returning raw template text
        cs_system = (
            '你是一个专业的产品客服人员。请根据以下【客服策略参考】，用自然、亲切的语气直接回答用户的问题。\n'
            '要求：\n'
            '1. 直接针对用户的具体问题回答，不要说"按通用客服流程处理"这类泛泛而谈的话。\n'
            '2. 回答要简洁具体，给出可操作的建议，不要重复相同内容。\n'
            '3. 不要输出思考过程，不要使用markdown格式。\n'
            '4. 不要重复用户的问题。\n\n'
            f'【客服策略参考】\n{policy_response.answer}'
        )

        messages = [
            {"role": "system", "content": cs_system},
            {"role": "user", "content": question},
        ]

        try:
            llm_answer = self._call_llm(messages)
        except Exception:
            llm_answer = policy_response.answer

        # Post-process: remove garbage patterns
        llm_answer = _clean_cs_answer(llm_answer)

        return {
            "answer": format_customer_service_answer(llm_answer),
            "image_ids": [],
            "images": [],
            "sources": ["customer_service_policy"],
            "references": [
                {
                    "chunk_id": f"policy_{topic}",
                    "title": "客服策略知识",
                    "text_snippet": question[:100],
                    "product_name": "customer_service_policy",
                    "score": str(route_decision.confidence),
                }
                for topic in policy_response.matched_topics
            ],
            "confidence": round(min(route_decision.confidence, policy_response.confidence), 2),
            "retrieval_debug": {
                "route": route_decision.route,
                "route_reason": route_decision.reason,
                "route_terms": route_decision.matched_terms,
                "matched_policy_topics": policy_response.matched_topics,
            },
        }

    def chat(self, request: ChatRequest) -> ChatResponse:
        """High-level chat with session memory."""
        smalltalk = _match_smalltalk_reply(request.question)
        if smalltalk is not None:
            intent, reply = smalltalk
            return ChatResponse(
                answer=reply,
                image_ids=[],
                images=[],
                sources=[],
                references=[],
                confidence=0.99,
                retrieval_debug={
                    "route": "smalltalk",
                    "intent": intent,
                },
            )

        session, turn_context = self._prepare_turn_context(request)
        if turn_context.context_reset_requested:
            if request.session_id:
                self.session_store.clear(request.session_id)
            return ChatResponse(
                answer="已清空本次会话上下文。你可以重新告诉我产品名称、型号、问题现象，或上传图片继续查询。",
                image_ids=[],
                images=[],
                sources=[],
                references=[],
                confidence=0.99,
                retrieval_debug={
                    "route": "session_control",
                    "action": "clear_context",
                    "session_id": request.session_id or "",
                },
            )

        if turn_context.needs_clarification:
            return ChatResponse(
                answer="我理解你可能想切换到另一个产品。请补充新的产品名称或型号后再问，我会避免沿用上一轮产品上下文。",
                image_ids=[],
                images=[],
                sources=[],
                references=[],
                confidence=0.55,
                retrieval_debug={
                    "route": "clarification",
                    "reason": turn_context.clarification_reason,
                    "session": {
                        "session_id": request.session_id or "",
                        "previous_product": session.current_product if session is not None else "",
                        "resolved_question": turn_context.resolved_question,
                    },
                },
            )

        image_result = self._analyze_uploaded_images(request)
        sub_questions = split_complex_question(request.question)
        if not sub_questions:
            sub_questions = [
                SubQuestion(
                    sub_question_id="q1",
                    text=request.question.strip(),
                    normalized_text=request.question.strip(),
                    intent="general",
                    depends_on_previous=False,
                )
            ]

        sub_results: list[dict[str, Any]] = []
        for sub_question in sub_questions:
            route_decision = self._resolve_route_decision(
                question=sub_question.normalized_text,
                session=session,
            )
            if route_decision.route == "customer_service":
                result = self._generate_customer_service_response(
                    question=sub_question.normalized_text,
                    route_decision=route_decision,
                    context_topics=session.current_service_topics if session is not None else [],
                )
                resolved_query = sub_question.normalized_text
            else:
                base_query = self._build_subquestion_query(
                    sub_question=sub_question,
                    original_question=request.question,
                    turn_context=turn_context,
                )
                resolved_query = base_query
                result = self.generate_response(
                    query=base_query,
                    history=turn_context.history,
                    image_input=request.images[0] if request.images else None,
                    dialog_summary=turn_context.dialog_summary,
                    image_context=image_result.combined_summary,
                    image_terms=image_result.retrieval_terms,
                    image_features=image_result.visual_features,
                )
            result["retrieval_debug"] = {
                **result.get("retrieval_debug", {}),
                "base_query": base_query if route_decision.route != "customer_service" else resolved_query,
                "resolved_query": resolved_query,
                "image_understanding": image_result.to_debug_dict(),
                "route_decision": {
                    "route": route_decision.route,
                    "confidence": route_decision.confidence,
                    "matched_terms": route_decision.matched_terms,
                    "manual_score": route_decision.manual_score,
                    "service_score": route_decision.service_score,
                    "reason": route_decision.reason,
                },
            }
            sub_results.append(result)

        merged_answer = self._merge_subquestion_answers(
            original_question=request.question,
            sub_questions=sub_questions,
            sub_results=sub_results,
        )
        merged_image_ids = _unique([
            image_id
            for result in sub_results
            for image_id in result["image_ids"]
        ])[:5]  # Cap at 5 images per answer
        merged_images = _merge_images([result["images"] for result in sub_results])
        merged_sources = _unique([
            source
            for result in sub_results
            for source in result["sources"]
        ])
        merged_references: list[dict[str, str]] = []
        for index, (sub_question, result) in enumerate(zip(sub_questions, sub_results), start=1):
            for reference in result["references"]:
                merged_references.append(
                    {
                        **reference,
                        "sub_question_id": sub_question.sub_question_id,
                        "sub_question_text": sub_question.normalized_text,
                        "sub_question_index": str(index),
                    }
                )
        merged_confidence = _merge_confidence([result["confidence"] for result in sub_results])
        merged_debug = {
            "session": {
                "session_id": request.session_id or "",
                "is_follow_up": turn_context.is_follow_up,
                "resolved_question": turn_context.resolved_question,
                "inherited_product": turn_context.inherited_product,
                "inherited_models": turn_context.inherited_models,
                "dialog_summary": turn_context.dialog_summary,
                "image_understanding": image_result.to_debug_dict(),
                "topic_switched": turn_context.topic_switched,
            },
            "sub_questions": [
                {
                    "sub_question_id": sub_question.sub_question_id,
                    "text": sub_question.text,
                    "normalized_text": sub_question.normalized_text,
                    "intent": sub_question.intent,
                    "depends_on_previous": sub_question.depends_on_previous,
                }
                for sub_question in sub_questions
            ],
            "sub_results": [
                {
                    "sub_question_id": sub_question.sub_question_id,
                    "confidence": result["confidence"],
                    "retrieval_debug": result["retrieval_debug"],
                }
                for sub_question, result in zip(sub_questions, sub_results)
            ],
        }

        # Update session
        if session is not None:
            session = self.context_manager.update_session(
                session=session,
                question=request.question,
                sub_questions=sub_questions,
                image_ids=merged_image_ids,
                sources=merged_sources,
                answer=merged_answer,
                turn_context=turn_context,
                uploaded_image_summary=image_result.combined_summary,
            )
            self._update_session_route_state(session=session, sub_results=sub_results)
            self.session_store.append_turn(
                session,
                user_question=request.question,
                assistant_answer=merged_answer,
            )

        return ChatResponse(
            answer=merged_answer,
            image_ids=merged_image_ids,
            images=merged_images,
            sources=merged_sources,
            references=merged_references,
            confidence=merged_confidence,
            retrieval_debug=merged_debug,
        )

    def _prepare_turn_context(
        self,
        request: ChatRequest,
    ) -> tuple[SessionState | None, TurnContext]:
        self._ensure_runtime_components()
        session: SessionState | None = None
        if request.session_id:
            session = self.session_store.get_or_create(request.session_id)
        turn_context = self.context_manager.resolve_turn(
            question=request.question,
            session=session,
        )
        return session, turn_context

    def _resolve_route_decision(
        self,
        *,
        question: str,
        session: SessionState | None,
    ) -> RouteDecision:
        route_decision = self.question_router.route(question)
        if route_decision.route == "customer_service":
            return route_decision
        if session is None or session.current_route not in {"customer_service", "mixed"}:
            return route_decision
        if not session.current_service_topics:
            return route_decision
        analysis = analyze_query(question)
        if analysis.products or analysis.models:
            return route_decision
        if not self._looks_like_customer_service_follow_up(question):
            return route_decision
        return RouteDecision(
            route="customer_service",
            confidence=max(route_decision.confidence, 0.72),
            matched_terms=session.current_service_topics[:3],
            manual_score=route_decision.manual_score,
            service_score=max(route_decision.service_score, 2),
            reason="inherit_customer_service_context",
        )

    def _looks_like_customer_service_follow_up(self, question: str) -> bool:
        normalized = re.sub(r"\s+", "", question.strip())
        if not normalized:
            return False
        if len(normalized) <= 14:
            return True
        return any(term in normalized for term in _SERVICE_FOLLOW_UP_TERMS)

    def _update_session_route_state(
        self,
        *,
        session: SessionState,
        sub_results: list[dict[str, Any]],
    ) -> None:
        route_names = [
            str(result.get("retrieval_debug", {}).get("route_decision", {}).get("route", ""))
            for result in sub_results
        ]
        service_topics = _unique(
            [
                topic
                for result in sub_results
                for topic in result.get("retrieval_debug", {}).get("matched_policy_topics", [])
            ]
        )
        if service_topics:
            session.current_route = "customer_service" if all(route == "customer_service" for route in route_names) else "mixed"
            session.current_service_topics = service_topics[:5]
            return
        if any(route == "manual_rag" for route in route_names):
            session.current_route = "manual_rag"
            session.current_service_topics = []

    def _ensure_runtime_components(self) -> None:
        if not hasattr(self, "session_store"):
            self.session_store = InMemorySessionStore(max_history_turns=MAX_HISTORY_TURNS)
        if not hasattr(self, "context_manager"):
            self.context_manager = ContextManager(max_history_turns=MAX_HISTORY_TURNS)
        if not hasattr(self, "question_router"):
            self.question_router = QuestionRouter()
        if not hasattr(self, "customer_service_policy"):
            self.customer_service_policy = CustomerServicePolicy()
        if not hasattr(self, "image_understander"):
            self.image_understander = ImageUnderstander(llm_client=self.llm_client)

    def _analyze_uploaded_images(self, request: ChatRequest) -> ImageUnderstandingResult:
        self._ensure_runtime_components()
        return self.image_understander.analyze_images(
            request.images or [],
            question=request.question,
        )

    # ------------------------------------------------------------------
    # LLM call — OpenAI-compatible API via LLMClient
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.llm_client.chat(
                messages,
                temperature=0.1,
                max_tokens=2048,
                strip_think=True,
            )
        except Exception as exc:
            return f"LLM 调用失败: {exc}"
