"""Agent orchestration: retrieve context -> build prompt -> call LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from industry_agent.agent.context_manager import ContextManager, TurnContext
from industry_agent.agent.customer_service_kb import CustomerServiceKnowledgeBase
from industry_agent.agent.customer_service_policy import CustomerServicePolicy
from industry_agent.agent.image_understanding import ImageUnderstandingResult, ImageUnderstander
from industry_agent.agent.prompts import (
    SUBQUESTION_MERGE_TEMPLATE,
    build_customer_service_system_prompt,
    build_manual_qa_system_prompt,
)
from industry_agent.agent.question_splitter import SubQuestion, split_complex_question
from industry_agent.agent.question_router import QuestionRouter, RouteDecision
from industry_agent.agent.response_formatter import (
    format_customer_service_answer,
    format_manual_answer,
    format_multi_question_answer,
)
from industry_agent.agent.session_store import InMemorySessionStore, SessionState
from industry_agent.agent.skills.image_skill import ImageSkill
from industry_agent.config import settings
from industry_agent.llm.client import LLMClient
from industry_agent.rag.factory import create_retriever
from industry_agent.rag.retriever import analyze_query

try:
    import httpx
except ImportError:  # pragma: no cover - optional for test environments
    httpx = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRIEVAL_LIMIT = 10        # chunks to retrieve before evidence filtering
FINAL_CONTEXT_CHUNKS = 5    # chunks passed into the LLM
MAX_CONTEXT_CHARS = 6000    # truncate context to fit model window
MAX_HISTORY_TURNS = 5       # keep last N turns per session
MIN_TOP_SCORE = 6.0         # below this, do not ask LLM to hallucinate
MIN_KEEP_SCORE = 4.0        # chunks below this score are discarded
MULTIMODAL_RETRIEVAL_LIMIT = 6

OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_model
OLLAMA_VISION_MODEL = settings.ollama_vision_model


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
_STRONG_MANUAL_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"根据现有资料无法准确回答此问题"),
    re.compile(r"根据现有资料无法回答此问题"),
    re.compile(r"Based on the available references,?\s+I cannot provide", flags=re.IGNORECASE),
    re.compile(r"The provided reference materials do not contain", flags=re.IGNORECASE),
    re.compile(r"The references only mention", flags=re.IGNORECASE),
)
_ENGLISH_INTERNAL_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Direct Conclusion\s*:", flags=re.IGNORECASE),
    re.compile(r"Details/Description\s*:", flags=re.IGNORECASE),
    re.compile(r"Operation/Steps\s*:", flags=re.IGNORECASE),
    re.compile(r"Notes?\s*:", flags=re.IGNORECASE),
)
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
_VISUAL_GROUNDING_TERMS: tuple[str, ...] = (
    "图", "图片", "配图", "图示", "示意图", "位置", "外观", "按钮", "按键",
    "接口", "插口", "指示灯", "灯", "闪烁", "部件", "零件", "表带", "尺寸",
    "更换", "拆卸", "安装", "连接", "屏幕", "显示",
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


def _is_ascii_heavy(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    return len(letters) >= 8 and len(letters) > len(cjk)


def _is_manual_fallback_answer(answer: str) -> bool:
    return "根据现有资料无法准确回答此问题" in answer or "根据现有资料无法回答此问题" in answer


def _looks_like_customer_service_llm_failure(answer: str, *, question: str) -> bool:
    normalized = _strip_thinking(str(answer)).strip()
    if not normalized:
        return True
    if normalized in {"模型未返回有效回答。"}:
        return True
    if normalized.startswith("LLM 调用失败:"):
        return True
    if any(token in normalized for token in ("【客服策略骨架】", "生成规则", "思考过程", "提示词")):
        return True
    if "根据现有资料无法" in normalized:
        return True
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) < 12:
        return True
    if _normalize_for_smalltalk(normalized) == _normalize_for_smalltalk(question):
        return True
    return False


def _should_use_extractive_manual_answer(answer: str) -> bool:
    normalized = _strip_thinking(str(answer)).strip()
    if not normalized:
        return True
    if _is_manual_fallback_answer(normalized):
        return True
    return any(pattern.search(normalized) for pattern in _STRONG_MANUAL_REFUSAL_PATTERNS)


def _should_force_extractive_manual_answer(query: str) -> bool:
    return _looks_like_instructional_query(query) or _looks_like_troubleshooting_query(query)


def _should_prefer_english_extractive_answer(answer: str, *, query: str) -> bool:
    if not _is_ascii_heavy(query):
        return False

    normalized = _strip_thinking(str(answer)).strip()
    if not normalized:
        return True
    if any(pattern.search(normalized) for pattern in _STRONG_MANUAL_REFUSAL_PATTERNS):
        return True
    if any(pattern.search(normalized) for pattern in _ENGLISH_INTERNAL_HEADING_PATTERNS):
        return True

    content = re.sub(r"(结论|操作/说明|注意事项|相关图片|无)", " ", normalized)
    content = re.sub(r"\s+", " ", content)
    return len(re.findall(r"[\u4e00-\u9fff]", content)) >= 6


def _manual_answer_needs_evidence_rescue(answer: str, *, query: str, evidence_chunks: list[dict[str, Any]]) -> bool:
    normalized = _strip_thinking(str(answer)).strip()
    if not normalized:
        return True
    if len(re.sub(r"\s+", "", normalized)) < 18:
        return True
    if any(pattern.search(normalized) for pattern in _STRONG_MANUAL_REFUSAL_PATTERNS):
        return True

    evidence_terms: list[str] = []
    analysis = analyze_query(query)
    evidence_terms.extend(analysis.products[:4])
    evidence_terms.extend(analysis.models[:4])
    evidence_terms.extend(analysis.phrases[:4])
    evidence_terms.extend(analysis.expanded_keywords[:6])
    evidence_terms.extend(analysis.keywords[:8])
    for chunk in evidence_chunks[:3]:
        evidence_terms.extend(_extract_keywords_from_text(str(chunk.get("title", ""))))
        evidence_terms.extend(_extract_keywords_from_text(str(chunk.get("text", ""))))
    evidence_terms = [term for term in _unique(evidence_terms) if len(term) >= 2]
    if not evidence_terms:
        return False

    overlap = _text_overlap_count(normalized, evidence_terms)
    if overlap >= 2:
        return False
    if _looks_like_instructional_query(query) and len(normalized) > 60:
        return True
    return overlap == 0


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
    troubleshooting_query = _looks_like_troubleshooting_query(query)
    title_candidates: list[tuple[float, str]] = []
    sentence_candidates: list[tuple[float, str]] = []
    for chunk in evidence_chunks[:3]:
        title = _clean_evidence_text(str(chunk.get("title", "")))
        text = _clean_evidence_text(str(chunk.get("text", "")))
        if title and not _looks_like_toc_noise(title):
            title_score = _text_overlap_count(title, query_terms) + 2.0
            if instructional_query:
                title_score += _instructional_sentence_bonus(title)
            if troubleshooting_query:
                title_score += _troubleshooting_sentence_bonus(title)
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
            if troubleshooting_query:
                score += _troubleshooting_sentence_bonus(sentence)
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

    lines = [conclusion]
    if details:
        lines.extend(details[:2])
    return "\n".join(lines).strip()


def _build_manual_answer_terms(query: str, answer: str) -> list[str]:
    analysis = analyze_query(query)
    terms = [
        *analysis.products[:4],
        *analysis.models[:4],
        *analysis.phrases[:6],
        *analysis.expanded_keywords[:8],
        *analysis.keywords[:10],
        *_extract_keywords_from_text(answer)[:12],
    ]
    return [term for term in _unique(terms) if len(str(term).strip()) >= 2]


def _should_ground_manual_images(
    *,
    query: str,
    answer: str,
    image_terms: list[str] | None,
    image_features: dict[str, list[str]] | None,
) -> bool:
    normalized_query = str(query)
    if any(term in normalized_query for term in _VISUAL_GROUNDING_TERMS):
        return True
    if image_terms:
        return True
    if image_features:
        for values in image_features.values():
            if any(str(value).strip() for value in values):
                return True
    normalized_answer = str(answer)
    return any(term in normalized_answer for term in _VISUAL_GROUNDING_TERMS)


def _select_grounded_manual_image_ids(
    *,
    query: str,
    answer: str,
    evidence_chunks: list[dict[str, Any]],
    image_terms: list[str] | None = None,
    image_features: dict[str, list[str]] | None = None,
) -> list[str]:
    if not evidence_chunks:
        return []

    answer_terms = _build_manual_answer_terms(query, answer)
    image_terms = [term for term in (image_terms or []) if len(str(term).strip()) >= 2]
    image_features = image_features or {}
    feature_terms = _unique(
        [
            *(image_features.get("component_terms", []) or []),
            *(image_features.get("status_terms", []) or []),
            *(image_features.get("issue_terms", []) or []),
        ]
    )
    should_ground = _should_ground_manual_images(
        query=query,
        answer=answer,
        image_terms=image_terms,
        image_features=image_features,
    )
    ranked_chunks: list[tuple[float, list[str]]] = []
    for chunk in evidence_chunks[:4]:
        image_ids = _parse_json_list(chunk.get("image_ids"))
        if not image_ids:
            continue
        chunk_text = f"{chunk.get('title', '')}\n{chunk.get('text', '')}"
        answer_overlap = _text_overlap_count(chunk_text, answer_terms)
        image_overlap = _text_overlap_count(chunk_text, image_terms)
        feature_overlap = _text_overlap_count(chunk_text, feature_terms)
        score = float(chunk.get("_evidence_score", chunk.get("_score", 0.0)))
        score += answer_overlap * 1.6
        score += image_overlap * 2.0
        score += feature_overlap * 1.8
        if answer_overlap > 0 and image_ids:
            score += 0.8
        if feature_overlap > 0 and image_ids:
            score += 0.8
        ranked_chunks.append((score, image_ids))

    if not ranked_chunks:
        return []

    ranked_chunks.sort(key=lambda item: item[0], reverse=True)
    if not should_ground and ranked_chunks[0][0] < MIN_KEEP_SCORE + 2.0:
        return []

    threshold = ranked_chunks[0][0] - (1.5 if should_ground else 0.8)
    selected: list[str] = []
    seen: set[str] = set()
    for score, image_ids in ranked_chunks:
        if score < threshold and selected:
            continue
        for image_id in image_ids:
            normalized = str(image_id).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            selected.append(normalized)
            if len(selected) >= 3:
                return selected
    return selected


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


def _extract_keywords_from_text(text: str) -> list[str]:
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", str(text))
    ascii_terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(text))
    return _unique([*cjk_terms[:10], *ascii_terms[:10]])


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
    return bool(
        re.search(r"\b(how|operate|operation|use|using|press|select|turn|open|close|install|remove|clean|set|check|adjust)\b", normalized)
        or re.search(r"(怎么|如何|步骤|方法|安装|更换|拆卸|清洁|设置|连接|使用|充电|佩戴|调节|检查|操作)", str(text))
    )


def _looks_like_troubleshooting_query(text: str) -> bool:
    normalized = str(text).lower()
    return bool(
        re.search(r"\b(meaning|mean|indicat|indicator|status|error|fault|warning|flash|flashing|blink|blinking|code)\b", normalized)
        or re.search(r"(什么意思|代表什么|含义|故障|闪烁|指示灯|报警|异常|无法|不工作|错误|报码|代码|状态)", str(text))
    )


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
        r"步骤",
        r"先",
        r"再",
        r"然后",
        r"最后",
        r"取下",
        r"装入",
        r"插入",
        r"连接",
        r"按下",
        r"点击",
        r"选择",
        r"确认",
        r"检查",
    ):
        if re.search(pattern, normalized):
            bonus += 2.0
    return min(bonus, 6.0)


def _troubleshooting_sentence_bonus(text: str) -> float:
    normalized = str(text).lower()
    bonus = 0.0
    for pattern in (
        r"\bmeans\b",
        r"\bindicates\b",
        r"\bstatus\b",
        r"\bwarning\b",
        r"\bflash(?:ing)?\b",
        r"\bblink(?:ing)?\b",
        r"表示",
        r"代表",
        r"含义",
        r"闪烁",
        r"指示灯",
        r"报警",
        r"错误",
        r"状态",
    ):
        if re.search(pattern, normalized):
            bonus += 1.5
    return min(bonus, 4.5)


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
        base_score = max(float(row.get("_score", 0.0)), float(row.get("_fusion_score", 0.0)))
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

    query_analysis = analyze_query(query)
    query_has_explicit_product = bool(query_analysis.products or query_analysis.models)
    preferred_products = {str(product) for product in query_analysis.products}
    top = ranked[0]
    if preferred_products:
        matched_anchor = next(
            (
                chunk for chunk in ranked
                if str(chunk.get("product_name", "")) in preferred_products
                and float(chunk.get("_evidence_score", 0.0)) >= MIN_KEEP_SCORE
            ),
            None,
        )
        if matched_anchor is not None:
            top = matched_anchor

    top_score = float(top.get("_evidence_score", 0.0))
    min_anchor_score = MIN_KEEP_SCORE if query_has_explicit_product else MIN_TOP_SCORE
    if top_score < min_anchor_score:
        return []

    top_product = top.get("product_name", "")
    top_query_overlap = int(top.get("_query_overlap", 0))
    top_image_overlap = int(top.get("_image_overlap", 0))
    filtered: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    cross_product_kept = 0
    for chunk in ranked:
        score = float(chunk.get("_evidence_score", 0.0))
        if score < MIN_KEEP_SCORE:
            continue
        same_product = not top_product or chunk.get("product_name") == top_product
        if not same_product:
            chunk_query_overlap = int(chunk.get("_query_overlap", 0))
            chunk_image_overlap = int(chunk.get("_image_overlap", 0))
            chunk_variant_hits = int(chunk.get("_variant_hits", 0))
            overlap_threshold = max(2, top_query_overlap)
            if query_has_explicit_product:
                continue
            if cross_product_kept >= 2:
                continue
            strong_query_alignment = chunk_query_overlap >= overlap_threshold
            strong_image_alignment = chunk_image_overlap > 0 and chunk_image_overlap >= top_image_overlap
            near_top_score = score >= top_score - 1.2
            variant_rescue = (
                chunk_variant_hits >= 2
                and chunk_query_overlap >= overlap_threshold
                and score >= top_score - 1.8
            )
            if not ((strong_query_alignment and near_top_score) or strong_image_alignment or variant_rescue):
                continue
        title_key = _title_key(str(chunk.get("title", "")))
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        filtered.append(chunk)
        if not same_product:
            cross_product_kept += 1
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
            "text_snippet": text[:320],
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
    """Retrieve -> assemble context -> call the configured LLM backend -> return answer."""

    def __init__(
        self,
        retriever: Any | None = None,
        base_url: str | None = None,
        model: str | None = None,
        llm_backend: str | None = None,
    ) -> None:
        self.retriever = retriever or create_retriever()
        self.llm_backend = (llm_backend or settings.llm_backend).strip().lower()
        self.model = model or (settings.ollama_model if self.llm_backend == "ollama" else settings.llm_model)
        resolved_base_url = base_url or (
            settings.ollama_base_url if self.llm_backend == "ollama" else settings.llm_base_url
        )
        self.base_url = resolved_base_url.rstrip("/")
        self.vision_model = (
            settings.ollama_vision_model if self.llm_backend == "ollama" else settings.llm_vision_model
        )
        self.session_store = InMemorySessionStore(max_history_turns=MAX_HISTORY_TURNS)
        self.context_manager = ContextManager(max_history_turns=MAX_HISTORY_TURNS)
        self.question_router = QuestionRouter()
        self.customer_service_policy = CustomerServicePolicy()
        self.customer_service_kb = CustomerServiceKnowledgeBase()
        if httpx is None and self.llm_backend == "ollama":
            raise RuntimeError("httpx is required to use AgentService with Ollama. Please install requirements.txt")
        self.http_client = httpx.Client(proxy=None, timeout=120.0) if httpx is not None else None
        self.llm_client = LLMClient(
            backend=self.llm_backend,
            base_url=self.base_url,
            model=self.model,
            vision_model=self.vision_model,
        )
        self.image_skill = ImageSkill(
            llm_backend=self.llm_backend,
            ollama_base_url=settings.ollama_base_url,
            ollama_vision_model=settings.ollama_vision_model,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
            llm_vision_model=settings.llm_vision_model,
        )
        self.image_understander = ImageUnderstander(
            base_url=settings.ollama_base_url,
            http_client=self.http_client,
            vision_model=OLLAMA_VISION_MODEL,
        )
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
                    "retrieval_status": self.retriever.retrieval_status()
                    if hasattr(self.retriever, "retrieval_status")
                    else {},
                    "image_terms": image_terms or [],
                    "image_features": image_features or {},
                    "reason": "low_confidence_or_no_evidence",
                },
            }

        # 2. Assemble context / collect metadata
        context, image_ids, sources, references = _assemble_context(evidence_chunks)
        confidence = _confidence_from_chunks(evidence_chunks)

        # 3. Build messages
        prompt_result = build_manual_qa_system_prompt(context)
        messages: list[dict[str, str]] = [{"role": "system", "content": prompt_result.content}]
        if dialog_summary:
            messages.append({"role": "system", "content": f"【会话上下文】\n{dialog_summary}"})
        if image_context:
            messages.append({"role": "system", "content": f"【用户上传图片信息】\n{image_context}"})

        # Append conversation history (if any)
        if history:
            messages.extend(history[-MAX_HISTORY_TURNS * 2 :])

        messages.append({"role": "user", "content": query})

        # 4. Call LLM
        extractive_answer = _build_extractive_manual_answer(
            query=query,
            evidence_chunks=evidence_chunks,
            image_ids=image_ids,
        )
        if _should_force_extractive_manual_answer(query):
            answer = extractive_answer
        else:
            answer = self._call_llm(messages)
            answer = format_manual_answer(answer, image_ids=[], compact=True)
            if _should_use_extractive_manual_answer(answer) or _should_prefer_english_extractive_answer(answer, query=query):
                answer = extractive_answer
            elif _manual_answer_needs_evidence_rescue(answer, query=query, evidence_chunks=evidence_chunks):
                answer = extractive_answer or answer

        grounded_image_ids = _select_grounded_manual_image_ids(
            query=query,
            answer=answer,
            evidence_chunks=evidence_chunks,
            image_terms=image_terms,
            image_features=image_features,
        )
        images = _image_details(grounded_image_ids, self.image_index)

        return {
            "answer": answer,
            "image_ids": grounded_image_ids,
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
                "retrieval_status": self.retriever.retrieval_status()
                if hasattr(self.retriever, "retrieval_status")
                else {},
                "retrieval_channels": _unique(
                    [
                        channel
                        for chunk in evidence_chunks
                        for channel in chunk.get("_retrieval_channels", [])
                    ]
                ),
                "image_terms": image_terms or [],
                "image_features": image_features or {},
                "candidate_image_ids": image_ids,
                "grounded_image_ids": grounded_image_ids,
                "candidate_titles": [str(chunk.get("title", "")) for chunk in chunks[:5]],
                "selected_titles": [str(chunk.get("title", "")) for chunk in evidence_chunks],
                "prompt": {
                    "rule_count": prompt_result.rule_count,
                    "has_context": prompt_result.has_context,
                    "anti_hallucination": True,
                },
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
        if not hasattr(self, "customer_service_kb"):
            self.customer_service_kb = CustomerServiceKnowledgeBase()
        policy_response = self.customer_service_policy.answer(
            question,
            context_topics=context_topics,
        )
        kb_hits = self.customer_service_kb.search(
            question,
            context_topics=[*policy_response.matched_topics, *(context_topics or [])],
            limit=4,
        )
        kb_context = self.customer_service_kb.build_context(kb_hits)
        prompt_context = (
            f"【客服策略骨架】\n{policy_response.answer}\n\n【客服知识参考】\n{kb_context}"
            if kb_context
            else policy_response.answer
        )
        prompt_result = build_customer_service_system_prompt(prompt_context)
        llm_messages = [
            {"role": "system", "content": prompt_result.content},
            {
                "role": "user",
                "content": (
                    "请基于上面的客服策略骨架和客服知识参考，直接回答下面这个用户问题。\n"
                    "优先吸收与当前场景最接近的客服知识条目，不要复述知识标题，不要编造新政策，不要输出 Markdown 标题。\n\n"
                    f"用户问题：{question}"
                ),
            },
        ]
        llm_answer = self._call_llm(llm_messages)
        used_policy_fallback = _looks_like_customer_service_llm_failure(llm_answer, question=question)
        final_answer = policy_response.answer if used_policy_fallback else llm_answer
        references = [
            {
                "chunk_id": f"policy_{topic}",
                "title": "客服策略知识",
                "text_snippet": question[:100],
                "product_name": "customer_service_policy",
                "score": str(route_decision.confidence),
            }
            for topic in policy_response.matched_topics
        ]
        references.extend(
            {
                "chunk_id": str(hit.get("entry_id", "")),
                "title": str(hit.get("title", "")),
                "text_snippet": str(hit.get("content", ""))[:320],
                "product_name": "customer_service_kb",
                "score": str(hit.get("score", "")),
            }
            for hit in kb_hits
        )
        sources = ["customer_service_policy"]
        if kb_hits:
            sources.append("customer_service_kb")
        return {
            "answer": format_customer_service_answer(final_answer),
            "image_ids": [],
            "images": [],
            "sources": sources,
            "references": references,
            "confidence": round(min(route_decision.confidence, policy_response.confidence), 2),
            "retrieval_debug": {
                "route": route_decision.route,
                "route_reason": route_decision.reason,
                "route_terms": route_decision.matched_terms,
                "matched_policy_topics": policy_response.matched_topics,
                "customer_service_kb": {
                    "hit_count": len(kb_hits),
                    "hit_titles": [str(hit.get("title", "")) for hit in kb_hits],
                    "hit_topics": [str(hit.get("topic", "")) for hit in kb_hits],
                    "hit_source_types": [str(hit.get("source_type", "")) for hit in kb_hits],
                },
                "customer_service_generation": {
                    "used_llm": True,
                    "used_policy_fallback": used_policy_fallback,
                    "prompt_rule_count": prompt_result.rule_count,
                    "has_policy_context": prompt_result.has_context,
                },
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
        turn_service_topics: list[str] = []
        for sub_question in sub_questions:
            route_decision = self._resolve_route_decision(
                question=sub_question.normalized_text,
                session=session,
                turn_service_topics=turn_service_topics,
            )
            if route_decision.route == "customer_service":
                result = self._generate_customer_service_response(
                    question=sub_question.normalized_text,
                    route_decision=route_decision,
                    context_topics=_unique(
                        [
                            *(session.current_service_topics if session is not None else []),
                            *turn_service_topics,
                        ]
                    ),
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
            turn_service_topics = _unique(
                [
                    *turn_service_topics,
                    *result["retrieval_debug"].get("matched_policy_topics", []),
                ]
            )

        merged_answer = self._merge_subquestion_answers(
            original_question=request.question,
            sub_questions=sub_questions,
            sub_results=sub_results,
        )
        merged_image_ids = _unique([
            image_id
            for result in sub_results
            for image_id in result["image_ids"]
        ])
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
        turn_service_topics: list[str] | None = None,
    ) -> RouteDecision:
        route_decision = self.question_router.route(question)
        if route_decision.route == "customer_service":
            return route_decision
        if turn_service_topics:
            analysis = analyze_query(question)
            if not (analysis.products or analysis.models) and self._looks_like_customer_service_follow_up(question):
                return RouteDecision(
                    route="customer_service",
                    confidence=max(route_decision.confidence, 0.74),
                    matched_terms=turn_service_topics[:3],
                    manual_score=route_decision.manual_score,
                    service_score=max(route_decision.service_score, 2),
                    reason="inherit_current_turn_customer_service_context",
                )
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
        if not hasattr(self, "llm_backend"):
            self.llm_backend = settings.llm_backend
        if not hasattr(self, "model"):
            self.model = settings.ollama_model if self.llm_backend == "ollama" else settings.llm_model
        if not hasattr(self, "base_url"):
            self.base_url = (
                settings.ollama_base_url if self.llm_backend == "ollama" else settings.llm_base_url
            ).rstrip("/")
        if not hasattr(self, "vision_model"):
            self.vision_model = (
                settings.ollama_vision_model if self.llm_backend == "ollama" else settings.llm_vision_model
            )
        if not hasattr(self, "llm_client"):
            self.llm_client = LLMClient(
                backend=self.llm_backend,
                base_url=self.base_url,
                model=self.model,
                vision_model=self.vision_model,
            )
        if not hasattr(self, "image_skill"):
            self.image_skill = ImageSkill(
                llm_backend=self.llm_backend,
                ollama_base_url=settings.ollama_base_url,
                ollama_vision_model=settings.ollama_vision_model,
                llm_base_url=settings.llm_base_url,
                llm_model=settings.llm_model,
                llm_vision_model=settings.llm_vision_model,
            )
        if not hasattr(self, "image_understander"):
            self.image_understander = ImageUnderstander(
                base_url=settings.ollama_base_url,
                http_client=getattr(self, "http_client", None),
                vision_model=OLLAMA_VISION_MODEL,
            )

    def _analyze_uploaded_images(self, request: ChatRequest) -> ImageUnderstandingResult:
        self._ensure_runtime_components()
        images = request.images or []
        if not images:
            return ImageUnderstandingResult(has_image_input=False)

        understander = getattr(self, "image_understander", None)
        if understander is not None and not isinstance(understander, ImageUnderstander):
            return understander.analyze_images(images, question=request.question)

        skill_result = self.image_skill.execute(images=images, question=request.question)
        if skill_result.success and isinstance(skill_result.data, ImageUnderstandingResult):
            return skill_result.data
        if understander is not None:
            return understander.analyze_images(images, question=request.question)
        return ImageUnderstandingResult(
            has_image_input=True,
            warnings=[skill_result.error or "图片理解组件不可用"],
        )

    # ------------------------------------------------------------------
    # LLM call — unified backend wrapper
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        try:
            content = self.llm_client.chat(
                messages,
                temperature=0.3,
                max_tokens=2048,
                strip_think=True,
            )
            return content.strip() if content.strip() else "模型未返回有效回答。"
        except Exception as exc:
            return f"LLM 调用失败: {exc}"
