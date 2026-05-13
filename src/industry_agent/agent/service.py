"""Agent orchestration: retrieve context -> build prompt -> call LLM."""

from __future__ import annotations

import json
import re
import os
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
from industry_agent.rag.query_expansion import QueryExpander
from industry_agent.rag.retriever import analyze_query

try:
    import httpx
except ImportError:  # pragma: no cover - optional for test environments
    httpx = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRIEVAL_LIMIT = 20        # chunks to retrieve before evidence filtering
FINAL_CONTEXT_CHUNKS = 8    # chunks passed into the LLM
MAX_CONTEXT_CHARS = 7000    # truncate context to fit model window
MAX_HISTORY_TURNS = 3       # keep last N turns per session
MIN_TOP_SCORE = 0.5         # below this, do not ask LLM to hallucinate
MIN_KEEP_SCORE = 0.4        # chunks below this score are discarded
MULTIMODAL_RETRIEVAL_LIMIT = 6
MAX_ANSWER_LENGTH = 500     # maximum answer length in characters
MAX_ENGLISH_ANSWER_LENGTH = 1000  # English needs more chars

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


def _is_english_text(text: str) -> bool:
    """Check if text is primarily English (vs Chinese)."""
    if not text or not text.strip():
        return False
    text_clean = text.strip()
    chinese_chars = len(re.findall(r'[一-鿿]', text_clean))
    english_words = len(re.findall(r'[a-zA-Z]{3,}', text_clean))
    return english_words >= 3 and chinese_chars < english_words


def _strip_llm_structured_format(text: str) -> str:
    """Strip structured format headers that some LLMs use.

    Transforms:
      结论：
      - 支持7天无理由退货。
      - 运费需自己承担。

      详情/说明：
      - xxx
    →
      支持7天无理由退货。运费需自己承担。
    """
    lines = text.strip().split("\n")
    cleaned_lines: list[str] = []
    in_structure = False
    for line in lines:
        stripped = line.strip()
        # Skip structured headers
        if stripped in ("结论：", "结论:", "目标：", "目标:"):
            in_structure = True
            continue
        if stripped.startswith(("详情", "说明", "Details", "Description")):
            in_structure = True
            # Also skip separator lines like "---" or "："
            continue
        # Skip section dividers
        if stripped in ("---", "－－－", "---", "") or re.match(r"^[—\-]{3,}$", stripped):
            continue
        # Strip leading bullet markers
        if in_structure:
            stripped = re.sub(r"^[-•·*]\s*", "", stripped)
        # Strip "# " manual heading markers anywhere they appear as headings
        # (e.g., "您好，# Note:" or "# Note:" or "，# Note:")
        stripped = re.sub(r"#+\s*", "", stripped).strip()
        cleaned_lines.append(stripped)
    result = "。".join(line for line in cleaned_lines if line)
    result = re.sub(r"。{2,}", "。", result)
    return result.strip() or text


def _is_raw_manual_text(text: str) -> bool:
    """Light check for raw manual text — trust the LLM more, reject only obvious garbage."""
    if not text or len(text.strip()) < 10:
        return True

    text_stripped = text.strip()

    # Reject obvious heading-only content
    if re.match(r'^#+\s', text_stripped):
        return True
    if re.match(r'^[A-Z\s]{12,}$', text_stripped[:60].strip()):
        return True

    # Reject "CAUTION"/"WARNING"/"IMPORTANT" manual labels at start
    if re.match(r'^(CAUTION|WARNING|IMPORTANT|NOTE|注意|警告)\b', text_stripped, re.IGNORECASE):
        return True

    # Reject page references
    if re.search(r"第\s*\d+[\s、，,，]*\d*\s*页", text_stripped):
        return True

    # Reject parts-list patterns
    if re.match(r'^\d+[\.\)]\s+\w{2,8}\s+\d+', text_stripped):
        return True

    # Reject obvious manual markers
    if ">>>" in text_stripped[:80]:
        return True

    # Accept everything else — the LLM generated it, trust it
    return False


def _clean_manual_markers(text: str) -> str:
    """Strip manual text markers to produce clean, readable text."""
    cleaned = text.strip()
    # Remove "#" headers
    cleaned = re.sub(r"^#\s*", "", cleaned)
    cleaned = re.sub(r"\n#\s*", "\n", cleaned)
    # Remove "C " markers (manual section connectors)
    cleaned = re.sub(r"\s+C\s+", " ", cleaned)
    # Remove ">>>" markers
    cleaned = re.sub(r"\s*>>>\s*", " ", cleaned)
    # Remove "・" bullet markers
    cleaned = cleaned.replace("・", "")
    # Remove "注：" and "注:" markers
    cleaned = re.sub(r"注[：:]\s*", "", cleaned)
    # Remove Roman numeral section markers (IX., VIII., IV., etc.)
    cleaned = re.sub(r"\b[IVX]+\.\s*", "", cleaned)
    # Remove page references (handles "第 23、24 页", "第5页", "见第5页")
    cleaned = re.sub(r"第\s*\d+[\s、，,，]*\d*\s*页", "", cleaned)
    cleaned = re.sub(r"见第\s*\d+.*?页.*?部分", "", cleaned)
    cleaned = re.sub(r"详见.*?章节", "", cleaned)
    # Remove image references
    cleaned = re.sub(r"\(相关配图：[^)]*\)", "", cleaned)
    # Remove manual-style line breaks
    cleaned = re.sub(r"\n{2,}", "。", cleaned)
    cleaned = re.sub(r"\n", "", cleaned)
    # Clean up extra spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" -|。")


def _chunk_is_low_quality(chunk: dict) -> bool:
    """Check if a chunk is low quality (parts list, TOC, overview) and should be skipped."""
    meta = chunk.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    chunk_type = meta.get("chunk_type", "")
    if chunk_type in ("parts_list", "toc"):
        return True
    is_toc = meta.get("is_toc", False)
    if is_toc:
        return True
    # Check if text is a parts list pattern (space-separated short items)
    text = str(chunk.get("text", ""))[:120]
    if " " in text and "，" not in text:
        items = [item.strip() for item in text.split(" ") if item.strip()]
        if len(items) >= 4 and all(len(item) < 8 for item in items):
            return True
    return False


def _extract_answer_relevance_terms(query: str) -> list[str]:
    """Extract meaningful question terms for relevance scoring (skip stopwords)."""
    query_lower = query.lower()
    terms = []
    _QW_STOP = frozenset(("您好", "什么", "如何", "怎么", "哪些", "为什么", "是否",
                          "可以", "需要", "请问", "一下", "关于", "这个", "那个", "一个",
                          "哪个", "多少", "怎样", "能", "没有", "还是", "不是", "就是"))
    for term in re.findall(r'[一-鿿]{2,}', query_lower):
        if term not in _QW_STOP:
            terms.append(term)
    for term in re.findall(r'[A-Za-z][A-Za-z0-9_-]{2,}', query_lower):
        if term not in ('how', 'what', 'when', 'where', 'why', 'which',
                        'the', 'and', 'for', 'can', 'you', 'your', 'are'):
            terms.append(term)
    return terms


def _build_conversational_answer(query: str, evidence_chunks: list, image_ids: list) -> str:
    """Build a question-aware conversational answer from the best evidence chunk.

    Strategy: extract question keywords → score each chunk by keyword overlap →
    pick best chunk → find sentences that contain question keywords → return.
    """
    if not evidence_chunks:
        return "根据现有资料，暂时无法提供更详细的信息。"

    query_terms = _extract_answer_relevance_terms(query)

    # Score each chunk by question-relevance
    chunk_scores: list[tuple[float, dict, list[str]]] = []
    for chunk in evidence_chunks[:5]:
        if _chunk_is_low_quality(chunk):
            continue
        text = str(chunk.get("text", ""))
        title = str(chunk.get("title", ""))
        combined = (title + " " + text).lower()

        overlap = sum(1 for t in query_terms if t in combined)
        title_overlap = sum(1 for t in query_terms if t in title.lower())

        # Find question-relevant sentences
        raw_sentences = re.split(r"[。！？]", text)
        relevant: list[tuple[int, str]] = []
        for s in raw_sentences:
            s = s.strip()
            if not s or len(s) < 10:
                continue
            s_lower = s.lower()
            s_overlap = sum(1 for t in query_terms if t in s_lower)
            if s_overlap > 0:
                relevant.append((s_overlap, s))

        relevant.sort(key=lambda x: x[0], reverse=True)
        best_sentence_overlap = relevant[0][0] if relevant else 0
        # Score: content overlap + 2x title overlap + best sentence bonus
        score = float(overlap) + float(title_overlap) * 2.0 + float(best_sentence_overlap) * 0.5
        # Boost by evidence score (from retriever)
        score += float(chunk.get("_evidence_score", chunk.get("_score", 0))) * 0.1

        chunk_scores.append((score, chunk, [s for _, s in relevant]))

    if not chunk_scores:
        # All chunks low-quality — use first chunk raw
        chunk_scores = [(0.0, evidence_chunks[0], [])]

    chunk_scores.sort(key=lambda x: (x[0], -len(x[2])), reverse=True)
    _, best_chunk, relevant_sents = chunk_scores[0]

    text = str(best_chunk.get("text", ""))
    text = _clean_manual_markers(text)

    # Use question-relevant sentences if available
    if relevant_sents:
        selected = relevant_sents[:4]
    else:
        # Fallback: filter substantive sentences
        raw_sentences = re.split(r"[。！？]", text)
        selected = []
        for s in raw_sentences:
            s = s.strip()
            if not s or len(s) < 8:
                continue
            if len(s) < 15 and not any(c in s for c in "，。、："):
                continue
            if re.match(r"^(图\d+|第\d+页|注[：:])", s):
                continue
            if re.match(r"^[一-鿿\s]+$", s) and len(s.split()) > 3:
                continue
            selected.append(s)
            if len(selected) >= 4:
                break

    if not selected:
        title = _clean_manual_markers(str(best_chunk.get("title", "")))
        return f"{title}。"

    # Clean selected sentences
    cleaned = []
    for s in selected:
        s = re.sub(r"^[一-鿿]{2,6}[（）\(\)\d]*\s+", "", s)
        s = re.sub(r"^\d+[\.\、]\s*", "", s)
        # Strip leading "2 " type numbering
        s = re.sub(r"^\d+\s+", "", s)
        if s and len(s) >= 8:
            cleaned.append(s)

    answer_body = "。".join(cleaned[:4])
    if not answer_body.endswith(("。", "！", "？")):
        answer_body += "。"

    return answer_body


def _validate_answer_grounding(context: str, answer: str) -> tuple[bool, float]:
    """Validate if the answer is grounded in the context (not hallucinated).

    Returns:
        Tuple of (is_grounded, grounding_score)
    """
    if not context or not answer:
        return True, 0.0

    # Simple check: if answer is very short, it's likely grounded
    if len(answer) < 50:
        return True, 0.8

    # Extract key terms from context (2+ character Chinese words or 3+ character English words)
    context_lower = context.lower()
    answer_lower = answer.lower()

    # Get Chinese terms from context
    context_terms = set()
    for term in re.findall(r'[一-鿿]{2,}', context_lower):
        context_terms.add(term)

    # Get Chinese terms from answer
    answer_terms = set()
    for term in re.findall(r'[一-鿿]{2,}', answer_lower):
        answer_terms.add(term)

    if not answer_terms:
        return True, 0.5

    # Calculate overlap
    overlap = context_terms & answer_terms
    overlap_ratio = len(overlap) / len(answer_terms) if answer_terms else 0

    # Grounding score based on overlap
    if overlap_ratio >= 0.25:
        grounding_score = 1.0
    elif overlap_ratio >= 0.15:
        grounding_score = 0.7
    elif overlap_ratio >= 0.08:
        grounding_score = 0.4
    else:
        grounding_score = 0.0

    return grounding_score >= 0.15, grounding_score


def _validate_topic_relevance(query: str, answer: str) -> tuple[bool, float]:
    """Validate if the answer is actually relevant to the question.

    Returns:
        Tuple of (is_relevant, relevance_score)
    """
    if not query or not answer:
        return True, 0.0

    # Simple check: if answer is very short, it's likely relevant
    if len(answer) < 50:
        return True, 0.8

    query_lower = query.lower()
    answer_lower = answer.lower()

    # Extract question keywords
    query_terms = set()
    for term in re.findall(r'[一-鿿]{2,}', query_lower):
        query_terms.add(term)

    # Extract answer keywords
    answer_terms = set()
    for term in re.findall(r'[一-鿿]{2,}', answer_lower):
        answer_terms.add(term)

    if not query_terms:
        return True, 0.5

    # Calculate overlap
    overlap = query_terms & answer_terms
    overlap_ratio = len(overlap) / len(query_terms) if query_terms else 0

    # Intent-specific validation: check if the answer addresses the core intent
    intent_score = 1.0

    # Question intent patterns - each maps to required answer keywords
    intent_map = {
        ("哪些", "什么物品", "不适合", "不宜", "不能"): ["不", "禁止", "不得", "避免", "不宜", "不适合"],
        ("附件", "配件", "配备", "包含"): ["附件", "配件", "配备", "包含", "标配"],
        ("规格", "参数", "技术"): ["规格", "参数", "尺寸", "重量", "功率", "电压", "容量"],
        ("位置", "佩戴", "戴在"): ["位置", "佩戴", "戴在", "穿戴", "放置"],
        ("保修", "保修期", "质保"): ["保修", "质保", "维护", "维修", "服务"],
        ("启动", "开机", "怎么开"): ["启动", "开机", "开启", "打开"],
        ("关闭", "关机", "怎么关"): ["关闭", "关机", "停机", "断电"],
        ("清洁", "清洗", "保养"): ["清洁", "清洗", "保养", "维护"],
        ("安装", "装配", "怎么装"): ["安装", "装配", "设置", "装入"],
        ("安全", "注意", "危险"): ["安全", "注意", "防护", "禁止", "警告"],
        ("故障", "问题", "异常", "报错"): ["故障", "问题", "异常", "报错", "维修"],
        ("充电", "电池"): ["充电", "电池", "电源", "电量"],
        ("显示", "屏幕", "界面"): ["显示", "屏幕", "界面", "指示灯"],
        ("运动", "锻炼", "训练"): ["运动", "锻炼", "训练", "健身", "心率"],
        ("追踪", "监测", "记录"): ["追踪", "监测", "记录", "数据", "睡眠"],
    }

    for intent_keywords, required_keywords in intent_map.items():
        if any(kw in query_lower for kw in intent_keywords):
            if not any(rk in answer_lower for rk in required_keywords):
                intent_score *= 0.3  # Heavy penalty if intent not addressed
            break

    # Relevance score based on overlap and intent
    if overlap_ratio >= 0.3:
        relevance_score = 1.0
    elif overlap_ratio >= 0.2:
        relevance_score = 0.7
    elif overlap_ratio >= 0.1:
        relevance_score = 0.4
    else:
        relevance_score = 0.0

    final_score = relevance_score * intent_score
    return final_score >= 0.2, final_score


def _extractive_fallback_with_topic(query: str, evidence_chunks: list, image_ids: list) -> str:
    """Build an extractive answer that stays on topic."""
    if not evidence_chunks:
        return "您好，根据现有资料，暂时无法提供更详细的信息。如需帮助随时联系我们。"

    # Extract question intent
    query_lower = query.lower()
    intent_keywords = []

    if any(kw in query_lower for kw in ["关闭", "停止", "关掉", "停机"]):
        intent_keywords = ["关闭", "停止", "关机", "断电", "停机"]
    elif any(kw in query_lower for kw in ["开启", "打开", "启动"]):
        intent_keywords = ["开启", "打开", "启动", "开机"]
    elif any(kw in query_lower for kw in ["安装", "装配"]):
        intent_keywords = ["安装", "装配", "设置"]
    elif any(kw in query_lower for kw in ["清洁", "清洗", "保养"]):
        intent_keywords = ["清洁", "清洗", "保养", "维护"]
    elif any(kw in query_lower for kw in ["安全", "注意"]):
        intent_keywords = ["安全", "注意", "防护", "禁止"]
    elif any(kw in query_lower for kw in ["维修", "修理"]):
        intent_keywords = ["维修", "修理", "故障"]
    elif any(kw in query_lower for kw in ["规格", "参数"]):
        intent_keywords = ["规格", "参数", "尺寸", "重量"]
    elif any(kw in query_lower for kw in ["配件", "附件"]):
        intent_keywords = ["配件", "附件", "部件"]

    # Find relevant sentences from evidence
    relevant_sentences = []
    for chunk in evidence_chunks[:3]:
        text = str(chunk.get("text", ""))
        sentences = re.split(r"[。！？]", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            if any(kw in sentence for kw in intent_keywords):
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= 2:
                    break
        if len(relevant_sentences) >= 2:
            break

    if relevant_sentences:
        answer = "您好，" + "。".join(relevant_sentences[:2]) + "。如需帮助随时联系我们。"
    else:
        # Fallback to first chunk's title
        first_title = str(evidence_chunks[0].get("title", ""))
        answer = f"您好，{first_title}。如需帮助随时联系我们。"

    return answer


def _normalize_for_smalltalk(text: str) -> str:
    return _NON_WORD_RE.sub("", text.strip().lower())


def _localize_answer(answer: str, query: str) -> str:
    """Convert Chinese greeting to English if the query is in English."""
    if not answer or len(answer) < 5:
        return answer
    english_words = len(re.findall(r'[a-zA-Z]{3,}', query))
    chinese_chars = len(re.findall(r'[一-鿿]', query))
    if english_words < 3 or chinese_chars >= english_words:
        return answer  # Keep as Chinese

    # Query is English — fix the answer greeting
    for cn_greeting, en_greeting in [
        ("您好，", "Hello, "),
        ("您好！", "Hello! "),
        ("你好，", "Hello, "),
        ("你好！", "Hello! "),
    ]:
        if answer.startswith(cn_greeting):
            answer = en_greeting + answer[len(cn_greeting):]
            break

    # Replace Chinese periods with English periods for English answers
    answer = answer.replace("。", ".")

    # Ensure single trailing period
    answer = answer.rstrip(".") + "."

    return answer


def _final_answer_cleanup(answer: str) -> str:
    """Final cleanup pass to remove manual text artifacts from the answer."""
    if not answer:
        return answer
    cleaned = answer.strip()
    # Remove bullet markers (•, ・, -)
    cleaned = cleaned.replace("•", "").replace("・", "").strip()
    # Remove "#" headers (e.g., "# 维修" → "维修")
    cleaned = re.sub(r"^#\s*", "", cleaned)
    cleaned = re.sub(r"\n#\s*", "\n", cleaned)
    # Remove Roman numeral section markers (IX., VIII., IV., etc.)
    cleaned = re.sub(r"\b[IVX]+\.\s*", "", cleaned)
    # Remove "第X页" references (handles "第 23、24 页", "第5页", "见第5页")
    cleaned = re.sub(r"第\s*\d+[\s、，,，]*\d*\s*页", "", cleaned)
    cleaned = re.sub(r"见第\s*\d+.*?页.*?部分", "", cleaned)
    cleaned = re.sub(r"详见.*?章节", "", cleaned)
    # Remove "请参阅/请查阅/请参考" references to manual sections
    cleaned = re.sub(r"请参阅[^。\n]*[。]?", "", cleaned)
    cleaned = re.sub(r"请查阅[^。\n]*[。]?", "", cleaned)
    cleaned = re.sub(r"请参考[^。\n]*[。]?", "", cleaned)
    cleaned = re.sub(r"详情请参阅[^。\n]*[。]?", "", cleaned)
    cleaned = re.sub(r"参考\d+产品[^。\n]*[。]?", "", cleaned)
    # Remove "相关配图：..." at the end
    cleaned = re.sub(r"[（(]相关配图：[^）)]*[）)]\s*$", "", cleaned)
    cleaned = re.sub(r"\(相关配图：[^)]*\)\s*$", "", cleaned)
    # Remove "■" bullets
    cleaned = cleaned.replace("■", "").strip()
    # Remove leading step numbers after punctuation (e.g., "。2 " → "。")
    cleaned = re.sub(r"[。]\s*\d+\s+", "。", cleaned)
    # Remove step numbers between Chinese periods (e.g., "。3。" → "。")
    cleaned = re.sub(r"。\s*\d+。", "。", cleaned)
    # Remove step numbers with English period before Chinese period (e.g., "。 3.。" → "。")
    cleaned = re.sub(r"。\s*\d+\.。", "。", cleaned)
    # Remove leading numbers at start of answer (e.g., "2 打开" → "打开")
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    # Remove "结论：" and "目标：" headings
    cleaned = re.sub(r"结论[：:]\s*", "", cleaned)
    cleaned = re.sub(r"目标[：:]\s*", "", cleaned)
    # Remove "Note:" label mid-answer
    cleaned = re.sub(r"\s+Note:\s*", " ", cleaned)
    # Remove step/section headings like "操作说明 4." or "步骤说明 2."
    cleaned = re.sub(r"(?:操作|步骤)说明\s*\d+[\.\s]*", "", cleaned)
    # Clean up extra spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # Remove trailing punctuation artifacts
    cleaned = re.sub(r"[，,。.、]+\s*$", "", cleaned)
    # Ensure it ends with proper punctuation
    if cleaned and not cleaned.endswith(("。", "！", "？", "）", ")", ".")):
        cleaned += "。"
    # Remove double periods (both Chinese and English)
    cleaned = cleaned.replace("。。", "。")
    cleaned = cleaned.replace("。。", "。")
    cleaned = cleaned.replace(",.", ".")
    cleaned = cleaned.replace("。,", "。")
    cleaned = cleaned.replace("。.", "。")
    cleaned = cleaned.replace(".。", ".")
    cleaned = cleaned.replace("。。", "。")
    cleaned = cleaned.replace("，，", "，")
    return cleaned.strip()


def _normalize(text: str) -> str:
    """Lowercase and strip whitespace for text comparison."""
    return text.strip().lower()


def _match_smalltalk_reply(question: str) -> tuple[str, str] | None:
    normalized = _normalize_for_smalltalk(question)
    if not normalized:
        return None
    for intent, pattern, reply in _SMALLTALK_PATTERNS:
        if pattern.fullmatch(normalized):
            return intent, reply
    return None


def _filter_evidence(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep high-confidence evidence with product-diversity awareness."""
    if not chunks:
        return []

    top = chunks[0]
    top_score = float(top.get("_score", 0.0))
    if top_score < MIN_TOP_SCORE:
        return []

    # Keep top chunks regardless of product, prefer same-product but allow cross-product
    top_product = top.get("product_name", "")
    filtered: list[dict[str, Any]] = []
    cross_product_count = 0
    for chunk in chunks:
        score = float(chunk.get("_score", 0.0))
        if score < MIN_KEEP_SCORE:
            continue
        same_product = not top_product or chunk.get("product_name") == top_product
        if not same_product:
            if cross_product_count >= 5:
                continue
            cross_product_count += 1
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
    if len(re.sub(r"\s+", "", normalized)) < 6:
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
    for chunk in evidence_chunks[:5]:
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
            # Skip raw manual labels (CAUTION, WARNING, etc.)
            if re.match(r'^(CAUTION|WARNING|IMPORTANT|NOTE|注意|警告)\b', sentence.strip(), re.IGNORECASE):
                continue
            # Skip incomplete sentences
            if sentence.strip().endswith((",", "and", "the", "with", "for", "of", "in")):
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
        if len(selected) >= 5:
            break

    if not selected:
        first = evidence_chunks[0]
        fallback_text = _clean_evidence_text(f"{first.get('title', '')} {first.get('text', '')}")[:320]
        # Strip raw manual labels from fallback
        fallback_text = re.sub(r'^(CAUTION|WARNING|IMPORTANT|NOTE|注意|警告)\s*', '', fallback_text, flags=re.IGNORECASE).strip()
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

    # If conclusion is a manual-style heading like "目标：" or "产品部件介绍", skip it
    _HEADING_LIKE_RE = re.compile(r"^[一-鿿A-Za-z]+[：:（(]")
    # Also detect short noun-only headings without colon (e.g., "产品部件介绍")
    def _is_noun_heading(text: str) -> bool:
        if len(text) > 15 or not text:
            return False
        # No verbs/adjectives/handle/operate words
        VERB_HINTS = ("的", "了", "是", "在", "有", "和", "或", "通过", "使用", "进行", "可以", "需要", "应该")
        return not any(v in text for v in VERB_HINTS) and not re.search(r"[，。,.]", text) and not text.startswith(("这", "那", "怎么", "如何", "可"))
    if (_HEADING_LIKE_RE.match(conclusion) or _is_noun_heading(conclusion)) and details:
        conclusion = details.pop(0)
    elif (_HEADING_LIKE_RE.match(conclusion) or _is_noun_heading(conclusion)) and not details:
        # Use the title of the first evidence chunk directly if it's informative
        first_chunk = evidence_chunks[0]
        chunk_text = str(first_chunk.get("text", ""))[:200]
        conclusion = _clean_submission_style_sentence(chunk_text) or conclusion

    lines = [conclusion]
    if details:
        lines.extend(details[:4])
    answer = "\n".join(lines).strip()

    # For Chinese extractive answers: simple natural wrapping, no forced template
    if not _is_ascii_heavy(query):
        # Remove "您好" from middle of answer
        all_lines = answer.split("\n")
        clean_lines = []
        for i, line in enumerate(all_lines):
            stripped = line.strip()
            # Strip leading 您好，for non-first lines
            if i > 0:
                for prefix in ["您好，", "您好！", "您好"]:
                    if stripped.startswith(prefix):
                        stripped = stripped[len(prefix):].strip()
                        break
            if stripped:
                if not stripped.endswith(("。", "！", "？")):
                    # Strip trailing English period before adding Chinese period
                    stripped = stripped.rstrip(".")
                    if stripped:
                        stripped += "。"
                # Strip leading "目标：", "结论：" etc.
                stripped = re.sub(r"^[一-鿿A-Za-z]+[：:（(].{0,20}?", "", stripped)
                if stripped:
                    clean_lines.append(stripped)
        if clean_lines:
            answer = "".join(clean_lines)
        else:
            answer = clean_lines[0] if clean_lines else answer
    else:
        # For English queries
        answer = re.sub(r"\s+", " ", answer).strip()
        # Add period between sentences where missed (only for known sentence-starter words)
        answer = re.sub(
            r"([a-zA-Z])\s+((?:The|It|This|That|These|Those|We|They|He|She|You|I|A|An|When|If|Do|Does|Did|Will|Would|Can|Could|Should|May|Might|Shall|To|In|On|For|With|By|After|Before|During|Remove|Install|Place|Press|Pull|Push|Turn|Open|Close|Set|Check|Make|Use|Keep|Hold|Insert|Slide|Lift|Lower|Raise|Rotate|Twist|Align|Fit|Connect|Disconnect|Ensure|Verify|Confirm|Select|Enter|Follow|Note|Reinstall|Reassemble|Repeat|Drop|Grasp|Grip|Snap|Clip|Attach|Detach|Twist|Squeeze|Flip|Store|Wipe|Brush|Rinse|Soak|Wash|Dry|Wait|Allow|Let|Start|Stop|Begin|Continue|Reset|Adjust|Loosen|Tighten|Remove)\b)",
            r"\1. \2",
            answer,
        )
        answer = answer.rstrip(".")
        if answer:
            answer += "."

    return answer


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
    # Strip leading step numbers: "2 打开前盖板" → "打开前盖板"
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    # Strip leading numbered list markers: "1." "2、" etc
    cleaned = re.sub(r"^\d+[\.、．)）]\s*", "", cleaned)
    # Strip "C." or "C " manual connectors
    cleaned = re.sub(r"^[A-Za-z][\.\s]\s*", "", cleaned)
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
    # Split on common sentence boundaries (。！？.!?) followed by whitespace, or newlines
    raw_parts = re.split(r"(?<=[。！？])\s+|(?<=[.!?])\s+|[\n\r]+", cleaned)
    sentences: list[str] = []
    merged = ""
    for part in raw_parts:
        sentence = part.strip(" -|")
        if not sentence:
            continue
        # Merge back fragments that are likely split abbreviations (e.g., "T." followed by "is")
        if merged and (len(sentence) < 25 or (sentence[0].islower() and not merged.endswith(("。", "！", "？")))):
            merged += " " + sentence
            continue
        if merged:
            if 18 <= len(merged) <= 450:
                sentences.append(merged)
        merged = sentence
    if merged and 18 <= len(merged) <= 450:
        sentences.append(merged)
    if not sentences and cleaned:
        sentences.append(cleaned[:450])
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

        # Title-question semantic relevance boost
        title_relevance_boost = 0.0
        if analysis is not None:
            # Check if title contains question intent keywords
            title_lower = title.lower()
            query_lower = query.lower()

            # Boost for action/intent matching
            action_keywords = {
                "安全": ["安全", "防护", "保护", "注意", "警告", "禁止"],
                "操作": ["操作", "使用", "运行", "启动", "关闭"],
                "维护": ["维护", "保养", "清洁", "清洗", "更换"],
                "安装": ["安装", "装配", "设置"],
                "故障": ["故障", "问题", "异常", "维修"],
                "规格": ["规格", "参数", "尺寸", "重量"],
                "配件": ["配件", "附件", "部件", "组件"],
            }

            for intent, keywords in action_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    if any(kw in title_lower for kw in keywords):
                        title_relevance_boost += 1.5

            # Boost for question word matching
            question_words = ["什么", "如何", "怎么", "哪些", "为什么", "是否", "能不能"]
            for word in question_words:
                if word in query_lower:
                    # Check if title answers this type of question
                    if word == "什么" and any(kw in title_lower for kw in ["是", "指", "为"]):
                        title_relevance_boost += 1.0
                    elif word == "如何" and any(kw in title_lower for kw in ["步骤", "方法", "如何"]):
                        title_relevance_boost += 1.0
                    elif word == "哪些" and any(kw in title_lower for kw in ["包括", "含有", "有"]):
                        title_relevance_boost += 1.0

        evidence_score = base_score
        evidence_score += min(query_overlap * 1.1, 5.0)
        evidence_score += min(image_overlap * 1.8, 5.4)
        evidence_score += min(title_component_overlap * 1.8 + text_component_overlap * 0.9, 4.8)
        evidence_score += min(title_status_overlap * 1.2 + text_status_overlap * 1.5, 4.6)
        evidence_score += min(issue_overlap * 1.6, 3.2)
        evidence_score += min(title_relevance_boost, 3.0)  # Cap at 3.0
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
        row["_title_relevance_boost"] = round(title_relevance_boost, 3)
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
            overlap_threshold = max(1, top_query_overlap)
            if query_has_explicit_product:
                if chunk_query_overlap < 2:
                    continue
            if cross_product_kept >= 5:
                continue
            strong_query_alignment = chunk_query_overlap >= overlap_threshold
            strong_image_alignment = chunk_image_overlap > 0 and chunk_image_overlap >= top_image_overlap
            near_top_score = score >= top_score - 2.0
            variant_rescue = (
                chunk_variant_hits >= 1
                and chunk_query_overlap >= overlap_threshold
                and score >= top_score - 2.5
            )
            if not ((strong_query_alignment and near_top_score) or strong_image_alignment or variant_rescue):
                if chunk_query_overlap < 1:
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

    # Rescue: ensure chunks whose titles match query keywords are always included
    # Replace the lowest-scored ORIGINAL chunks if necessary (not rescue-added ones)
    if ranked and len(filtered) >= FINAL_CONTEXT_CHUNKS:
        query_analysis = analyze_query(query)
        # Build rescue terms: full keywords + 2-char sliding window substrings
        rescue_terms: set[str] = set()
        for kw in query_analysis.keywords:
            norm_kw = _normalize(kw)
            rescue_terms.add(norm_kw)
            for i in range(len(norm_kw) - 1):
                sub = norm_kw[i:i + 2]
                if sub and sub not in {"的", "了", "在", "是"}:
                    rescue_terms.add(sub)
        for phrase in query_analysis.phrases:
            rescue_terms.add(_normalize(phrase))
            for word in _normalize(phrase).split():
                if len(word) >= 2:
                    rescue_terms.add(word)
        rescue_terms.discard("")
        if rescue_terms:
            original_count = len(filtered)
            existing_titles = {str(c.get("title", "")) for c in filtered}
            for chunk in ranked:
                title = str(chunk.get("title", ""))
                if title in existing_titles:
                    continue
                title_lower = _normalize(title)
                if any(term in title_lower for term in rescue_terms):
                    # Only replace original (non-rescued) chunks
                    if len(filtered) >= FINAL_CONTEXT_CHUNKS and original_count > 0:
                        # Find and remove the last original chunk
                        for idx in range(original_count - 1, -1, -1):
                            if idx < len(filtered):
                                removed = filtered.pop(idx)
                                original_count -= 1
                                break
                    filtered.append(chunk)
                    existing_titles.add(title)

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
        # Strip "参阅/查阅/参考...说明书" references from context to prevent LLM echoing them
        body = re.sub(r"详情请参阅[^。\n]*[。]?", "", body)
        body = re.sub(r"请参阅[^。\n]*[。]?", "", body)
        body = re.sub(r"请查阅[^。\n]*[。]?", "", body)
        body = re.sub(r"请参考[^。\n]*[。]?", "", body)
        body = re.sub(r"建议参阅[^。\n]*[。]?", "", body)
        body = re.sub(r"参考\d+产品[^。\n]*[。]?", "", body)
        body = re.sub(r"\s{2,}", " ", body).strip()
        if img_ids:
            body += f"\n（相关配图：{', '.join(img_ids)}）"
        part = f"{header}\n{body}"

        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            # Keep priority ordering: if current chunk doesn't fit and we already
            # have context, stop adding. Only truncate if this is the first chunk.
            if total_chars == 0:
                max_body_len = MAX_CONTEXT_CHARS - len(header) - 50
                body = body[:max_body_len]
                part = f"{header}\n{body}"
                parts.append(part)
                total_chars += len(part)
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

    def _rerank_chunks_by_relevance(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        limit: int = FINAL_CONTEXT_CHUNKS,
    ) -> list[dict[str, Any]]:
        """Re-rank evidence chunks by relevance to the query before LLM call."""
        if not chunks:
            return []
        analysis = analyze_query(query)
        query_terms = set()
        for kw in analysis.keywords:
            query_terms.add(_normalize(kw))
        for phrase in analysis.phrases:
            query_terms.add(_normalize(phrase))
        for prod in analysis.products:
            query_terms.add(_normalize(prod))
        query_terms.discard("")
        if not query_terms:
            return chunks[:limit]
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            title = _normalize(str(chunk.get("title", "")))
            text = _normalize(str(chunk.get("text", "")))
            combined = title + " " + text
            overlap = sum(1 for t in query_terms if t in combined)
            title_overlap = sum(1 for t in query_terms if t in title)
            score = overlap + title_overlap * 2.0
            score += float(chunk.get("_evidence_score", 0)) * 0.5
            scored.append((score, idx, chunk))
        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [chunk for _, _, chunk in scored[:limit]]

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

        # Optional: LLM query expansion for better recall (Phase 4)
        _enable_qe = os.getenv("INDUSTRY_AGENT_ENABLE_QUERY_EXPANSION", "0").strip().lower()
        if _enable_qe in {"1", "true", "on"}:
            try:
                _expander = QueryExpander()
                _expanded = _expander.expand(query)
                for _eq in _expanded.get("queries", []):
                    if _eq and _eq != query:
                        candidate_groups.append(
                            ("expanded", self.retriever.search(_eq, limit=RETRIEVAL_LIMIT))
                        )
            except Exception:
                pass

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

        # Use retriever's original ranking (already optimized by _score).
        # Only apply light evidence filtering to remove clearly irrelevant chunks.
        evidence_chunks = _filter_evidence(chunks)

        if not evidence_chunks and chunks:
            evidence_chunks = chunks[:5]

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
                    "reason": "no_chunks_retrieved",
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

        # 4. Build conversational answer from evidence (primary path)
        conversational_answer = _build_conversational_answer(
            query=query,
            evidence_chunks=evidence_chunks,
            image_ids=image_ids,
        )

        # Try LLM
        answer = self._call_llm(messages)
        if answer and not answer.startswith("LLM 调用失败:"):
            answer = _strip_thinking(answer)
            answer = _strip_llm_structured_format(answer)

        # Check if LLM answer is usable (not raw manual text, not explicit refusal)
        use_llm = False
        if answer and not answer.startswith("LLM 调用失败:") and len(answer.strip()) >= 10:
            if not _is_raw_manual_text(answer) and not _should_use_extractive_manual_answer(answer):
                use_llm = True

        if use_llm:
            # Use LLM answer with minimal cleanup — no forced sections
            answer = format_manual_answer(answer, image_ids=[])
        else:
            # Fallback: extractive answer from evidence
            extractive_answer = _build_extractive_manual_answer(
                query=query,
                evidence_chunks=evidence_chunks,
                image_ids=image_ids,
            )
            if extractive_answer and len(extractive_answer) > 20:
                answer = extractive_answer
            else:
                answer = conversational_answer

        # Answer length truncation
        max_len = MAX_ENGLISH_ANSWER_LENGTH if _is_ascii_heavy(answer) else MAX_ANSWER_LENGTH
        if len(answer) > max_len:
            truncated = answer[:max_len]
            # Try to break at last sentence boundary
            for sep in ["。", "！", "？", ".", "!", "?"]:
                pos = truncated.rfind(sep)
                if pos > max_len * 0.5:
                    truncated = truncated[:pos + 1]
                    break
            answer = truncated.strip()

        # Final cleanup
        answer = _final_answer_cleanup(answer)
        answer = _localize_answer(answer, query)

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
