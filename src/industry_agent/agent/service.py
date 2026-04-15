"""Agent orchestration: retrieve context -> build prompt -> call LLM."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from industry_agent.agent.context_manager import ContextManager, TurnContext
from industry_agent.agent.customer_service_policy import CustomerServicePolicy
from industry_agent.agent.image_understanding import ImageUnderstandingResult, ImageUnderstander
from industry_agent.agent.question_splitter import SubQuestion, split_complex_question
from industry_agent.agent.question_router import QuestionRouter, RouteDecision
from industry_agent.agent.response_formatter import format_customer_service_answer, format_manual_answer
from industry_agent.agent.session_store import InMemorySessionStore, SessionState
from industry_agent.config import settings
from industry_agent.rag.retriever import SQLiteRetriever

try:
    import httpx
except ImportError:  # pragma: no cover - optional for test environments
    httpx = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRIEVAL_LIMIT = 8         # chunks to retrieve before evidence filtering
FINAL_CONTEXT_CHUNKS = 4    # chunks passed into the LLM
MAX_CONTEXT_CHARS = 4000    # truncate context to fit model window
MAX_HISTORY_TURNS = 5       # keep last N turns per session
MIN_TOP_SCORE = 10.0        # below this, do not ask LLM to hallucinate
MIN_KEEP_SCORE = 8.0        # chunks below this score are discarded

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava-phi3")

SYSTEM_TEMPLATE = """\
你是一个专业的工业产品客服智能体。请严格遵守以下规则：

1. **只基于下方【参考资料】回答**，不得编造任何信息。
2. 如果参考资料不足以回答问题，请明确说明"根据现有资料无法回答此问题"。
3. 如果用户一次问了多个问题，请拆成多个小标题逐一回答。
4. 回答格式固定为：
   - 结论：
   - 操作/说明：
   - 注意事项：
   - 相关图片：
5. 相关图片只能使用参考资料中出现的图片 ID，不要编造图片 ID。
6. 直接给出回答，不要输出任何思考过程。

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


# ---------------------------------------------------------------------------
# Agent service
# ---------------------------------------------------------------------------

class AgentService:
    """Retrieve -> assemble context -> call Ollama LLM -> return answer."""

    def __init__(
        self,
        retriever: SQLiteRetriever | None = None,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
    ) -> None:
        self.retriever = retriever or SQLiteRetriever()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session_store = InMemorySessionStore(max_history_turns=MAX_HISTORY_TURNS)
        self.context_manager = ContextManager(max_history_turns=MAX_HISTORY_TURNS)
        self.question_router = QuestionRouter()
        self.customer_service_policy = CustomerServicePolicy()
        if httpx is None:
            raise RuntimeError("httpx is required to use AgentService with Ollama. Please install requirements.txt")
        self.http_client = httpx.Client(proxy=None, timeout=120.0)
        self.image_understander = ImageUnderstander(
            base_url=self.base_url,
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
    ) -> dict[str, Any]:
        # 1. Retrieve
        chunks = self.retriever.search(query, limit=RETRIEVAL_LIMIT)
        evidence_chunks = _filter_evidence(chunks)

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
        answer = format_manual_answer(answer, image_ids=image_ids)

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
                "top_title": evidence_chunks[0].get("title", "") if evidence_chunks else "",
                "top_product": evidence_chunks[0].get("product_name", "") if evidence_chunks else "",
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

        sub_answer_blocks = []
        for index, (sub_question, result) in enumerate(zip(sub_questions, sub_results), start=1):
            sub_answer_blocks.append(
                f"问题{index}：{sub_question.normalized_text}\n回答：{result['answer']}"
            )
        system_msg = SUBQUESTION_MERGE_TEMPLATE.format(
            original_question=original_question,
            sub_answers="\n\n".join(sub_answer_blocks),
        )
        return self._call_llm([{"role": "system", "content": system_msg}])

    def _generate_customer_service_response(
        self,
        *,
        question: str,
        route_decision: RouteDecision,
    ) -> dict[str, Any]:
        policy_response = self.customer_service_policy.answer(question)
        return {
            "answer": format_customer_service_answer(policy_response.answer),
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
            route_decision = self.question_router.route(sub_question.normalized_text)
            if route_decision.route == "customer_service":
                result = self._generate_customer_service_response(
                    question=sub_question.normalized_text,
                    route_decision=route_decision,
                )
                resolved_query = sub_question.normalized_text
            else:
                resolved_query = self._build_subquestion_query(
                    sub_question=sub_question,
                    original_question=request.question,
                    turn_context=turn_context,
                )
                if image_result.retrieval_hint:
                    resolved_query = f"{resolved_query} {image_result.retrieval_hint}".strip()
                result = self.generate_response(
                    query=resolved_query,
                    history=turn_context.history,
                    image_input=request.images[0] if request.images else None,
                    dialog_summary=turn_context.dialog_summary,
                    image_context=image_result.combined_summary,
                )
            result["retrieval_debug"] = {
                **result.get("retrieval_debug", {}),
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
            self.image_understander = ImageUnderstander(
                base_url=self.base_url,
                http_client=getattr(self, "http_client", None),
                vision_model=OLLAMA_VISION_MODEL,
            )

    def _analyze_uploaded_images(self, request: ChatRequest) -> ImageUnderstandingResult:
        self._ensure_runtime_components()
        return self.image_understander.analyze_images(
            request.images or [],
            question=request.question,
        )

    # ------------------------------------------------------------------
    # LLM call — Ollama native /api/chat (supports think=false)
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        try:
            resp = self.http_client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,           # disable qwen3 thinking mode
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1024,  # max output tokens
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            content = _strip_thinking(content)
            return content.strip() if content.strip() else "模型未返回有效回答。"
        except Exception as exc:
            return f"LLM 调用失败: {exc}"
