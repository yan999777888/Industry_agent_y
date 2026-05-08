"""Agent Orchestrator — skill-based multi-modal customer service agent.

Coordinates the execution flow:
    1. Routing skill  → classify question type
    2. Image skill    → analyze uploaded images (if any)
    3. Retrieval skill → search knowledge base
    4. LLM generation → generate answer with context
    5. Evaluation skill → self-assess answer quality

Designed as a drop-in alternative to the existing AgentService.chat()
while being more modular and extensible.

Usage:
    orchestrator = AgentOrchestrator()
    response = orchestrator.run(
        question="我的电钻指示灯闪烁是什么意思？",
        images=[base64_str],          # optional
        session_id="user_123",        # optional
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from industry_agent.agent.context_manager import ContextManager, TurnContext
from industry_agent.agent.response_formatter import format_manual_answer
from industry_agent.agent.service import (
    ChatRequest,
    ChatResponse,
    _assemble_context,
    _confidence_from_chunks,
    _filter_evidence_for_query,
    _image_details,
    _load_image_index,
    _merge_confidence,
    _merge_images,
    _merge_retrieval_candidates,
    _parse_json_list,
    _strip_thinking,
    _unique,
    FINAL_CONTEXT_CHUNKS,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_TURNS,
    SYSTEM_TEMPLATE,
)
from industry_agent.agent.session_store import InMemorySessionStore, SessionState
from industry_agent.agent.skills import SkillResult, get_skill
from industry_agent.config import settings


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

def _build_llm_messages(
    *,
    context: str,
    query: str,
    dialog_summary: str | None = None,
    image_context: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the messages list for LLM chat completion."""
    system_msg = SYSTEM_TEMPLATE.format(context=context if context else "（未找到相关资料）")
    messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]

    if dialog_summary:
        messages.append({"role": "system", "content": f"【会话上下文】\n{dialog_summary}"})
    if image_context:
        messages.append({"role": "system", "content": f"【用户上传图片信息】\n{image_context}"})
    if history:
        messages.extend(history[-MAX_HISTORY_TURNS * 2 :])

    messages.append({"role": "user", "content": query})
    return messages


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """Skill-based orchestrator for the multi-modal customer service agent.

    Replaces the monolithic AgentService.chat() flow with a modular
    skill pipeline while maintaining API compatibility.
    """

    def __init__(self) -> None:
        # Skills (lazy-loaded)
        self._routing_skill = None
        self._retrieval_skill = None
        self._image_skill = None
        self._evaluation_skill = None

        # Session management
        self.session_store = InMemorySessionStore(max_history_turns=MAX_HISTORY_TURNS)
        self.context_manager = ContextManager(max_history_turns=MAX_HISTORY_TURNS)

        # LLM client (lazy-loaded)
        self._llm_client = None

        # Image index
        self.image_index = _load_image_index()

    # ------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------

    @property
    def routing_skill(self):
        if self._routing_skill is None:
            self._routing_skill = get_skill("routing")
        return self._routing_skill

    @property
    def retrieval_skill(self):
        if self._retrieval_skill is None:
            self._retrieval_skill = get_skill("retrieval")
        return self._retrieval_skill

    @property
    def image_skill(self):
        if self._image_skill is None:
            self._image_skill = get_skill("image")
        return self._image_skill

    @property
    def evaluation_skill(self):
        if self._evaluation_skill is None:
            self._evaluation_skill = get_skill("evaluation")
        return self._evaluation_skill

    @property
    def llm_client(self):
        if self._llm_client is None:
            from industry_agent.llm.client import LLMClient

            self._llm_client = LLMClient()
        return self._llm_client

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        images: list[str] | None = None,
        session_id: str | None = None,
    ) -> ChatResponse:
        """Process a user question through the skill pipeline.

        Args:
            question: User's question text.
            images: Optional list of Base64-encoded images.
            session_id: Optional session ID for multi-turn dialogue.

        Returns:
            ChatResponse compatible with the existing API.
        """
        request = ChatRequest(
            question=question,
            images=images,
            session_id=session_id,
        )
        return self.chat(request)

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Full chat pipeline matching AgentService.chat() interface."""

        # 1. Prepare session context
        session, turn_context = self._prepare_session(request)

        # Handle session control commands
        control_response = self._handle_session_control(request, session, turn_context)
        if control_response:
            return control_response

        # 2. Route the question
        route_result = self.routing_skill.execute(
            question=request.question,
            session_context=self._get_session_context(session),
        )
        route_data = route_result.data or {}
        route = route_data.get("route", "manual_rag")

        # 3. Handle smalltalk
        if route == "smalltalk":
            return ChatResponse(
                answer=route_data.get("reply", "你好！"),
                image_ids=[],
                images=[],
                sources=[],
                references=[],
                confidence=0.99,
                retrieval_debug={"route": "smalltalk", "intent": route_data.get("intent", "")},
            )

        # 4. Handle customer service
        if route == "customer_service":
            from industry_agent.agent.service import _match_smalltalk_reply
            from industry_agent.agent.customer_service_policy import CustomerServicePolicy

            policy = CustomerServicePolicy()
            policy_response = policy.answer(request.question)
            from industry_agent.agent.response_formatter import format_customer_service_answer

            return ChatResponse(
                answer=format_customer_service_answer(policy_response.answer),
                image_ids=[],
                images=[],
                sources=["customer_service_policy"],
                references=[],
                confidence=min(route_data.get("confidence", 0.7), policy_response.confidence),
                retrieval_debug={"route": "customer_service", **route_data},
            )

        # 5. Analyze images
        image_result_data = {}
        image_context = ""
        image_terms: list[str] = []
        image_features: dict[str, list[str]] = {}
        if request.images:
            img_result = self.image_skill.execute(
                images=request.images,
                question=request.question,
            )
            if img_result.success and img_result.data:
                img_data = img_result.data
                image_context = img_data.combined_summary
                image_terms = img_data.retrieval_terms
                image_features = img_data.visual_features
                image_result_data = img_data.to_debug_dict()

        # 6. Retrieve context (manual RAG path)
        retrieval_result = self.retrieval_skill.execute(
            query=turn_context.resolved_question or request.question,
            limit=10,
        )

        chunks = retrieval_result.data or []
        evidence_chunks = _filter_evidence_for_query(
            chunks,
            query=request.question,
            image_terms=image_terms,
            image_features=image_features,
        )

        if not evidence_chunks:
            return ChatResponse(
                answer="根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。",
                image_ids=[],
                images=[],
                sources=[],
                references=[],
                confidence=0.15,
                retrieval_debug={
                    "route": "manual_rag",
                    "retrieved_count": len(chunks),
                    "reason": "no_evidence",
                    **image_result_data,
                },
            )

        # 7. Assemble context and generate
        context_str, image_ids, sources, references = _assemble_context(evidence_chunks)
        images_detail = _image_details(image_ids, self.image_index)
        confidence = _confidence_from_chunks(evidence_chunks)

        messages = _build_llm_messages(
            context=context_str,
            query=turn_context.resolved_question or request.question,
            dialog_summary=turn_context.dialog_summary,
            image_context=image_context,
            history=turn_context.history,
        )

        # 8. Call LLM
        answer = self.llm_client.chat(messages)
        answer = format_manual_answer(answer, image_ids=image_ids)

        # 9. Evaluate (non-blocking)
        eval_result = self.evaluation_skill.execute(
            question=request.question,
            answer=answer,
            context=context_str,
        )

        # 10. Update session
        if session is not None:
            self.context_manager.update_session(
                session=session,
                question=request.question,
                sub_questions=[],
                image_ids=image_ids,
                sources=sources,
                answer=answer,
                turn_context=turn_context,
                uploaded_image_summary=image_context,
            )
            self.session_store.append_turn(
                session,
                user_question=request.question,
                assistant_answer=answer,
            )

        return ChatResponse(
            answer=answer,
            image_ids=image_ids,
            images=images_detail,
            sources=sources,
            references=references,
            confidence=confidence,
            retrieval_debug={
                "route": "manual_rag",
                "retrieved_count": len(chunks),
                "evidence_count": len(evidence_chunks),
                "top_title": evidence_chunks[0].get("title", "") if evidence_chunks else "",
                "image_understanding": image_result_data,
                "evaluation": eval_result.data if eval_result.success else None,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_session(
        self, request: ChatRequest
    ) -> tuple[SessionState | None, TurnContext]:
        session: SessionState | None = None
        if request.session_id:
            session = self.session_store.get_or_create(request.session_id)
        turn_context = self.context_manager.resolve_turn(
            question=request.question,
            session=session,
        )
        return session, turn_context

    def _handle_session_control(
        self,
        request: ChatRequest,
        session: SessionState | None,
        turn_context: TurnContext,
    ) -> ChatResponse | None:
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
                retrieval_debug={"route": "session_control", "action": "clear"},
            )

        if turn_context.needs_clarification:
            return ChatResponse(
                answer="我理解你可能想切换到另一个产品。请补充新的产品名称或型号后再问，我会避免沿用上一轮产品上下文。",
                image_ids=[],
                images=[],
                sources=[],
                references=[],
                confidence=0.55,
                retrieval_debug={"route": "clarification"},
            )
        return None

    def _get_session_context(self, session: SessionState | None) -> dict[str, Any]:
        if session is None:
            return {}
        return {
            "current_route": session.current_route,
            "current_service_topics": session.current_service_topics,
            "current_product": session.current_product,
        }
