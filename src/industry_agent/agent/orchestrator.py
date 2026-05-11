"""Optional orchestrator backend inspired by the modular skill pipeline."""

from __future__ import annotations

from typing import Any

from industry_agent.agent.context_manager import ContextManager
from industry_agent.agent.customer_service_kb import CustomerServiceKnowledgeBase
from industry_agent.agent.customer_service_policy import CustomerServicePolicy
from industry_agent.agent.prompts import build_customer_service_system_prompt, build_manual_qa_system_prompt
from industry_agent.agent.question_splitter import SubQuestion, split_complex_question
from industry_agent.agent.response_formatter import (
    format_customer_service_answer,
    format_manual_answer,
    format_multi_question_answer,
)
from industry_agent.agent.service import (
    ChatRequest,
    ChatResponse,
    FINAL_CONTEXT_CHUNKS,
    MAX_HISTORY_TURNS,
    RETRIEVAL_LIMIT,
    _assemble_context,
    _build_extractive_manual_answer,
    _confidence_from_chunks,
    _filter_evidence_for_query,
    _image_details,
    _load_image_index,
    _match_smalltalk_reply,
    _merge_confidence,
    _merge_images,
    _should_use_extractive_manual_answer,
    _unique,
)
from industry_agent.agent.session_store import InMemorySessionStore, SessionState
from industry_agent.agent.skills import get_skill
from industry_agent.config import settings
from industry_agent.llm.client import LLMClient
from industry_agent.rag.retriever import analyze_query


_SERVICE_FOLLOW_UP_TERMS: tuple[str, ...] = (
    "那", "还", "还有", "需要", "准备", "材料", "多久", "几天", "怎么办",
    "可以吗", "能不能", "怎么申请", "怎么处理", "流程", "凭证", "证明",
    "谁承担", "联系谁", "审核", "下一步", "然后呢",
)


class AgentOrchestrator:
    """Modular backend that keeps the current interface but uses skills."""

    def __init__(self) -> None:
        self.session_store = InMemorySessionStore(max_history_turns=MAX_HISTORY_TURNS)
        self.context_manager = ContextManager(max_history_turns=MAX_HISTORY_TURNS)
        self.customer_service_policy = CustomerServicePolicy()
        self.customer_service_kb = CustomerServiceKnowledgeBase()
        self.image_index = _load_image_index()
        self.llm_client = LLMClient()
        self.routing_skill = get_skill("routing")
        self.retrieval_skill = get_skill("retrieval")
        self.image_skill = get_skill("image")
        self.evaluation_skill = get_skill("evaluation")

    def chat(self, request: ChatRequest) -> ChatResponse:
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
                retrieval_debug={"route": "smalltalk", "intent": intent},
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
                retrieval_debug={"route": "session_control", "action": "clear_context"},
            )
        if turn_context.needs_clarification:
            return ChatResponse(
                answer="我理解你可能想切换到另一个产品。请补充新的产品名称或型号后再问，我会避免沿用上一轮产品上下文。",
                image_ids=[],
                images=[],
                sources=[],
                references=[],
                confidence=0.55,
                retrieval_debug={"route": "clarification", "reason": turn_context.clarification_reason},
            )

        image_result = self.image_skill.execute(images=request.images or [], question=request.question).data
        if image_result is None:
            from industry_agent.agent.image_understanding import ImageUnderstandingResult

            image_result = ImageUnderstandingResult(has_image_input=False)

        sub_questions = split_complex_question(request.question) or [
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
            if route_decision["route"] == "customer_service":
                result = self._generate_customer_service_response(
                    question=sub_question.normalized_text,
                    context_topics=_unique(
                        [*(session.current_service_topics if session is not None else []), *turn_service_topics]
                    ),
                    route_decision=route_decision,
                )
            else:
                base_query = self.context_manager.build_subquestion_query(
                    sub_question=sub_question,
                    original_question=request.question,
                    turn_context=turn_context,
                )
                result = self._generate_manual_response(
                    query=base_query,
                    question=request.question,
                    history=turn_context.history,
                    dialog_summary=turn_context.dialog_summary,
                    image_context=image_result.combined_summary,
                    image_terms=image_result.retrieval_terms,
                    image_features=image_result.visual_features,
                )
                result["retrieval_debug"] = {
                    **result.get("retrieval_debug", {}),
                    "base_query": base_query,
                    "resolved_query": base_query,
                }
            result["retrieval_debug"] = {
                **result.get("retrieval_debug", {}),
                "route_decision": route_decision,
                "image_understanding": image_result.to_debug_dict(),
            }
            turn_service_topics = _unique(
                [*turn_service_topics, *result["retrieval_debug"].get("matched_policy_topics", [])]
            )
            sub_results.append(result)

        merged_answer = self._merge_subquestion_answers(sub_questions=sub_questions, sub_results=sub_results)
        merged_image_ids = _unique([image_id for result in sub_results for image_id in result["image_ids"]])
        merged_images = _merge_images([result["images"] for result in sub_results])
        merged_sources = _unique([source for result in sub_results for source in result["sources"]])
        merged_references = [
            {
                **reference,
                "sub_question_id": sub_question.sub_question_id,
                "sub_question_text": sub_question.normalized_text,
            }
            for sub_question, result in zip(sub_questions, sub_results)
            for reference in result["references"]
        ]
        merged_confidence = _merge_confidence([result["confidence"] for result in sub_results])
        merged_debug = {
            "session": {
                "session_id": request.session_id or "",
                "is_follow_up": turn_context.is_follow_up,
                "resolved_question": turn_context.resolved_question,
                "inherited_product": turn_context.inherited_product,
                "inherited_models": turn_context.inherited_models,
                "dialog_summary": turn_context.dialog_summary,
                "topic_switched": turn_context.topic_switched,
            },
            "sub_results": [
                {
                    "sub_question_id": sub_question.sub_question_id,
                    "confidence": result["confidence"],
                    "retrieval_debug": result["retrieval_debug"],
                }
                for sub_question, result in zip(sub_questions, sub_results)
            ],
        }

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
            self.session_store.append_turn(session, user_question=request.question, assistant_answer=merged_answer)

        return ChatResponse(
            answer=merged_answer,
            image_ids=merged_image_ids,
            images=merged_images,
            sources=merged_sources,
            references=merged_references,
            confidence=merged_confidence,
            retrieval_debug=merged_debug,
        )

    def _prepare_turn_context(self, request: ChatRequest) -> tuple[SessionState | None, Any]:
        session: SessionState | None = None
        if request.session_id:
            session = self.session_store.get_or_create(request.session_id)
        turn_context = self.context_manager.resolve_turn(question=request.question, session=session)
        return session, turn_context

    def _resolve_route_decision(
        self,
        *,
        question: str,
        session: SessionState | None,
        turn_service_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        route_result = self.routing_skill.execute(
            question=question,
            session_context={
                "current_route": session.current_route if session is not None else "",
                "current_service_topics": session.current_service_topics if session is not None else [],
            },
        )
        route_data = dict(route_result.data or {"route": "manual_rag", "confidence": 0.5})
        if route_data.get("route") == "customer_service":
            return route_data
        if turn_service_topics:
            analysis = analyze_query(question)
            if not (analysis.products or analysis.models) and self._looks_like_customer_service_follow_up(question):
                return {
                    "route": "customer_service",
                    "confidence": max(float(route_data.get("confidence", 0.5)), 0.74),
                    "matched_terms": turn_service_topics[:3],
                    "manual_score": route_data.get("manual_score", 0),
                    "service_score": max(int(route_data.get("service_score", 0)), 2),
                    "reason": "inherit_current_turn_customer_service_context",
                }
        return route_data

    def _generate_customer_service_response(
        self,
        *,
        question: str,
        context_topics: list[str],
        route_decision: dict[str, Any],
    ) -> dict[str, Any]:
        policy_response = self.customer_service_policy.answer(question, context_topics=context_topics)
        kb_hits = self.customer_service_kb.search(
            question,
            context_topics=[*policy_response.matched_topics, *context_topics],
            limit=4,
        )
        kb_context = self.customer_service_kb.build_context(kb_hits)
        prompt_context = (
            f"【客服策略骨架】\n{policy_response.answer}\n\n【客服知识参考】\n{kb_context}"
            if kb_context
            else policy_response.answer
        )
        prompt_result = build_customer_service_system_prompt(prompt_context)
        messages = [
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
        llm_answer = self.llm_client.chat(messages)
        normalized_llm = llm_answer.strip()
        used_policy_fallback = (
            not normalized_llm
            or normalized_llm.startswith("LLM 调用失败:")
            or "根据现有资料无法" in normalized_llm
        )
        return {
            "answer": format_customer_service_answer(policy_response.answer if used_policy_fallback else llm_answer),
            "image_ids": [],
            "images": [],
            "sources": ["customer_service_policy", *(["customer_service_kb"] if kb_hits else [])],
            "references": [
                *[
                    {
                        "chunk_id": f"policy_{topic}",
                        "title": "客服策略知识",
                        "text_snippet": question[:100],
                        "product_name": "customer_service_policy",
                        "score": str(route_decision.get("confidence", 0.7)),
                    }
                    for topic in policy_response.matched_topics
                ],
                *[
                    {
                        "chunk_id": str(hit.get("entry_id", "")),
                        "title": str(hit.get("title", "")),
                        "text_snippet": str(hit.get("content", ""))[:320],
                        "product_name": "customer_service_kb",
                        "score": str(hit.get("score", "")),
                    }
                    for hit in kb_hits
                ],
            ],
            "confidence": round(min(float(route_decision.get("confidence", 0.7)), policy_response.confidence), 2),
            "retrieval_debug": {
                "matched_policy_topics": policy_response.matched_topics,
                "route": "customer_service",
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

    def _generate_manual_response(
        self,
        *,
        query: str,
        question: str,
        history: list[dict[str, str]],
        dialog_summary: str,
        image_context: str,
        image_terms: list[str],
        image_features: dict[str, list[str]],
    ) -> dict[str, Any]:
        retrieval_result = self.retrieval_skill.execute(query=query, limit=RETRIEVAL_LIMIT)
        chunks = list(retrieval_result.data or [])
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
                "retrieval_debug": {"reason": "low_confidence_or_no_evidence"},
            }

        context, image_ids, sources, references = _assemble_context(evidence_chunks)
        prompt_result = build_manual_qa_system_prompt(context)
        messages: list[dict[str, str]] = [{"role": "system", "content": prompt_result.content}]
        if dialog_summary:
            messages.append({"role": "system", "content": f"【会话上下文】\n{dialog_summary}"})
        if image_context:
            messages.append({"role": "system", "content": f"【用户上传图片信息】\n{image_context}"})
        if history:
            messages.extend(history[-MAX_HISTORY_TURNS * 2 :])
        messages.append({"role": "user", "content": query})

        answer = self.llm_client.chat(messages)
        answer = format_manual_answer(answer, image_ids=image_ids)
        if _should_use_extractive_manual_answer(answer):
            answer = _build_extractive_manual_answer(
                query=query,
                evidence_chunks=evidence_chunks,
                image_ids=image_ids,
            )

        evaluation = self.evaluation_skill.execute(question=question, answer=answer, context=context)
        return {
            "answer": answer,
            "image_ids": image_ids,
            "images": _image_details(image_ids, self.image_index),
            "sources": sources,
            "references": references,
            "confidence": _confidence_from_chunks(evidence_chunks),
            "retrieval_debug": {
                "retrieved_count": len(chunks),
                "evidence_count": len(evidence_chunks),
                "candidate_titles": [str(chunk.get("title", "")) for chunk in chunks[:5]],
                "selected_titles": [str(chunk.get("title", "")) for chunk in evidence_chunks[:FINAL_CONTEXT_CHUNKS]],
                "evaluation": evaluation.data if evaluation.success else {"error": evaluation.error},
            },
        }

    def _merge_subquestion_answers(
        self,
        *,
        sub_questions: list[SubQuestion],
        sub_results: list[dict[str, Any]],
    ) -> str:
        if len(sub_results) == 1:
            return str(sub_results[0]["answer"])
        return format_multi_question_answer(
            [
                (sub_question.normalized_text, str(result["answer"]))
                for sub_question, result in zip(sub_questions, sub_results)
            ]
        )

    def _looks_like_customer_service_follow_up(self, question: str) -> bool:
        normalized = "".join(question.strip().split())
        if not normalized:
            return False
        if len(normalized) <= 14:
            return True
        return any(term in normalized for term in _SERVICE_FOLLOW_UP_TERMS)

    def _update_session_route_state(self, *, session: SessionState, sub_results: list[dict[str, Any]]) -> None:
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
