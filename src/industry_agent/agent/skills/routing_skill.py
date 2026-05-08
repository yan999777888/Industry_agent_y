"""Routing skill — classifies user questions and routes to appropriate strategy.

Wraps the existing QuestionRouter with additional context-aware routing
capabilities for the orchestrator.
"""

from __future__ import annotations

from typing import Any

from industry_agent.agent.skills import BaseSkill, SkillResult


class RoutingSkill(BaseSkill):
    """Route user questions to manual RAG, customer service, or smalltalk."""

    name = "routing"
    description = "路由技能：分析用户问题类型，路由到说明书检索、客服策略或闲聊"

    def __init__(self) -> None:
        self._router = None
        self._smalltalk_matcher = None

    @property
    def router(self) -> Any:
        if self._router is None:
            from industry_agent.agent.question_router import QuestionRouter

            self._router = QuestionRouter()
        return self._router

    def execute(
        self,
        *,
        question: str,
        session_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SkillResult:
        """Route the question to the appropriate handler.

        Args:
            question: User's question text.
            session_context: Optional session state for context-aware routing.

        Returns:
            SkillResult with data=RouteDecision info dict.
        """
        try:
            # Check smalltalk first
            from industry_agent.agent.service import _match_smalltalk_reply

            smalltalk = _match_smalltalk_reply(question)
            if smalltalk is not None:
                intent, reply = smalltalk
                return SkillResult(
                    success=True,
                    data={
                        "route": "smalltalk",
                        "intent": intent,
                        "reply": reply,
                        "confidence": 0.99,
                    },
                    metadata={"route": "smalltalk", "intent": intent},
                )

            # Check session context for customer service follow-up
            if session_context:
                from industry_agent.agent.service import AgentService

                svc = AgentService.__new__(AgentService)
                if hasattr(svc, "_looks_like_customer_service_follow_up"):
                    current_route = session_context.get("current_route", "")
                    service_topics = session_context.get("current_service_topics", [])
                    if current_route in ("customer_service", "mixed") and service_topics:
                        from industry_agent.rag.retriever import analyze_query

                        analysis = analyze_query(question)
                        if not analysis.products and not analysis.models:
                            from industry_agent.agent.question_router import RouteDecision

                            decision = RouteDecision(
                                route="customer_service",
                                confidence=0.72,
                                matched_terms=service_topics[:3],
                                reason="inherit_customer_service_context",
                            )
                            return SkillResult(
                                success=True,
                                data={
                                    "route": decision.route,
                                    "confidence": decision.confidence,
                                    "matched_terms": decision.matched_terms,
                                    "reason": decision.reason,
                                },
                                metadata={"route": decision.route},
                            )

            # Standard routing
            decision = self.router.route(question)
            return SkillResult(
                success=True,
                data={
                    "route": decision.route,
                    "confidence": decision.confidence,
                    "matched_terms": decision.matched_terms,
                    "manual_score": decision.manual_score,
                    "service_score": decision.service_score,
                    "reason": decision.reason,
                },
                metadata={"route": decision.route, "confidence": decision.confidence},
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                error=str(exc),
                data={"route": "manual_rag", "confidence": 0.5},
            )

    def is_available(self) -> bool:
        try:
            _ = self.router
            return True
        except Exception:
            return False
