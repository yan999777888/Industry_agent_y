"""Routing skill — classify user questions and detect simple follow-ups."""

from __future__ import annotations

from typing import Any

from industry_agent.agent.skills import BaseSkill, SkillResult
from industry_agent.agent.question_router import QuestionRouter
from industry_agent.agent.service import _match_smalltalk_reply
from industry_agent.rag.retriever import analyze_query


class RoutingSkill(BaseSkill):
    name = "routing"
    description = "路由技能：分析用户问题类型，路由到说明书检索、客服策略或闲聊"

    def __init__(self) -> None:
        self.router = QuestionRouter()

    def execute(
        self,
        *,
        question: str,
        session_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SkillResult:
        try:
            smalltalk = _match_smalltalk_reply(question)
            if smalltalk is not None:
                intent, reply = smalltalk
                return SkillResult(
                    success=True,
                    data={"route": "smalltalk", "intent": intent, "reply": reply, "confidence": 0.99},
                    metadata={"route": "smalltalk", "intent": intent},
                )

            if session_context:
                current_route = str(session_context.get("current_route", ""))
                current_topics = list(session_context.get("current_service_topics", []) or [])
                analysis = analyze_query(question)
                if current_route in {"customer_service", "mixed"} and current_topics and not analysis.products and not analysis.models:
                    return SkillResult(
                        success=True,
                        data={
                            "route": "customer_service",
                            "confidence": 0.72,
                            "matched_terms": current_topics[:3],
                            "reason": "inherit_customer_service_context",
                        },
                        metadata={"route": "customer_service"},
                    )

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
                metadata={"route": decision.route},
            )
        except Exception as exc:
            return SkillResult(success=False, error=str(exc), data={"route": "manual_rag", "confidence": 0.5})
