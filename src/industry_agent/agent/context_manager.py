"""Context resolution helpers for structured multi-turn dialogue."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from industry_agent.agent.question_splitter import SubQuestion
from industry_agent.agent.session_store import SessionState
from industry_agent.rag.retriever import QueryAnalysis, analyze_query


_FOLLOW_UP_TERMS = (
    "这个", "这个呢", "那个", "它", "它的", "该设备", "这台设备", "刚才", "上面",
    "继续", "另外", "还有", "那", "那它", "那这个", "那么", "然后", "接着",
)
_SHORT_FOLLOW_UP_RE = re.compile(r"^(那|这|它|继续|还有|另外)")


@dataclass(frozen=True)
class TurnContext:
    """Resolved conversation context for the current user turn."""

    raw_question: str
    resolved_question: str
    analysis: QueryAnalysis
    is_follow_up: bool
    inherited_product: str = ""
    inherited_models: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    dialog_summary: str = ""


class ContextManager:
    """Resolve follow-up intent and update session-level dialogue state."""

    def __init__(self, *, max_history_turns: int = 5) -> None:
        self.max_history_turns = max_history_turns

    def resolve_turn(self, *, question: str, session: SessionState | None) -> TurnContext:
        cleaned_question = question.strip()
        analysis = analyze_query(cleaned_question)
        if session is None or not session.history:
            return TurnContext(
                raw_question=cleaned_question,
                resolved_question=cleaned_question,
                analysis=analysis,
                is_follow_up=False,
            )

        is_follow_up = self._is_follow_up(cleaned_question, analysis, session)
        inherited_product = ""
        inherited_models: list[str] = []
        resolved_question = cleaned_question

        if is_follow_up:
            if not analysis.products and session.current_product:
                inherited_product = session.current_product
            if not analysis.models and session.current_models and (inherited_product or self._contains_follow_up_reference(cleaned_question)):
                inherited_models = session.current_models[:2]
            resolved_question = self._augment_query(
                cleaned_question,
                product=inherited_product,
                models=inherited_models,
            )

        return TurnContext(
            raw_question=cleaned_question,
            resolved_question=resolved_question,
            analysis=analysis,
            is_follow_up=is_follow_up,
            inherited_product=inherited_product,
            inherited_models=inherited_models,
            history=session.history[-self.max_history_turns * 2 :],
            dialog_summary=session.dialog_summary,
        )

    def build_subquestion_query(
        self,
        *,
        sub_question: SubQuestion,
        original_question: str,
        turn_context: TurnContext,
    ) -> str:
        base_query = sub_question.normalized_text
        if sub_question.text != original_question.strip() and (sub_question.depends_on_previous or len(base_query) < 10):
            base_query = f"{turn_context.resolved_question} {base_query}".strip()

        explicit_product = turn_context.analysis.products[0] if turn_context.analysis.products else ""
        explicit_models = turn_context.analysis.models[:2]
        sub_analysis = analyze_query(base_query)

        if not sub_analysis.products and explicit_product:
            base_query = self._augment_query(base_query, product=explicit_product, models=explicit_models if not sub_analysis.models else [])
            sub_analysis = analyze_query(base_query)

        if not sub_analysis.products and turn_context.inherited_product:
            base_query = self._augment_query(
                base_query,
                product=turn_context.inherited_product,
                models=turn_context.inherited_models if not sub_analysis.models else [],
            )

        return base_query.strip()

    def update_session(
        self,
        *,
        session: SessionState,
        question: str,
        sub_questions: list[SubQuestion],
        image_ids: list[str],
        sources: list[str],
        answer: str,
        turn_context: TurnContext,
        uploaded_image_summary: str = "",
    ) -> SessionState:
        analysis = turn_context.analysis
        if analysis.products:
            session.current_product = analysis.products[0]
        elif turn_context.inherited_product:
            session.current_product = turn_context.inherited_product
        elif sources:
            session.current_product = sources[0]

        if analysis.models:
            session.current_models = analysis.models[:3]
        elif turn_context.inherited_models:
            session.current_models = turn_context.inherited_models[:3]

        session.recent_questions.append(question.strip())
        session.recent_sub_questions.extend(
            sub_question.normalized_text
            for sub_question in sub_questions
            if sub_question.normalized_text
        )
        session.recent_image_ids = _merge_unique(session.recent_image_ids, image_ids)
        if uploaded_image_summary:
            session.recent_user_image_summaries.append(uploaded_image_summary)
        session.dialog_summary = self._build_dialog_summary(session, answer=answer)
        return session

    def _is_follow_up(self, question: str, analysis: QueryAnalysis, session: SessionState) -> bool:
        if not session.history:
            return False
        normalized = _normalize(question)
        if any(term in normalized for term in (_normalize(term) for term in _FOLLOW_UP_TERMS)):
            return True
        if _SHORT_FOLLOW_UP_RE.search(question.strip()):
            return True
        if not analysis.products and not analysis.models and session.current_product:
            return True
        return False

    def _contains_follow_up_reference(self, question: str) -> bool:
        normalized = _normalize(question)
        return any(term in normalized for term in (_normalize(term) for term in _FOLLOW_UP_TERMS))

    def _augment_query(self, query: str, *, product: str = "", models: list[str] | None = None) -> str:
        parts: list[str] = []
        if product:
            parts.append(product)
        if models:
            parts.extend(model for model in models if model)
        parts.append(query.strip())
        return " ".join(_dedupe_preserve_order(parts))

    def _build_dialog_summary(self, session: SessionState, *, answer: str) -> str:
        parts: list[str] = []
        if session.current_product:
            parts.append(f"当前讨论产品：{session.current_product}")
        if session.current_models:
            parts.append(f"最近提到型号：{'、'.join(session.current_models[:2])}")
        if session.recent_sub_questions:
            parts.append(f"最近问题：{'；'.join(session.recent_sub_questions[-3:])}")
        if session.recent_image_ids:
            parts.append(f"最近相关图片：{', '.join(session.recent_image_ids[-3:])}")
        if session.recent_user_image_summaries:
            parts.append(f"最近上传图片：{session.recent_user_image_summaries[-1][:120]}")
        if answer.strip():
            parts.append(f"上一轮回答摘要：{answer.strip()[:120]}")
        return "；".join(parts)


def _merge_unique(existing: list[str], new_values: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*existing, *new_values]:
        if not value or value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        for token in value.split():
            token = token.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
    return result


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())
