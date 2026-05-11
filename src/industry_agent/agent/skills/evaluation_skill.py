"""Evaluation skill — lightweight answer quality self-check."""

from __future__ import annotations

import re
from typing import Any

from industry_agent.agent.skills import BaseSkill, SkillResult


class EvaluationSkill(BaseSkill):
    name = "evaluation"
    description = "评估技能：对回答进行轻量自评"

    def execute(
        self,
        *,
        question: str,
        answer: str,
        context: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        try:
            scores = self._heuristic_evaluate(question, answer, context)
            return SkillResult(success=True, data=scores, metadata={"method": "heuristic"})
        except Exception as exc:
            return SkillResult(success=False, error=str(exc))

    def _heuristic_evaluate(self, question: str, answer: str, context: str) -> dict[str, Any]:
        faithfulness = self._compute_faithfulness(answer, context) if context else None
        relevancy = self._compute_relevancy(question, answer)
        completeness = self._compute_completeness(answer)
        valid_scores = [score for score in (faithfulness, relevancy, completeness) if score is not None]
        return {
            "faithfulness": round(faithfulness, 2) if faithfulness is not None else None,
            "relevancy": round(relevancy, 2),
            "completeness": round(completeness, 2),
            "overall": round(sum(valid_scores) / max(len(valid_scores), 1), 2),
            "is_fallback": "根据现有资料无法" in answer,
            "has_pic_markers": "<PIC>" in answer,
            "answer_length": len(answer),
        }

    def _compute_faithfulness(self, answer: str, context: str) -> float:
        answer_sentences = [s.strip() for s in re.split(r"[。！？.!?]", answer) if s.strip()]
        if not answer_sentences:
            return 0.0
        context_normalized = re.sub(r"\s+", "", context.lower())
        grounded_count = 0
        for sentence in answer_sentences:
            words = [w for w in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", sentence) if len(w) >= 2]
            if not words:
                continue
            matches = sum(1 for word in words if word.lower() in context_normalized)
            if matches / len(words) >= 0.3:
                grounded_count += 1
        return grounded_count / len(answer_sentences)

    def _compute_relevancy(self, question: str, answer: str) -> float:
        question_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", question.lower()))
        if not question_terms:
            return 0.5
        answer_normalized = answer.lower()
        matched = sum(1 for term in question_terms if term in answer_normalized)
        return min(matched / len(question_terms), 1.0)

    def _compute_completeness(self, answer: str) -> float:
        score = 0.0
        if len(answer) >= 50:
            score += 0.3
        if len(answer) >= 150:
            score += 0.2
        if re.search(r"\d+[.、]", answer):
            score += 0.2
        if "<PIC>" in answer:
            score += 0.15
        if "根据现有资料无法" not in answer:
            score += 0.15
        return min(score, 1.0)
