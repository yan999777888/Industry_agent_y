"""Evaluation skill — self-assessment of RAG answer quality.

Implements lightweight evaluation inspired by all-in-rag's RAGAS approach:
- Faithfulness: does the answer stick to retrieved context?
- Relevancy: does the answer address the question?
- Context precision: are the retrieved chunks relevant?
"""

from __future__ import annotations

import re
from typing import Any

from industry_agent.agent.skills import BaseSkill, SkillResult


class EvaluationSkill(BaseSkill):
    """Evaluate RAG answer quality using heuristic and LLM-based scoring."""

    name = "evaluation"
    description = "评估技能：对 RAG 生成的回答进行质量自评"

    def execute(
        self,
        *,
        question: str,
        answer: str,
        context: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        """Evaluate answer quality.

        Args:
            question: Original user question.
            answer: Generated answer to evaluate.
            context: Retrieved context used for generation.

        Returns:
            SkillResult with scores dict as data.
        """
        try:
            scores = self._heuristic_evaluate(question, answer, context)
            return SkillResult(
                success=True,
                data=scores,
                metadata={"method": "heuristic"},
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                error=str(exc),
            )

    def _heuristic_evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> dict[str, Any]:
        """Rule-based quality scoring."""
        scores: dict[str, Any] = {}

        # Faithfulness: check if answer content appears in context
        if context:
            faithfulness = self._compute_faithfulness(answer, context)
            scores["faithfulness"] = round(faithfulness, 2)
        else:
            scores["faithfulness"] = None

        # Relevancy: check if question terms appear in answer
        relevancy = self._compute_relevancy(question, answer)
        scores["relevancy"] = round(relevancy, 2)

        # Completeness: check answer length and structure
        completeness = self._compute_completeness(answer)
        scores["completeness"] = round(completeness, 2)

        # Overall score (weighted average)
        valid_scores = [v for v in [scores["faithfulness"], scores["relevancy"], scores["completeness"]] if v is not None]
        scores["overall"] = round(sum(valid_scores) / max(len(valid_scores), 1), 2)

        # Quality flags
        scores["is_fallback"] = "根据现有资料无法" in answer
        scores["has_pic_markers"] = "<PIC>" in answer
        scores["answer_length"] = len(answer)

        return scores

    def _compute_faithfulness(self, answer: str, context: str) -> float:
        """How much of the answer is grounded in the context (0-1)."""
        answer_sentences = [s.strip() for s in re.split(r"[。！？.!?]", answer) if s.strip()]
        if not answer_sentences:
            return 0.0

        context_normalized = re.sub(r"\s+", "", context.lower())
        grounded_count = 0
        for sentence in answer_sentences:
            # Check if key fragments of the sentence appear in context
            words = [w for w in re.findall(r"[一-鿿]{2,}|[A-Za-z]{3,}", sentence) if len(w) >= 2]
            if not words:
                continue
            matches = sum(1 for w in words if w.lower() in context_normalized)
            if matches / len(words) >= 0.3:
                grounded_count += 1

        return grounded_count / len(answer_sentences)

    def _compute_relevancy(self, question: str, answer: str) -> float:
        """How well the answer addresses the question (0-1)."""
        question_terms = set(re.findall(r"[一-鿿]{2,}|[A-Za-z]{3,}", question.lower()))
        if not question_terms:
            return 0.5

        answer_normalized = answer.lower()
        matched = sum(1 for term in question_terms if term in answer_normalized)
        return min(matched / len(question_terms), 1.0)

    def _compute_completeness(self, answer: str) -> float:
        """Answer structural completeness (0-1)."""
        score = 0.0

        # Has reasonable length
        if len(answer) >= 50:
            score += 0.3
        if len(answer) >= 150:
            score += 0.2

        # Has structured content (lists, steps)
        if re.search(r"\d+[.、]", answer):
            score += 0.2

        # Has PIC markers for visual content
        if "<PIC>" in answer:
            score += 0.15

        # Not just a fallback message
        if "根据现有资料无法" not in answer:
            score += 0.15

        return min(score, 1.0)

    def llm_evaluate(
        self,
        *,
        question: str,
        answer: str,
        context: str = "",
    ) -> SkillResult:
        """LLM-based evaluation (requires configured LLM client).

        Uses the LLM as a judge to score the answer quality.
        """
        try:
            from industry_agent.llm.client import LLMClient

            client = LLMClient()
            prompt = f"""请作为一个客服质量评审员，对以下问答对进行评分（1-5分）。

【用户问题】
{question}

【检索到的参考资料】
{context or "（无）"}

【客服回答】
{answer}

请从以下维度评价：
1. 回答是否准确回应了问题
2. 回答是否有深度、结构清晰
3. 图文结合是否恰当

请仅输出一个 JSON 对象，格式：
{{"score": <1-5>, "reason": "<简短评语>"}}"""

            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            import json

            score_data = json.loads(response)
            return SkillResult(
                success=True,
                data=score_data,
                metadata={"method": "llm_judge"},
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                error=str(exc),
                metadata={"method": "llm_judge"},
            )
