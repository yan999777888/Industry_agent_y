"""Customer-service agent orchestration."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from industry_agent.rag.retriever import SearchResponse, SearchResult, SQLiteRetriever

SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s+|\n+")


@dataclass
class ChatRequest:
    question: str
    image_base64: str | None = None
    session_id: str | None = None
    top_k: int = 5


@dataclass
class SourceCitation:
    manual_id: str
    product_name: str
    chunk_id: str
    title: str
    source_path: str
    score: float

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChatResponse:
    answer: str
    image_ids: list[str]
    sources: list[SourceCitation]
    confidence: float
    debug: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "image_ids": self.image_ids,
            "sources": [item.to_record() for item in self.sources],
            "confidence": self.confidence,
            "debug": self.debug,
        }


class CustomerServiceAgent:
    """Minimal orchestrator for retrieval-grounded customer service."""

    def __init__(self, retriever: SQLiteRetriever | None = None) -> None:
        self.retriever = retriever or SQLiteRetriever()

    def chat(self, request: ChatRequest) -> ChatResponse:
        retrieval = self.retriever.retrieve(request.question, limit=max(request.top_k, 5))
        filtered_results = self._select_consistent_results(retrieval.results)

        if not filtered_results or self._should_clarify(retrieval, filtered_results):
            return ChatResponse(
                answer=(
                    "我暂时没有在当前说明书知识库中找到足够明确的依据来回答这个问题。"
                    "如果你能补充产品名称、型号、报错现象，或上传相关图片，我可以继续帮你查。"
                ),
                image_ids=[],
                sources=[],
                confidence=0.18,
                debug={"query": retrieval.query.to_record(), "retrieved_count": 0},
            )

        answer = self._compose_answer(request.question, retrieval, filtered_results)
        image_ids = _unique_in_order(
            image_id for result in filtered_results[:3] for image_id in result.image_ids
        )[:5]
        sources = [
            SourceCitation(
                manual_id=result.manual_id,
                product_name=result.product_name,
                chunk_id=result.chunk_id,
                title=result.title,
                source_path=result.source_path,
                score=result.score,
            )
            for result in filtered_results[:3]
        ]
        confidence = _estimate_confidence(filtered_results)
        return ChatResponse(
            answer=answer,
            image_ids=image_ids,
            sources=sources,
            confidence=confidence,
            debug={
                "query": retrieval.query.to_record(),
                "selected_chunk_ids": [item.chunk_id for item in filtered_results[:3]],
            },
        )

    def _select_consistent_results(self, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        top_product = results[0].product_name
        coherent = [item for item in results if item.product_name == top_product]
        if coherent:
            return coherent
        return results

    def _should_clarify(self, retrieval: SearchResponse, results: list[SearchResult]) -> bool:
        if not results:
            return True
        top = results[0]
        has_strong_anchor = bool(retrieval.query.product_terms or retrieval.query.model_terms)
        if top.score < 6.0:
            return True
        if not has_strong_anchor and top.score < 10.0:
            return True
        return False

    def _compose_answer(
        self,
        question: str,
        retrieval: SearchResponse,
        results: list[SearchResult],
    ) -> str:
        top = results[0]
        question_text = question.strip()

        if _contains_any(question_text, ["代表什么含义", "什么意思", "含义"]) and top.image_ids:
            lines = _meaning_lines_from_text(top.text)
            if lines:
                answer_lines = [f"根据{top.product_name}说明书，这些指示含义如下："]
                answer_lines.extend(f"{index}. {line}" for index, line in enumerate(lines, start=1))
                answer_lines.append("相关示意图我一并附在结果中，便于你对照查看。")
                return "\n".join(answer_lines)

        if _contains_any(question_text, ["表带", "尺寸", "更换"]):
            evidence = self._collect_best_sentences(question_text, results, max_sentences=4)
            answer_lines = [f"根据{top.product_name}说明书，和表带相关的信息如下："]
            answer_lines.extend(f"{index}. {line}" for index, line in enumerate(evidence, start=1))
            if top.image_ids:
                answer_lines.append("相关表带尺寸或更换示意图已一并返回，方便你对照。")
            return "\n".join(answer_lines)

        if _contains_any(question_text, ["安装"]):
            evidence = self._collect_best_sentences(question_text, results, max_sentences=4)
            answer_lines = [f"根据{top.product_name}说明书，安装要求如下："]
            answer_lines.extend(f"{index}. {line}" for index, line in enumerate(evidence, start=1))
            return "\n".join(answer_lines)

        evidence = self._collect_best_sentences(question_text, results, max_sentences=4)
        answer_lines = [f"根据{top.product_name}说明书，我查到的相关信息如下："]
        answer_lines.extend(f"{index}. {line}" for index, line in enumerate(evidence, start=1))
        if top.image_ids:
            answer_lines.append("相关图片也已一起返回，方便你进一步核对。")
        return "\n".join(answer_lines)

    def _collect_best_sentences(
        self,
        question: str,
        results: list[SearchResult],
        *,
        max_sentences: int,
    ) -> list[str]:
        keywords = self.retriever.analyze_query(question).keywords
        selected: list[str] = []
        for result in results[:3]:
            sentences = _split_sentences(result.text)
            scored = sorted(
                ((self._sentence_score(sentence, keywords), sentence) for sentence in sentences),
                key=lambda item: (item[0], len(item[1])),
                reverse=True,
            )
            for score, sentence in scored:
                if score <= 0:
                    continue
                clean = _cleanup_sentence(sentence)
                if not clean or clean in selected:
                    continue
                selected.append(clean)
                if len(selected) >= max_sentences:
                    return selected
        if not selected:
            return [_cleanup_sentence(results[0].text)]
        return selected

    def _sentence_score(self, sentence: str, keywords: list[str]) -> float:
        normalized_sentence = sentence.lower()
        score = 0.0
        for keyword in keywords:
            if keyword.lower() in normalized_sentence:
                score += 2.0
        if len(sentence) <= 60:
            score += 0.5
        return score


def _meaning_lines_from_text(text: str) -> list[str]:
    lines = [_cleanup_sentence(part) for part in text.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _split_sentences(text: str) -> list[str]:
    return [item.strip() for item in SENTENCE_SPLIT_RE.split(text) if item.strip()]


def _cleanup_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^#\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _estimate_confidence(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    top_score = results[0].score
    second_score = results[1].score if len(results) > 1 else 0.0
    confidence = 0.35 + min(top_score / 20.0, 0.45) + min((top_score - second_score) / 20.0, 0.15)
    return round(min(confidence, 0.98), 2)


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _unique_in_order(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered
