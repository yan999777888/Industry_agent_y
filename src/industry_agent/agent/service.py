"""Customer-service agent service placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatRequest:
    question: str
    image_base64: str | None = None
    session_id: str | None = None


@dataclass
class ChatResponse:
    answer: str
    image_ids: list[str]
    sources: list[str]


class CustomerServiceAgent:
    """Future orchestration layer for multimodal understanding, RAG, and answer generation."""

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError("chat orchestration will be implemented after the RAG pipeline is ready")
