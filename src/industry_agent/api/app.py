"""FastAPI application factory.

Provides the /chat endpoint that wires together:
  retriever -> context assembly -> LLM -> structured response
"""

from __future__ import annotations

import time
import uuid

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise RuntimeError("Install API dependencies first: pip install -r requirements.txt") from exc


# ── Request / Response models ────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户的客服问题")
    images: list[str] = Field(default_factory=list, description="Base64 图片列表（可选）")
    session_id: str | None = Field(default=None, description="会话 ID，用于多轮对话")


class ReferenceItem(BaseModel):
    chunk_id: str = ""
    title: str = ""
    text_snippet: str = ""


class ResponseData(BaseModel):
    answer: str
    session_id: str
    image_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    references: list[ReferenceItem] = Field(default_factory=list)
    timestamp: int


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: ResponseData


# ── App factory ──────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(title="Industry Agent", version="0.1.0")

    @app.on_event("startup")
    def startup_event():
        from industry_agent.rag.retriever import SQLiteRetriever
        from industry_agent.agent.service import AgentService

        print("Initializing retriever & agent service ...")
        app.state.retriever = SQLiteRetriever()
        app.state.agent = AgentService(retriever=app.state.retriever)
        print("Ready.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest) -> ChatResponse:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")

        session_id = body.session_id or f"s_{uuid.uuid4().hex[:8]}"

        try:
            from industry_agent.agent.service import ChatRequest as SvcReq
            resp = app.state.agent.chat(
                SvcReq(
                    question=body.question,
                    images=body.images,
                    session_id=session_id,
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

        return ChatResponse(
            data=ResponseData(
                answer=resp.answer,
                session_id=session_id,
                image_ids=resp.image_ids,
                sources=resp.sources,
                references=[
                    ReferenceItem(**ref) for ref in resp.references
                ],
                timestamp=int(time.time()),
            )
        )

    return app
