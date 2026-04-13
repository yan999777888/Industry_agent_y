"""FastAPI application factory."""

from __future__ import annotations

from industry_agent.agent.service import ChatRequest, CustomerServiceAgent

try:
    from fastapi import Body, FastAPI, HTTPException
    from pydantic import BaseModel, Field
    _FASTAPI_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    _FASTAPI_IMPORT_ERROR = exc
    Body = None
    FastAPI = None
    HTTPException = None

    class BaseModel:  # type: ignore[override]
        pass

    def Field(*args, **kwargs):  # type: ignore[misc]
        return None


class ChatPayload(BaseModel):
    """Request body for `/chat`."""

    question: str = Field(..., min_length=1, description="User question in text form.")
    image_base64: str | None = Field(default=None, description="Optional base64-encoded image.")
    session_id: str | None = Field(default=None, description="Optional session id for future multi-turn use.")
    top_k: int = Field(default=5, ge=1, le=10, description="Retriever top-k.")


def create_app():
    if FastAPI is None or HTTPException is None or Body is None or _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError("Install API dependencies first: pip install -r requirements.txt") from _FASTAPI_IMPORT_ERROR

    agent = CustomerServiceAgent()
    app = FastAPI(title="Industry Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(payload: ChatPayload = Body(...)) -> dict[str, object]:
        try:
            response = agent.chat(
                ChatRequest(
                    question=payload.question,
                    image_base64=payload.image_base64,
                    session_id=payload.session_id,
                    top_k=payload.top_k,
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive API guard
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

        return response.to_record()

    return app
