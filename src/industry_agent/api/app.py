"""FastAPI application factory.

The API dependency is optional for now. Install `.[api]` before serving this app.
"""

from __future__ import annotations


def create_app():
    try:
        from fastapi import FastAPI
    except ImportError as exc:
        raise RuntimeError("Install API dependencies first: pip install -e '.[api]'") from exc

    app = FastAPI(title="Industry Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat() -> dict[str, str]:
        return {"message": "chat endpoint scaffold is ready; RAG orchestration is pending"}

    return app
