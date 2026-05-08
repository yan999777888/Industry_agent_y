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

from industry_agent.agent.runtime_checks import assert_startup_ready, run_startup_checks


# ── Request / Response models ────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户的客服问题")
    images: list[str] = Field(default_factory=list, description="Base64 图片列表（可选）")
    session_id: str | None = Field(default=None, description="会话 ID，用于多轮对话")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误详情")


class ReferenceItem(BaseModel):
    chunk_id: str = ""
    title: str = ""
    text_snippet: str = ""
    product_name: str = ""
    score: str = ""


class ImageItem(BaseModel):
    image_id: str = ""
    file_name: str = ""
    path: str = ""
    exists: bool = False


class ResponseData(BaseModel):
    answer: str
    session_id: str
    image_ids: list[str] = Field(default_factory=list)
    images: list[ImageItem] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    references: list[ReferenceItem] = Field(default_factory=list)
    confidence: float = 0.0
    timestamp: int


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: ResponseData


# ── App factory ──────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Industry Agent",
        version="0.1.0",
        description=(
            "面向工业产品客服场景的多模态问答服务。"
            "当前提供 `/health` 健康检查和 `/chat` 问答接口，"
            "支持说明书检索问答、轻量客服策略、多轮对话和图片理解。"
        ),
    )

    @app.on_event("startup")
    def startup_event():
        from industry_agent.rag.retriever import SQLiteRetriever
        from industry_agent.agent.service import AgentService

        print("Initializing retriever & agent service ...")
        report = run_startup_checks()
        app.state.health_report = report
        assert_startup_ready(report)
        app.state.retriever = SQLiteRetriever()
        app.state.agent = AgentService(retriever=app.state.retriever)
        print("Ready.")

    @app.get(
        "/health",
        summary="健康检查",
        description="返回知识库索引和 LLM API 的启动检查结果。",
    )
    def health() -> dict:
        report = getattr(app.state, "health_report", None)
        if report is None:
            return {"status": "unknown"}
        return report.to_dict()

    @app.post(
        "/chat",
        response_model=ChatResponse,
        summary="客服问答",
        description=(
            "接收一个问题和可选图片，返回结构化答案、相关图片、来源和置信度。"
            "接口同时支持说明书 RAG、客服策略路由、多轮上下文继承和图片辅助理解。"
        ),
        responses={
            400: {
                "model": ErrorResponse,
                "description": "请求参数错误，例如 question 为空。",
            },
            500: {
                "model": ErrorResponse,
                "description": "服务内部异常，例如对话编排或模型调用失败。",
            },
            503: {
                "model": ErrorResponse,
                "description": "依赖不可用，例如知识库索引不存在或启动检查未通过。",
            },
        },
    )
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
                images=[
                    ImageItem(**image) for image in resp.images
                ],
                sources=resp.sources,
                references=[
                    ReferenceItem(**ref) for ref in resp.references
                ],
                confidence=resp.confidence,
                timestamp=int(time.time()),
            )
        )

    return app
