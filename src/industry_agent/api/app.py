"""FastAPI application factory.

Competition API spec:
  - POST /chat  (唯一客服交互入口)
  - GET  /health (健康检查，无需认证)
  - Bearer Token 认证 (KAFU_API_TOKEN)
  - UTF-8, 20s text / 30s multimodal timeout
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise RuntimeError("Install API dependencies first: pip install -r requirements.txt") from exc

from industry_agent.agent.runtime_checks import assert_startup_ready, run_startup_checks
from industry_agent.config import settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ── Auth config ──────────────────────────────────────────────────────────
KAFU_API_TOKEN = os.getenv("KAFU_API_TOKEN", "")
security = HTTPBearer(auto_error=False)


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
    retrieval_debug: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: ResponseData


# ── Auth dependency ──────────────────────────────────────────────────────

async def verify_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    """Bearer Token 认证。/health 不走此依赖。"""
    # 如果未配置 token，跳过认证（开发模式）
    if not KAFU_API_TOKEN:
        return

    if credentials is None or credentials.credentials != KAFU_API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="未提供有效的认证令牌。请在 Header 中设置 Authorization: Bearer {KAFU_API_TOKEN}",
        )


# ── Timeout wrapper ──────────────────────────────────────────────────────

class RequestTimeoutMiddleware:
    """根据请求内容自动设置超时：文本 20s，多模态 30s。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # 检查是否有图片 → 多模态 30s，否则 20s
            request = Request(scope, receive)
            body = await request.body()
            has_images = b'"images"' in body and b'[]' not in body.split(b'"images"')[1][:20] if b'"images"' in body else False
            scope["state"] = scope.get("state", {})
            scope["state"]["timeout"] = 30 if has_images else 20
        await self.app(scope, receive, send)


# ── App factory ──────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Industry Agent - 客服智能体",
        version="1.0.0",
        description=(
            "面向工业产品客服场景的多模态问答服务。\n"
            "核心端点: POST /chat (文本/图片客服咨询)\n"
            "认证方式: Bearer Token (KAFU_API_TOKEN)\n"
            "接口超时: 20s (文本) / 30s (多模态)"
        ),
    )

    # CORS - 允许所有来源（比赛测试用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 静态文件 + 首页 ──────────────────────────────────────────────
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return HTMLResponse(content=index.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Industry Agent API</h1><p>GET /health | POST /chat</p>")

    @app.on_event("startup")
    def startup_event():
        logger.info("Initializing agent service ...")
        report = run_startup_checks(
            base_url=settings.ollama_base_url if settings.llm_backend == "ollama" else settings.llm_base_url,
            model=settings.ollama_model if settings.llm_backend == "ollama" else settings.llm_model,
            vision_model=settings.ollama_vision_model if settings.llm_backend == "ollama" else settings.llm_vision_model,
            llm_backend=settings.llm_backend,
            api_key=settings.llm_api_key,
        )
        app.state.health_report = report
        assert_startup_ready(report)
        app.state.agent_backend = settings.agent_backend
        if settings.agent_backend == "orchestrator":
            from industry_agent.agent.orchestrator import AgentOrchestrator
            app.state.agent = AgentOrchestrator()
        else:
            from industry_agent.agent.service import AgentService
            from industry_agent.rag.factory import create_retriever
            app.state.retriever = create_retriever()
            app.state.agent = AgentService(retriever=app.state.retriever)
        logger.info("Ready. Auth=%s", "enabled" if KAFU_API_TOKEN else "disabled (dev mode)")

    # ── 健康检查（无需认证） ──────────────────────────────────────────
    @app.get(
        "/health",
        summary="健康检查",
        description="返回索引、LLM 服务、模型的启动检查结果。无需认证。",
    )
    def health() -> dict:
        report = getattr(app.state, "health_report", None)
        if report is None:
            return {"status": "unknown"}
        payload = report.to_dict()
        payload["agent_backend"] = getattr(app.state, "agent_backend", settings.agent_backend)
        payload["llm_backend"] = settings.llm_backend
        retriever = getattr(app.state, "retriever", None)
        if retriever is not None and hasattr(retriever, "retrieval_status"):
            payload["retrieval"] = retriever.retrieval_status()
        return payload

    # ── 客服问答（需要认证） ──────────────────────────────────────────
    @app.post(
        "/chat",
        response_model=ChatResponse,
        summary="客服问答",
        description=(
            "接收一个问题和可选图片，返回结构化答案、相关图片、来源和置信度。\n"
            "请求头: Authorization: Bearer {KAFU_API_TOKEN}\n"
            "超时: 文本 20s / 多模态 30s"
        ),
        responses={
            400: {"model": ErrorResponse, "description": "请求参数错误"},
            401: {"model": ErrorResponse, "description": "认证失败，缺少或无效的 Bearer Token"},
            500: {"model": ErrorResponse, "description": "服务内部异常"},
            503: {"model": ErrorResponse, "description": "依赖不可用"},
        },
    )
    async def chat(
        body: ChatRequest,
        _auth=Depends(verify_token),
    ) -> ChatResponse:
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")

        session_id = body.session_id or f"s_{uuid.uuid4().hex[:8]}"

        try:
            import asyncio
            from industry_agent.agent.service import ChatRequest as SvcReq

            # 直接执行，不设超时
            resp = await asyncio.to_thread(
                app.state.agent.chat,
                SvcReq(
                    question=body.question,
                    images=body.images,
                    session_id=session_id,
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("chat failed")
            raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

        return ChatResponse(
            data=ResponseData(
                answer=resp.answer,
                session_id=session_id,
                image_ids=resp.image_ids,
                images=[ImageItem(**image) for image in resp.images],
                sources=resp.sources,
                references=[ReferenceItem(**ref) for ref in resp.references],
                confidence=resp.confidence,
                timestamp=int(time.time()),
                retrieval_debug=resp.retrieval_debug,
            )
        )

    # ── 图片服务 ─────────────────────────────────────────────────────
    @app.get("/images/{image_id}", summary="获取说明书图片")
    async def get_image(image_id: str):
        """根据 image_id 返回对应的插图文件。image_id 格式如 Manual11_1"""
        img_dir = settings.image_dir
        logger.info("Image request: id=%s dir=%s exists=%s", image_id, img_dir, img_dir.exists())
        # 尝试常见扩展名
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            path = img_dir / f"{image_id}{ext}"
            if path.exists():
                import mimetypes
                mime, _ = mimetypes.guess_type(str(path))
                return FileResponse(path, media_type=mime or "image/jpeg")
        raise HTTPException(status_code=404, detail=f"图片 {image_id} 不存在于 {img_dir}")

    # ── 全局异常处理 ──────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"code": 500, "msg": f"Internal server error: {exc}", "data": None},
        )

    return app
