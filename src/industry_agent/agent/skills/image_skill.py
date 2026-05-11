"""Image understanding skill — supports local Ollama and cloud vision APIs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from industry_agent.agent.image_understanding import (
    ImageObservation,
    ImageUnderstandingResult,
    ImageUnderstander,
    _build_combined_summary_text,
    _build_metadata_summary,
    _decode_base64_image,
    _detect_image_type,
    _read_image_size,
)
from industry_agent.agent.skills import BaseSkill, SkillResult
from industry_agent.config import settings
from industry_agent.llm.client import LLMClient

try:  # pragma: no cover - optional dependency
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore[assignment]


class ImageSkill(BaseSkill):
    name = "image"
    description = "图像理解技能：分析用户上传的产品图片，提取视觉特征"

    def __init__(
        self,
        *,
        llm_backend: str | None = None,
        ollama_base_url: str | None = None,
        ollama_vision_model: str | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_vision_model: str | None = None,
    ) -> None:
        self.llm_backend = (llm_backend or settings.llm_backend).strip().lower()
        self.ollama_base_url = (ollama_base_url or settings.ollama_base_url).strip()
        self.ollama_vision_model = (ollama_vision_model or settings.ollama_vision_model).strip()
        self.llm_base_url = (llm_base_url or settings.llm_base_url).strip()
        self.llm_model = (llm_model or settings.llm_model).strip()
        self.llm_vision_model = (llm_vision_model or settings.llm_vision_model).strip()
        self._understander: ImageUnderstander | None = None
        self._cloud_llm: LLMClient | None = None

    @property
    def understander(self) -> ImageUnderstander:
        if self._understander is None:
            client = None
            if httpx is not None:
                client = httpx.Client(proxy=None, timeout=120.0)
            self._understander = ImageUnderstander(
                base_url=self.ollama_base_url,
                http_client=client,
                vision_model=self.ollama_vision_model,
            )
        return self._understander

    @property
    def cloud_llm(self) -> LLMClient:
        if self._cloud_llm is None:
            self._cloud_llm = LLMClient(
                backend="openai_compatible",
                base_url=self.llm_base_url,
                model=self.llm_model,
                vision_model=self.llm_vision_model,
            )
        return self._cloud_llm

    def execute(
        self,
        *,
        images: list[str] | None = None,
        question: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        if not images:
            return SkillResult(success=True, data=ImageUnderstandingResult(has_image_input=False), metadata={"has_images": False})
        try:
            if self.cloud_llm.supports_multimodal and self.llm_backend in {"openai_compatible", "api"}:
                result = self._analyze_with_cloud(images, question=question)
            else:
                result = self.understander.analyze_images(images, question=question)
            return SkillResult(
                success=True,
                data=result,
                metadata={
                    "has_images": True,
                    "image_count": len(images),
                    "used_vision_model": result.used_vision_model,
                },
            )
        except Exception as exc:
            return SkillResult(success=False, error=str(exc), metadata={"has_images": True, "image_count": len(images)})

    def _analyze_with_cloud(self, images: list[str], *, question: str) -> ImageUnderstandingResult:
        observations: list[ImageObservation] = []
        warnings: list[str] = []
        for index, raw_image in enumerate(images, start=1):
            decoded = _decode_base64_image(raw_image)
            if decoded is None:
                warning = f"图片{index} 不是有效的 Base64 图像数据"
                observations.append(
                    ImageObservation(
                        image_index=index,
                        format="UNKNOWN",
                        mime_type="application/octet-stream",
                        file_size=0,
                        summary=warning,
                        warning=warning,
                        source="invalid",
                    )
                )
                warnings.append(warning)
                continue
            image_bytes, normalized_base64 = decoded
            format_name, mime_type = _detect_image_type(image_bytes)
            width, height = _read_image_size(image_bytes, format_name)
            summary = _build_metadata_summary(
                image_index=index,
                format_name=format_name,
                mime_type=mime_type,
                file_size=len(image_bytes),
                width=width,
                height=height,
            )
            visual_summary = ""
            try:
                visual_summary = self.cloud_llm.chat_with_image(
                    question or "请描述这张用户上传的产品图片，重点关注部件、按钮、指示灯、接口和故障现象。",
                    normalized_base64,
                    max_tokens=160,
                )
            except Exception:
                visual_summary = ""
            observations.append(
                ImageObservation(
                    image_index=index,
                    format=format_name,
                    mime_type=mime_type,
                    file_size=len(image_bytes),
                    width=width,
                    height=height,
                    summary=summary,
                    visual_summary=visual_summary,
                    source="cloud_vision" if visual_summary else "metadata",
                )
            )

        combined_summary = "；".join(
            _build_combined_summary_text(item) for item in observations if item.summary or item.visual_summary
        )
        visual_features = self.understander._extract_visual_features(question=question, observations=observations)  # type: ignore[attr-defined]
        retrieval_terms = self.understander._build_retrieval_terms(visual_features=visual_features)  # type: ignore[attr-defined]
        return ImageUnderstandingResult(
            has_image_input=True,
            observations=observations,
            combined_summary=combined_summary,
            retrieval_hint=" ".join(retrieval_terms),
            retrieval_terms=retrieval_terms,
            visual_features=visual_features,
            used_vision_model=self.cloud_llm.vision_model or "",
            warnings=warnings,
        )
