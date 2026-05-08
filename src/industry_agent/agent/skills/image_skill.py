"""Image understanding skill — analyzes user-uploaded images.

Wraps the existing ImageUnderstander and enhances it with cloud vision API
support via the LLMClient.
"""

from __future__ import annotations

from typing import Any

from industry_agent.agent.skills import BaseSkill, SkillResult


class ImageSkill(BaseSkill):
    """Analyze uploaded product images to extract visual features for retrieval."""

    name = "image"
    description = "图像理解技能：分析用户上传的产品图片，提取部件、状态、故障等视觉特征"

    def __init__(self) -> None:
        self._understander = None

    @property
    def understander(self) -> Any:
        if self._understander is None:
            from industry_agent.agent.image_understanding import ImageUnderstander
            from industry_agent.config import settings
            from industry_agent.llm.client import LLMClient

            # Try cloud vision model first, fall back to Ollama
            try:
                llm = LLMClient()
                if llm.vision_model:
                    self._understander = ImageUnderstander(
                        base_url="",
                        http_client=None,
                        vision_model="",
                    )
                    self._cloud_llm = llm
                    return self._understander
            except Exception:
                pass

            # Fallback to Ollama
            try:
                import httpx

                from industry_agent.agent.service import (
                    OLLAMA_BASE_URL,
                    OLLAMA_VISION_MODEL,
                )

                self._understander = ImageUnderstander(
                    base_url=OLLAMA_BASE_URL,
                    http_client=httpx.Client(proxy=None, timeout=120.0),
                    vision_model=OLLAMA_VISION_MODEL,
                )
            except Exception:
                self._understander = ImageUnderstander(
                    base_url="",
                    http_client=None,
                    vision_model="",
                )
        return self._understander

    def execute(
        self,
        *,
        images: list[str] | None = None,
        question: str = "",
        **kwargs: Any,
    ) -> SkillResult:
        """Analyze images and return visual features.

        Args:
            images: List of Base64-encoded image strings.
            question: User's question for context.

        Returns:
            SkillResult with data=ImageUnderstandingResult.
        """
        if not images:
            return SkillResult(
                success=True,
                data=None,
                metadata={"has_images": False},
            )

        try:
            result = self.understander.analyze_images(images, question=question)
            return SkillResult(
                success=True,
                data=result,
                metadata={
                    "has_images": True,
                    "image_count": len(images),
                    "used_vision_model": result.used_vision_model,
                    "retrieval_terms": result.retrieval_terms,
                    "visual_features": result.visual_features,
                },
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                error=str(exc),
                metadata={"has_images": True, "image_count": len(images)},
            )

    def is_available(self) -> bool:
        try:
            _ = self.understander
            return True
        except Exception:
            return False
