"""User-uploaded image understanding helpers."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import struct
from dataclasses import asdict, dataclass, field
from typing import Any

from industry_agent.rag.retriever import extract_keywords

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]


VISION_PROMPT_TEMPLATE = """\
请根据这张用户上传的产品图片，输出简短描述，帮助客服检索相关说明书内容。要求：
1. 只描述图片里可以直接观察到的内容，不要编造。
2. 优先关注产品类型、部件、按钮、指示灯、接口、屏幕、故障现象、安装位置。
3. 控制在 1-2 句话，使用中文。
4. 如果看不清，请明确说明“图片信息有限”。

用户问题：{question}
"""


@dataclass(frozen=True)
class ImageObservation:
    image_index: int
    format: str
    mime_type: str
    file_size: int
    width: int | None = None
    height: int | None = None
    summary: str = ""
    visual_summary: str = ""
    source: str = "metadata"
    warning: str = ""


@dataclass(frozen=True)
class ImageUnderstandingResult:
    has_image_input: bool
    observations: list[ImageObservation] = field(default_factory=list)
    combined_summary: str = ""
    retrieval_hint: str = ""
    used_vision_model: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "has_image_input": self.has_image_input,
            "combined_summary": self.combined_summary,
            "retrieval_hint": self.retrieval_hint,
            "used_vision_model": self.used_vision_model,
            "warnings": self.warnings,
            "observations": [asdict(item) for item in self.observations],
        }


class ImageUnderstander:
    """Analyze uploaded images and optionally call a vision-capable Ollama model."""

    def __init__(
        self,
        *,
        base_url: str,
        http_client: Any = None,
        vision_model: str | None = None,
        max_vision_images: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client
        self.vision_model = (vision_model or os.getenv("OLLAMA_VISION_MODEL", "")).strip()
        self.max_vision_images = max_vision_images

    def analyze_images(self, images: list[str] | None, *, question: str = "") -> ImageUnderstandingResult:
        if not images:
            return ImageUnderstandingResult(has_image_input=False)

        observations: list[ImageObservation] = []
        warnings: list[str] = []
        caption_inputs: list[tuple[int, str]] = []

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
            observations.append(
                ImageObservation(
                    image_index=index,
                    format=format_name,
                    mime_type=mime_type,
                    file_size=len(image_bytes),
                    width=width,
                    height=height,
                    summary=summary,
                )
            )
            if normalized_base64 and len(caption_inputs) < self.max_vision_images:
                caption_inputs.append((index, normalized_base64))

        used_vision_model = ""
        if self._can_use_vision():
            used_vision_model = self.vision_model
            for image_index, base64_payload in caption_inputs:
                caption = self._caption_image(base64_payload, question=question)
                if not caption:
                    continue
                observation = observations[image_index - 1]
                observations[image_index - 1] = ImageObservation(
                    **{
                        **asdict(observation),
                        "visual_summary": caption,
                        "source": "ollama_vision",
                    }
                )

        combined_summary = "；".join(
            _build_combined_summary_text(item) for item in observations if item.summary or item.visual_summary
        )
        retrieval_hint = self._build_retrieval_hint(question=question, observations=observations)
        return ImageUnderstandingResult(
            has_image_input=True,
            observations=observations,
            combined_summary=combined_summary,
            retrieval_hint=retrieval_hint,
            used_vision_model=used_vision_model,
            warnings=warnings,
        )

    def _can_use_vision(self) -> bool:
        return bool(self.vision_model and self.http_client is not None)

    def _caption_image(self, base64_payload: str, *, question: str) -> str:
        if not self._can_use_vision():
            return ""
        try:
            response = self.http_client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": VISION_PROMPT_TEMPLATE.format(question=question or "请描述这张图片"),
                    "images": [base64_payload],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 160,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return ""

        content = str(payload.get("response", "")).strip()
        content = re.sub(r"\s+", " ", content)
        return content[:200]

    def _build_retrieval_hint(self, *, question: str, observations: list[ImageObservation]) -> str:
        visual_text = " ".join(
            item.visual_summary
            for item in observations
            if item.visual_summary
        )
        if not visual_text:
            return ""
        keywords = extract_keywords(f"{question} {visual_text}")
        return " ".join(_unique(keywords)[:12])


def _decode_base64_image(value: str) -> tuple[bytes, str] | None:
    text = value.strip()
    if not text:
        return None
    if text.startswith("data:"):
        match = re.match(r"^data:[^;]+;base64,(.*)$", text, flags=re.DOTALL)
        if not match:
            return None
        text = match.group(1).strip()
    text = re.sub(r"\s+", "", text)
    padding = len(text) % 4
    if padding:
        text += "=" * (4 - padding)
    try:
        return base64.b64decode(text, validate=True), text
    except (binascii.Error, ValueError):
        return None


def _detect_image_type(image_bytes: bytes) -> tuple[str, str]:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG", "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "JPEG", "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "GIF", "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "WEBP", "image/webp"
    return "UNKNOWN", "application/octet-stream"


def _read_image_size(image_bytes: bytes, format_name: str) -> tuple[int | None, int | None]:
    if Image is not None:  # pragma: no branch - tiny utility
        try:
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as image:
                return int(image.width), int(image.height)
        except Exception:
            pass

    if format_name == "PNG" and len(image_bytes) >= 24:
        width, height = struct.unpack(">II", image_bytes[16:24])
        return int(width), int(height)
    if format_name == "GIF" and len(image_bytes) >= 10:
        width, height = struct.unpack("<HH", image_bytes[6:10])
        return int(width), int(height)
    if format_name == "JPEG":
        return _read_jpeg_size(image_bytes)
    if format_name == "WEBP":
        return _read_webp_size(image_bytes)
    return None, None


def _read_jpeg_size(image_bytes: bytes) -> tuple[int | None, int | None]:
    offset = 2
    length = len(image_bytes)
    while offset + 9 < length:
        if image_bytes[offset] != 0xFF:
            offset += 1
            continue
        marker = image_bytes[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        if offset + 2 > length:
            break
        segment_length = struct.unpack(">H", image_bytes[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > length:
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if offset + 7 <= length:
                height, width = struct.unpack(">HH", image_bytes[offset + 3 : offset + 7])
                return int(width), int(height)
            break
        offset += segment_length
    return None, None


def _read_webp_size(image_bytes: bytes) -> tuple[int | None, int | None]:
    if len(image_bytes) < 30:
        return None, None
    chunk_type = image_bytes[12:16]
    if chunk_type == b"VP8X" and len(image_bytes) >= 30:
        width = 1 + int.from_bytes(image_bytes[24:27], "little")
        height = 1 + int.from_bytes(image_bytes[27:30], "little")
        return width, height
    if chunk_type == b"VP8 " and len(image_bytes) >= 30:
        width, height = struct.unpack("<HH", image_bytes[26:30])
        return width & 0x3FFF, height & 0x3FFF
    return None, None


def _build_metadata_summary(
    *,
    image_index: int,
    format_name: str,
    mime_type: str,
    file_size: int,
    width: int | None,
    height: int | None,
) -> str:
    size_text = f"{file_size / 1024:.1f}KB" if file_size >= 1024 else f"{file_size}B"
    dims_text = f"{width}x{height}" if width and height else "未知尺寸"
    return f"上传图片{image_index}：格式 {format_name}（{mime_type}），尺寸 {dims_text}，大小 {size_text}"


def _build_combined_summary_text(observation: ImageObservation) -> str:
    if observation.visual_summary:
        return f"{observation.summary}。视觉描述：{observation.visual_summary}"
    return observation.summary


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
