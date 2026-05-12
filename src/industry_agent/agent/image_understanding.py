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


VISION_PROMPT_TEMPLATE = (
    "请根据这张用户上传的产品图片，输出简短描述，帮助客服检索相关说明书内容。要求：\n"
    "1. 只描述图片里可以直接观察到的内容，不要编造。\n"
    "2. 优先关注产品类型、品牌、型号、部件名称、按钮、指示灯颜色和状态、接口类型、屏幕显示内容、故障现象、安装位置。\n"
    "3. 如果能识别出产品型号或品牌，务必在描述中提及。\n"
    "4. 控制在 2-3 句话，使用中文。\n"
    "5. 如果看不清，请明确说明“图片信息有限”。\n"
    "\n"
    "用户问题：{question}\n"
)

_NOISY_VISUAL_TERMS: set[str] = {
    "图片", "图像", "画面", "设备", "这个", "那个",
    "其中", "一个", "一些", "可能", "显示",
    "看到", "部分", "区域", "部件", "位置",
    "东西", "起来", "相关", "用户", "上传",
    "内容", "信息", "里的", "中的", "具有",
    "通过", "可以", "用于", "以及", "还有",
    "这张", "该图像", "该设备", "一个电子设备",
    "电子", "电子设备",
}
_VISUAL_DOMAIN_HINTS: tuple[str, ...] = (
    "指示灯", "按钮", "接口", "屏幕",
    "电池", "充电", "开关", "表带",
    "卡扣", "旋钮", "插槽", "线缆",
    "红灯", "蓝灯", "闪烁",
    "裂纹", "破损", "划痕",
)
_VISUAL_COMPONENT_TERMS: tuple[str, ...] = (
    "指示灯", "按钮", "接口", "屏幕",
    "电池", "充电器", "电池组", "开关",
    "表带", "卡扣", "旋钮", "插槽",
    "线缆", "显示屏", "端口", "插头",
)
_VISUAL_STATUS_TERMS: tuple[str, ...] = (
    "红灯", "蓝灯", "绿灯", "闪烁",
    "发亮", "熄灭", "松动", "脱落",
    "安装", "拆卸", "充电", "断开",
    "连接", "锁定", "卡住",
)
_VISUAL_ISSUE_TERMS: tuple[str, ...] = (
    "裂纹", "破损", "划痕", "烧焦",
    "变形", "故障", "报警", "漏水",
    "异响", "发热", "过热", "污渍",
)


@dataclass(frozen=True)
class ImageObservation:
    image_index: int
    format: str
    mime_type: str
    file_size: int
    width: int = None  # type: ignore[assignment]
    height: int = None  # type: ignore[assignment]
    summary: str = ""
    visual_summary: str = ""
    source: str = "metadata"
    warning: str = ""


@dataclass(frozen=True)
class ImageUnderstandingResult:
    has_image_input: bool
    observations: list = field(default_factory=list)  # list[ImageObservation]
    combined_summary: str = ""
    retrieval_hint: str = ""
    retrieval_terms: list = field(default_factory=list)  # list[str]
    visual_features: dict = field(default_factory=dict)  # dict[str, list[str]]
    used_vision_model: str = ""
    warnings: list = field(default_factory=list)  # list[str]

    def to_debug_dict(self) -> dict:
        return {
            "has_image_input": self.has_image_input,
            "combined_summary": self.combined_summary,
            "retrieval_hint": self.retrieval_hint,
            "retrieval_terms": self.retrieval_terms,
            "visual_features": self.visual_features,
            "used_vision_model": self.used_vision_model,
            "warnings": self.warnings,
            "observations": [asdict(item) for item in self.observations],
        }


class ImageUnderstander:
    """Analyze uploaded images and optionally call a vision-capable Ollama model."""

    def __init__(
        self,
        *,
        base_url,
        http_client=None,
        vision_model=None,
        max_vision_images=2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.http_client = http_client
        self.vision_model = (vision_model or os.getenv("OLLAMA_VISION_MODEL", "")).strip()
        self.max_vision_images = max_vision_images

    def analyze_images(self, images, *, question=""):
        if not images:
            return ImageUnderstandingResult(has_image_input=False)

        observations = []
        warnings = []
        caption_inputs = []

        for index, raw_image in enumerate(images, start=1):
            decoded = _decode_base64_image(raw_image)
            if decoded is None:
                warning = "图片%d 不是有效的 Base64 图像数据" % index
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
        elif caption_inputs:
            warnings.append("视觉模型不可用，仅使用图片元数据进行检索")

        combined_summary = "；".join(
            _build_combined_summary_text(item) for item in observations if item.summary or item.visual_summary
        )
        visual_features = self._extract_visual_features(question=question, observations=observations)
        retrieval_terms = self._build_retrieval_terms(visual_features=visual_features)
        retrieval_hint = " ".join(retrieval_terms)
        return ImageUnderstandingResult(
            has_image_input=True,
            observations=observations,
            combined_summary=combined_summary,
            retrieval_hint=retrieval_hint,
            retrieval_terms=retrieval_terms,
            visual_features=visual_features,
            used_vision_model=used_vision_model,
            warnings=warnings,
        )

    def _can_use_vision(self):
        return bool(self.vision_model and self.http_client is not None)

    def _caption_image(self, base64_payload, *, question):
        if not self._can_use_vision():
            return ""
        try:
            response = self.http_client.post(
                "%s/api/generate" % self.base_url,
                json={
                    "model": self.vision_model,
                    "prompt": VISION_PROMPT_TEMPLATE.format(question=question or "请描述这张图片"),
                    "images": [base64_payload],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return ""

        content = str(payload.get("response", "")).strip()
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"^\d+\.\s*", "", content)
        content = re.sub(r"\s*\d+\.\s*", "；", content)
        content = re.sub(r"；+", "；", content)
        return content[:300]

    def _extract_visual_features(self, *, question, observations):
        visual_text = " ".join(
            item.visual_summary
            for item in observations
            if item.visual_summary
        )
        if not visual_text:
            return {
                "component_terms": [],
                "status_terms": [],
                "issue_terms": [],
                "other_terms": [],
            }
        cleaned_visual_text = _clean_visual_summary(visual_text)
        domain_terms = [term for term in _VISUAL_DOMAIN_HINTS if term in cleaned_visual_text]
        keywords = extract_keywords(cleaned_visual_text)
        filtered_keywords = [
            keyword
            for keyword in _unique([*domain_terms, *keywords])
            if _is_useful_visual_keyword(keyword, question=question)
        ]
        component_terms = [term for term in filtered_keywords if term in _VISUAL_COMPONENT_TERMS]
        status_terms = [term for term in filtered_keywords if term in _VISUAL_STATUS_TERMS]
        issue_terms = [term for term in filtered_keywords if term in _VISUAL_ISSUE_TERMS]
        other_terms = [
            term
            for term in filtered_keywords
            if term not in component_terms and term not in status_terms and term not in issue_terms
        ]
        return {
            "component_terms": component_terms[:4],
            "status_terms": status_terms[:4],
            "issue_terms": issue_terms[:4],
            "other_terms": other_terms[:4],
        }

    def _build_retrieval_terms(self, *, visual_features):
        return _unique(
            [
                *visual_features.get("component_terms", []),
                *visual_features.get("status_terms", []),
                *visual_features.get("issue_terms", []),
                *visual_features.get("other_terms", []),
            ]
        )[:8]

    def _build_retrieval_hint(self, *, question, observations):
        visual_features = self._extract_visual_features(question=question, observations=observations)
        return " ".join(self._build_retrieval_terms(visual_features=visual_features))


def _decode_base64_image(value):
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


def _detect_image_type(image_bytes):
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG", "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "JPEG", "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "GIF", "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "WEBP", "image/webp"
    return "UNKNOWN", "application/octet-stream"


def _read_image_size(image_bytes, format_name):
    if Image is not None:
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


def _read_jpeg_size(image_bytes):
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


def _read_webp_size(image_bytes):
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


def _build_metadata_summary(*, image_index, format_name, mime_type, file_size, width, height):
    size_text = "%.1fKB" % (file_size / 1024) if file_size >= 1024 else "%dB" % file_size
    dims_text = "%dx%d" % (width, height) if width and height else "未知尺寸"
    return "上传图片%d：格式 %s（%s），尺寸 %s，大小 %s" % (
        image_index, format_name, mime_type, dims_text, size_text
    )


def _build_combined_summary_text(observation):
    if observation.visual_summary:
        return "%s。视觉描述：%s" % (observation.summary, observation.visual_summary)
    return observation.summary


def _clean_visual_summary(text):
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(图片中|图像中|该图像显示了|该设备还有|可以通过连接线来控制|可能是)", " ", cleaned)
    cleaned = re.sub(r"[。；,，]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_useful_visual_keyword(keyword, *, question):
    term = keyword.strip()
    if not term or len(term) < 2:
        return False
    if term in _NOISY_VISUAL_TERMS:
        return False
    if any(noisy in term for noisy in ("图片", "图像", "设备", "电子")):
        return False
    if "图片" in term or "图像" in term:
        return False
    if re.fullmatch(r"[0-9A-Za-z]+", term) and len(term) < 3:
        return False
    return True


def _unique(values):
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
