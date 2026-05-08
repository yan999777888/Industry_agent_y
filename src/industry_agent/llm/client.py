"""Unified LLM client for OpenAI-compatible cloud APIs.

Supports Xiaomi MiMo, DeepSeek, Kimi (Moonshot), and any other provider
that implements the OpenAI chat completions API.

Configuration via environment variables:
    LLM_API_KEY    — API key
    LLM_BASE_URL   — API base URL
    LLM_MODEL      — Model name
    LLM_VISION_MODEL — Vision-capable model (optional)

Usage:
    client = LLMClient()
    answer = client.chat([{"role": "user", "content": "你好"}])
"""

from __future__ import annotations

import re
from typing import Any

from industry_agent.config import settings


def _strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks from model output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


class LLMClient:
    """Cloud LLM client wrapping the OpenAI SDK.

    Works with any OpenAI-compatible API endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_base_url
        self.model = model or settings.llm_model
        self.vision_model = vision_model or settings.llm_vision_model
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=120.0,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        model: str | None = None,
        strip_think: bool = True,
        system_prompt: str | None = None,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (MiMo default: 1.0).
            max_tokens: Maximum tokens to generate.
            model: Override model name for this call.
            strip_think: Whether to strip thinking blocks from response.
            system_prompt: Optional system message to prepend.

        Returns:
            The assistant's reply text.
        """
        final_messages: list[dict[str, str]] = []

        # Prepend system message if provided and not already present
        if system_prompt:
            if not messages or messages[0].get("role") != "system":
                final_messages.append({"role": "system", "content": system_prompt})

        final_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=final_messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False,
        )
        content = response.choices[0].message.content or ""
        if strip_think:
            content = _strip_thinking(content)
        return content.strip() if content.strip() else "模型未返回有效回答。"

    def chat_with_image(
        self,
        question: str,
        image_base64: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ) -> str:
        """Send a multimodal request with an image.

        Args:
            question: Text question about the image.
            image_base64: Base64-encoded image string.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.

        Returns:
            The model's description/analysis of the image.
        """
        model = self.vision_model or self.model

        # Clean base64 input
        image_data = image_base64.strip()
        if image_data.startswith("data:"):
            match = re.match(r"^data:[^;]+;base64,(.*)$", image_data, flags=re.DOTALL)
            if match:
                image_data = match.group(1).strip()

        content: list[dict[str, Any]] = []
        content.append({"type": "text", "text": question})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        })

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=0.95,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()

    def is_available(self) -> bool:
        """Check if the LLM API is reachable."""
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=5,
            )
            return True
        except Exception:
            return False
