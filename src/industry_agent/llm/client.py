"""Unified LLM client for local Ollama and OpenAI-compatible cloud APIs."""

from __future__ import annotations

import re
from typing import Any

from industry_agent.config import settings

try:  # pragma: no cover - optional dependency
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore[assignment]


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


class LLMClient:
    """Backend-agnostic LLM client.

    Supported backends:
    - ``ollama``: local Ollama `/api/chat`
    - ``openai_compatible`` / ``api``: cloud API compatible with OpenAI SDK
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
    ) -> None:
        self.backend = (backend or settings.llm_backend).strip().lower()
        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or (
            settings.ollama_base_url if self.backend == "ollama" else settings.llm_base_url
        )
        self.model = model or (
            settings.ollama_model if self.backend == "ollama" else settings.llm_model
        )
        self.vision_model = vision_model or (
            settings.ollama_vision_model if self.backend == "ollama" else settings.llm_vision_model
        )
        self._client: Any | None = None

    @property
    def is_openai_compatible(self) -> bool:
        return self.backend in {"openai_compatible", "api"}

    @property
    def supports_multimodal(self) -> bool:
        return self.is_openai_compatible and bool(self.vision_model)

    @property
    def client(self) -> Any:
        if not self.is_openai_compatible:
            return None
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        model: str | None = None,
        strip_think: bool = True,
        system_prompt: str | None = None,
    ) -> str:
        final_messages: list[dict[str, str]] = []
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        if self.backend == "ollama":
            return self._chat_ollama(
                final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                strip_think=strip_think,
            )
        if self.is_openai_compatible:
            return self._chat_openai_compatible(
                final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                strip_think=strip_think,
            )
        raise ValueError(f"unsupported llm backend: {self.backend}")

    def chat_with_image(
        self,
        question: str,
        image_base64: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        if not self.is_openai_compatible:
            raise RuntimeError("chat_with_image currently requires an OpenAI-compatible backend.")

        image_data = image_base64.strip()
        if image_data.startswith("data:"):
            match = re.match(r"^data:[^;]+;base64,(.*)$", image_data, flags=re.DOTALL)
            if match:
                image_data = match.group(1).strip()

        content: list[dict[str, Any]] = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ]
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.vision_model or self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=0.95,
            stream=False,
        )
        return _strip_thinking(response.choices[0].message.content or "")

    def is_available(self) -> bool:
        try:
            if self.backend == "ollama":
                if httpx is None:
                    return False
                with httpx.Client(proxy=None, timeout=10.0) as client:
                    response = client.get(f"{self.base_url.rstrip('/')}/api/tags")
                    response.raise_for_status()
                return True
            if self.is_openai_compatible:
                if not self.api_key:
                    return False
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_completion_tokens=5,
                )
                return True
        except Exception:
            return False
        return False

    def _chat_ollama(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        model: str | None,
        strip_think: bool,
    ) -> str:
        if httpx is None:
            raise RuntimeError("httpx is required for Ollama backend.")
        with httpx.Client(proxy=None, timeout=120.0) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/api/chat",
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
        return _strip_thinking(content) if strip_think else content.strip()

    def _chat_openai_compatible(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        model: str | None,
        strip_think: bool,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False,
        )
        content = response.choices[0].message.content or ""
        return _strip_thinking(content) if strip_think else content.strip()
