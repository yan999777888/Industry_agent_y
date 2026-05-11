from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - optional API dependency
    TestClient = None  # type: ignore[assignment]

from industry_agent.agent.image_understanding import ImageUnderstandingResult
from industry_agent.agent.runtime_checks import StartupHealthReport
from industry_agent.agent.service import AgentService, ChatRequest
from industry_agent.agent.skills import SkillResult

if TestClient is not None:
    from industry_agent.api import app as api_app_module
else:  # pragma: no cover - optional API dependency
    api_app_module = None


class _StubRetriever:
    def search(self, query: str, *, limit: int = 5) -> list[dict[str, str]]:
        return []


class _FakeLLMClient:
    init_kwargs: dict[str, object] | None = None
    last_chat: tuple[list[dict[str, str]], dict[str, object]] | None = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def chat(self, messages, **kwargs):
        type(self).last_chat = (messages, kwargs)
        return "cloud backend answer"


class _FakeImageSkill:
    last_kwargs: dict[str, object] | None = None
    init_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def execute(self, **kwargs):
        type(self).last_kwargs = kwargs
        return SkillResult(
            success=True,
            data=ImageUnderstandingResult(
                has_image_input=True,
                combined_summary="上传图片1：检测到指示灯区域",
                retrieval_hint="指示灯 红灯",
                retrieval_terms=["指示灯", "红灯"],
                visual_features={
                    "component_terms": ["指示灯"],
                    "status_terms": ["红灯"],
                    "issue_terms": [],
                    "other_terms": [],
                },
                used_vision_model="fake-vision",
            ),
        )


class BackendIntegrationTests(unittest.TestCase):
    def test_service_uses_unified_llm_client_for_cloud_backend(self) -> None:
        with patch("industry_agent.agent.service.LLMClient", _FakeLLMClient), patch(
            "industry_agent.agent.service.ImageSkill",
            _FakeImageSkill,
        ):
            agent = AgentService(
                retriever=_StubRetriever(),
                llm_backend="openai_compatible",
                base_url="https://api.example.com/v1",
                model="demo-model",
            )
            answer = agent._call_llm([{"role": "user", "content": "hello"}])

        self.assertEqual(answer, "cloud backend answer")
        self.assertEqual(_FakeLLMClient.init_kwargs["backend"], "openai_compatible")
        self.assertEqual(_FakeLLMClient.init_kwargs["base_url"], "https://api.example.com/v1")
        self.assertEqual(_FakeLLMClient.init_kwargs["model"], "demo-model")
        self.assertIsNotNone(_FakeLLMClient.last_chat)

    def test_service_uses_image_skill_for_uploaded_images(self) -> None:
        with patch("industry_agent.agent.service.ImageSkill", _FakeImageSkill):
            agent = AgentService(
                retriever=_StubRetriever(),
                llm_backend="openai_compatible",
                base_url="https://api.example.com/v1",
                model="demo-model",
            )
            result = agent._analyze_uploaded_images(
                ChatRequest(
                    question="这个指示灯是什么意思？",
                    images=["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBgN8L1n4AAAAASUVORK5CYII="],
                )
            )

        self.assertTrue(result.has_image_input)
        self.assertEqual(result.used_vision_model, "fake-vision")
        self.assertEqual(result.retrieval_terms, ["指示灯", "红灯"])
        self.assertEqual(_FakeImageSkill.init_kwargs["llm_backend"], "openai_compatible")
        self.assertEqual(_FakeImageSkill.last_kwargs["question"], "这个指示灯是什么意思？")

    @unittest.skipIf(TestClient is None, "fastapi is not installed")
    def test_health_endpoint_reports_selected_backends(self) -> None:
        fake_settings = SimpleNamespace(
            agent_backend="orchestrator",
            llm_backend="openai_compatible",
            ollama_base_url="http://localhost:11434",
            ollama_model="qwen3.5:2b",
            ollama_vision_model="llava-phi3",
            llm_base_url="https://api.example.com/v1",
            llm_model="demo-model",
            llm_vision_model="demo-vision",
        )
        report = StartupHealthReport(status="ok", components=[])

        with patch.object(api_app_module, "settings", fake_settings), patch.object(
            api_app_module,
            "run_startup_checks",
            return_value=report,
        ), patch.object(api_app_module, "assert_startup_ready"), patch(
            "industry_agent.agent.orchestrator.AgentOrchestrator",
            return_value=object(),
        ):
            app = api_app_module.create_app()
            with TestClient(app) as client:
                payload = client.get("/health").json()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["agent_backend"], "orchestrator")
        self.assertEqual(payload["llm_backend"], "openai_compatible")


if __name__ == "__main__":
    unittest.main()
