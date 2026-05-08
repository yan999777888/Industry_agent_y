"""Runtime health checks for startup and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from industry_agent.config import settings
from industry_agent.llm.client import LLMClient


@dataclass
class ComponentStatus:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass
class StartupHealthReport:
    status: str
    components: list[ComponentStatus] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "components": [asdict(component) for component in self.components],
        }


def run_startup_checks(
    *,
    processed_dir: Path = settings.processed_dir,
) -> StartupHealthReport:
    components: list[ComponentStatus] = []

    index_path = processed_dir / "index.sqlite"
    components.append(
        ComponentStatus(
            name="index.sqlite",
            ok=index_path.exists(),
            detail=str(index_path),
            required=True,
        )
    )

    image_index_path = processed_dir / "images.jsonl"
    components.append(
        ComponentStatus(
            name="images.jsonl",
            ok=image_index_path.exists(),
            detail=str(image_index_path),
            required=False,
        )
    )

    # LLM API connectivity check (non-blocking — chat endpoint has its own error handling)
    try:
        client = LLMClient()
        api_ok = client.is_available()
    except Exception as exc:
        components.append(
            ComponentStatus(
                name="llm_api",
                ok=False,
                detail=str(exc),
                required=False,
            )
        )
    else:
        components.append(
            ComponentStatus(
                name="llm_api",
                ok=api_ok,
                detail=f"{settings.llm_base_url} | model: {settings.llm_model}",
                required=False,
            )
        )

    status = "ok" if all(component.ok or not component.required for component in components) else "degraded"
    return StartupHealthReport(status=status, components=components)


def assert_startup_ready(report: StartupHealthReport) -> None:
    failures = [
        component
        for component in report.components
        if component.required and not component.ok
    ]
    if not failures:
        return
    joined = "; ".join(f"{item.name}: {item.detail}" for item in failures)
    raise RuntimeError(f"startup checks failed: {joined}")
