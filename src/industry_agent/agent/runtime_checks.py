"""Runtime health checks for startup and diagnostics."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from industry_agent.config import settings

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


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
    base_url: str,
    model: str,
    vision_model: str = "",
    llm_backend: str = "ollama",
    api_key: str = "",
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

    backend = llm_backend.strip().lower()
    components.append(
        ComponentStatus(
            name="llm_backend",
            ok=backend in {"ollama", "openai_compatible", "api"},
            detail=backend or "unknown",
            required=True,
        )
    )

    if backend == "ollama":
        if httpx is None:
            components.append(
                ComponentStatus(
                    name="httpx",
                    ok=False,
                    detail="httpx not installed",
                    required=True,
                )
            )
        else:
            components.append(
                ComponentStatus(
                    name="ollama_base_url",
                    ok=bool(base_url),
                    detail=base_url or "(empty)",
                    required=True,
                )
            )
            try:
                with httpx.Client(proxy=None, timeout=10.0) as client:
                    resp = client.get(f"{base_url.rstrip('/')}/api/tags")
                    resp.raise_for_status()
                    payload = resp.json()
            except Exception as exc:
                components.append(
                    ComponentStatus(
                        name="ollama",
                        ok=False,
                        detail=str(exc),
                        required=True,
                    )
                )
            else:
                model_names = _extract_model_names(payload)
                components.append(
                    ComponentStatus(
                        name="ollama",
                        ok=True,
                        detail=base_url,
                        required=True,
                    )
                )
                components.append(
                    ComponentStatus(
                        name="text_model",
                        ok=model in model_names,
                        detail=model,
                        required=True,
                    )
                )
                if vision_model:
                    components.append(
                        ComponentStatus(
                            name="vision_model",
                            ok=vision_model in model_names,
                            detail=vision_model,
                            required=False,
                        )
                    )
    else:
        components.append(
            ComponentStatus(
                name="llm_base_url",
                ok=bool(base_url),
                detail=base_url or "(empty)",
                required=True,
            )
        )
        components.append(
            ComponentStatus(
                name="llm_api_key",
                ok=bool(api_key),
                detail="configured" if api_key else "missing",
                required=True,
            )
        )
        components.append(
            ComponentStatus(
                name="text_model",
                ok=bool(model),
                detail=model or "(empty)",
                required=True,
            )
        )
        if vision_model:
            components.append(
                ComponentStatus(
                    name="vision_model",
                    ok=True,
                    detail=vision_model,
                    required=False,
                )
            )

    # ── DashScope checks ──────────────────────────────────────────────
    if settings.dashscope_enabled:
        _check_dashscope(components)

    status = "ok" if all(component.ok or not component.required for component in components) else "degraded"
    return StartupHealthReport(status=status, components=components)


def _check_dashscope(components: list[ComponentStatus]) -> None:
    """Add DashScope-related health checks."""
    components.append(
        ComponentStatus(
            name="dashscope_api_key",
            ok=bool(settings.dashscope_api_key),
            detail="configured" if settings.dashscope_api_key else "DASHSCOPE_API_KEY not set",
            required=True,
        )
    )
    components.append(
        ComponentStatus(
            name="dashscope_llm",
            ok=bool(settings.dashscope_llm_model),
            detail=f"{settings.dashscope_llm_model} @ {settings.dashscope_base_url}",
            required=True,
        )
    )
    components.append(
        ComponentStatus(
            name="dashscope_embedding",
            ok=bool(settings.dashscope_embedding_model),
            detail=settings.dashscope_embedding_model,
            required=True,
        )
    )
    components.append(
        ComponentStatus(
            name="dashscope_rerank",
            ok=bool(settings.dashscope_rerank_model),
            detail=settings.dashscope_rerank_model,
            required=False,
        )
    )
    # Vector count check
    db_path = settings.processed_dir / "index.sqlite"
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            conn.close()
            vec_ok = count > 0
        except Exception:
            count = 0
            vec_ok = False
        components.append(
            ComponentStatus(
                name="chunk_vectors",
                ok=vec_ok,
                detail=f"{count} vectors" if vec_ok else "empty or missing",
                required=True,
            )
        )
    # BM25 check
    bm25_path = settings.processed_dir / "bm25_index.pkl"
    components.append(
        ComponentStatus(
            name="bm25_index",
            ok=bm25_path.exists(),
            detail=str(bm25_path) if bm25_path.exists() else "not built",
            required=False,
        )
    )


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


def _extract_model_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for item in payload.get("models", []) or []:
        name = str(item.get("name", "")).strip()
        model = str(item.get("model", "")).strip()
        if name:
            names.add(name)
            names.add(name.split(":")[0])
        if model:
            names.add(model)
            names.add(model.split(":")[0])
    return names
