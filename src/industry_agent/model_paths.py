"""Helpers for resolving local Hugging Face model directories."""

from __future__ import annotations

import os
from pathlib import Path

from industry_agent.config import settings


def slugify_model_name(model_name: str) -> str:
    """Convert a repo id like ``BAAI/bge-m3`` into a safe local directory name."""
    normalized = model_name.strip().strip("/")
    return normalized.replace("/", "--")


def model_dir_for_name(model_name: str, *, models_root: Path | None = None) -> Path:
    root = models_root or settings.models_dir
    return root / slugify_model_name(model_name)


def resolve_local_model_path(model_name: str, *, models_root: Path | None = None) -> str:
    """Return local path when a downloaded model exists, otherwise original name."""
    normalized = model_name.strip()
    if not normalized:
        return normalized

    explicit_path = Path(normalized).expanduser()
    if explicit_path.exists():
        return str(explicit_path.resolve())

    candidate = model_dir_for_name(normalized, models_root=models_root)
    if candidate.exists():
        return str(candidate.resolve())
    return normalized


def prefer_local_model_env(model_name: str) -> str:
    """Resolve via env override first, then by ``./models`` fallback."""
    override = os.getenv("INDUSTRY_AGENT_MODELS_DIR", "").strip()
    models_root = Path(override).expanduser() if override else None
    return resolve_local_model_path(model_name, models_root=models_root)
