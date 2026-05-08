"""Project-level paths and default settings."""

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by scripts and future services."""

    project_root: Path = PROJECT_ROOT
    knowledge_dir: Path = PROJECT_ROOT / "Knowledge_base"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed" / "kb"
    max_chunk_chars: int = 1200

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    vector_index_path: Path = PROJECT_ROOT / "data" / "processed" / "kb" / "vector.index"

    # Retrieval mode: "sqlite" | "vector" | "hybrid"
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")

    # LLM (OpenAI-compatible cloud API)
    llm_api_key: str = os.getenv("LLM_API_KEY", "sk-")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.xiaomimimo.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "mimo-v2.5-pro")
    llm_vision_model: str = os.getenv("LLM_VISION_MODEL", "mimo-v2.5-pro")

    @property
    def image_dir(self) -> Path:
        return self.knowledge_dir / "插图"


settings = Settings()
