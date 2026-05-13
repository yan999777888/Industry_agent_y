"""Project-level paths and default settings."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by scripts and future services."""

    project_root: Path = PROJECT_ROOT
    knowledge_dir: Path = PROJECT_ROOT / "Knowledge_base"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed" / "kb"
    max_chunk_chars: int = 1200
    agent_backend: str = os.getenv("INDUSTRY_AGENT_AGENT_BACKEND", "service")
    llm_backend: str = os.getenv("INDUSTRY_AGENT_LLM_BACKEND", "openai_compatible")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
    ollama_vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "llava-phi3")
    llm_api_key: str = os.getenv("LLM_API_KEY", "tp-cx5mi0rw3ehfudkdh44xxhpceik5nt1fctxi1phsn8x07jgy")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "mimo-v2.5-pro")
    llm_vision_model: str = os.getenv("LLM_VISION_MODEL", "mimo-v2.5-pro")
    retrieval_mode: str = _env("RETRIEVAL_MODE", "INDUSTRY_AGENT_RETRIEVAL_MODE", default="hybrid")
    embedding_model: str = _env("EMBEDDING_MODEL", "INDUSTRY_AGENT_EMBEDDING_MODEL", default="BAAI/bge-m3")
    vector_index_path: Path = PROJECT_ROOT / "data" / "processed" / "kb" / "vector.index"

    @property
    def image_dir(self) -> Path:
        return self.knowledge_dir / "插图"


settings = Settings()
