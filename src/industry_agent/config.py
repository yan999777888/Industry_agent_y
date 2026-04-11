"""Project-level paths and default settings."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by scripts and future services."""

    project_root: Path = PROJECT_ROOT
    knowledge_dir: Path = PROJECT_ROOT / "Knowledge_base"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed" / "kb"
    max_chunk_chars: int = 1200

    @property
    def image_dir(self) -> Path:
        return self.knowledge_dir / "插图"


settings = Settings()
