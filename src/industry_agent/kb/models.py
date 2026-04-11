"""Data models used by the knowledge-base pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManualDocument:
    manual_id: str
    product_name: str
    source_path: Path
    text: str
    image_ids: list[str]
    pic_count: int
    parse_mode: str

    def to_record(self, chunk_count: int) -> dict[str, Any]:
        return {
            "manual_id": self.manual_id,
            "product_name": self.product_name,
            "source_path": str(self.source_path),
            "char_count": len(self.text),
            "pic_count": self.pic_count,
            "image_count": len(self.image_ids),
            "chunk_count": chunk_count,
            "parse_mode": self.parse_mode,
        }


@dataclass
class KnowledgeChunk:
    chunk_id: str
    manual_id: str
    product_name: str
    source_path: str
    title: str
    text: str
    image_ids: list[str]
    section_index: int
    chunk_index: int
    char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageRecord:
    image_id: str
    file_name: str | None
    path: str | None
    exists: bool
    referenced_by: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
