"""Build cleaned chunks and local search indexes from the raw knowledge base."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from industry_agent.config import settings
from industry_agent.kb.chunker import chunk_manual
from industry_agent.kb.index_store import build_sqlite_index, write_json, write_jsonl
from industry_agent.kb.models import ImageRecord, KnowledgeChunk, ManualDocument
from industry_agent.kb.parser import attach_image_markers, load_manual


def build_knowledge_base(
    knowledge_dir: Path = settings.knowledge_dir,
    output_dir: Path = settings.processed_dir,
    *,
    max_chunk_chars: int = settings.max_chunk_chars,
) -> dict[str, Any]:
    """Run the full clean -> chunk -> index pipeline."""

    project_root = settings.project_root
    manual_paths = sorted(knowledge_dir.glob("*.txt"))
    image_dir = knowledge_dir / "插图"
    image_files = _scan_image_files(image_dir, project_root)

    manuals: list[ManualDocument] = []
    chunks: list[KnowledgeChunk] = []
    manual_records: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    image_to_manuals: dict[str, set[str]] = defaultdict(set)
    image_to_chunks: dict[str, set[str]] = defaultdict(set)
    actual_missing_images: dict[tuple[str, str], None] = {}

    for manual_path in manual_paths:
        manual = load_manual(manual_path)
        manuals.append(manual)

        marked_text, attached_image_ids, unmatched_pic_count = attach_image_markers(
            manual.text,
            manual.image_ids,
        )
        if manual.pic_count != len(manual.image_ids):
            warnings.append(
                {
                    "type": "pic_image_count_mismatch",
                    "manual_id": manual.manual_id,
                    "pic_count": manual.pic_count,
                    "image_count": len(manual.image_ids),
                    "unmatched_pic_count": unmatched_pic_count,
                    "extra_image_count": max(len(manual.image_ids) - manual.pic_count, 0),
                }
            )

        for image_id in attached_image_ids:
            image_to_manuals[image_id].add(manual.manual_id)
            if image_id not in image_files:
                actual_missing_images[(manual.manual_id, image_id)] = None

        manual_chunks = chunk_manual(
            manual,
            marked_text,
            project_root=project_root,
            max_chars=max_chunk_chars,
        )
        chunks.extend(manual_chunks)
        manual_records.append(manual.to_record(chunk_count=len(manual_chunks)))

        for chunk in manual_chunks:
            for image_id in chunk.image_ids:
                image_to_chunks[image_id].add(chunk.chunk_id)

    image_records = _build_image_records(
        image_files=image_files,
        image_to_manuals=image_to_manuals,
        image_to_chunks=image_to_chunks,
    )
    warnings.extend(
        {
            "type": "missing_image_file",
            "manual_id": manual_id,
            "image_id": image_id,
        }
        for manual_id, image_id in sorted(actual_missing_images)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manuals.json", manual_records)
    write_jsonl(output_dir / "chunks.jsonl", [chunk.to_record() for chunk in chunks])
    write_jsonl(output_dir / "images.jsonl", [image.to_record() for image in image_records])
    sqlite_summary = build_sqlite_index(
        output_dir / "index.sqlite",
        chunks=chunks,
        images=image_records,
        manual_records=manual_records,
    )

    summary = {
        "knowledge_dir": str(knowledge_dir.relative_to(project_root)),
        "output_dir": str(output_dir.relative_to(project_root)),
        "manual_count": len(manuals),
        "chunk_count": len(chunks),
        "referenced_image_count": len(image_to_manuals),
        "disk_image_count": len(image_files),
        "missing_image_count": sum(1 for item in warnings if item["type"] == "missing_image_file"),
        "mismatch_count": sum(1 for item in warnings if item["type"] == "pic_image_count_mismatch"),
        "max_chunk_chars": max_chunk_chars,
        "sqlite": sqlite_summary,
        "warnings": warnings,
    }
    write_json(output_dir / "build_summary.json", summary)
    return summary


def _scan_image_files(image_dir: Path, project_root: Path) -> dict[str, dict[str, str]]:
    image_files: dict[str, dict[str, str]] = {}
    if not image_dir.exists():
        return image_files
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image_files[path.stem] = {
            "file_name": path.name,
            "path": str(path.relative_to(project_root)),
        }
    return image_files


def _build_image_records(
    *,
    image_files: dict[str, dict[str, str]],
    image_to_manuals: dict[str, set[str]],
    image_to_chunks: dict[str, set[str]],
) -> list[ImageRecord]:
    all_image_ids = sorted(set(image_files) | set(image_to_manuals) | set(image_to_chunks))
    records: list[ImageRecord] = []
    for image_id in all_image_ids:
        file_info = image_files.get(image_id)
        records.append(
            ImageRecord(
                image_id=image_id,
                file_name=file_info["file_name"] if file_info else None,
                path=file_info["path"] if file_info else None,
                exists=bool(file_info),
                referenced_by=sorted(image_to_manuals.get(image_id, set())),
                chunk_ids=sorted(image_to_chunks.get(image_id, set())),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned knowledge-base chunks and indexes.")
    parser.add_argument("--knowledge-dir", type=Path, default=settings.knowledge_dir)
    parser.add_argument("--output-dir", type=Path, default=settings.processed_dir)
    parser.add_argument("--max-chunk-chars", type=int, default=settings.max_chunk_chars)
    args = parser.parse_args()

    summary = build_knowledge_base(
        knowledge_dir=args.knowledge_dir,
        output_dir=args.output_dir,
        max_chunk_chars=args.max_chunk_chars,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
