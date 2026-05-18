"""Build cleaned chunks and local search indexes from the raw knowledge base."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from industry_agent.config import settings
from industry_agent.kb.chunker import chunk_manual
from industry_agent.kb.index_store import build_sqlite_index, write_json, write_jsonl
from industry_agent.kb.models import ImageRecord, KnowledgeChunk, ManualDocument
from industry_agent.kb.parser import attach_image_markers, load_manuals


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
        for manual in load_manuals(manual_path):
            manuals.append(manual)

            attachment = attach_image_markers(
                manual.text,
                manual.image_ids,
            )
            marked_text = attachment.marked_text
            attached_image_ids = attachment.attached_image_ids
            unmatched_pic_count = attachment.unmatched_pic_count
            if manual.pic_count != len(manual.image_ids):
                warnings.append(
                    {
                        "type": "pic_image_count_mismatch",
                        "manual_id": manual.manual_id,
                        "pic_count": manual.pic_count,
                        "image_count": len(manual.image_ids),
                        "unmatched_pic_count": unmatched_pic_count,
                        "extra_image_count": max(len(manual.image_ids) - manual.pic_count, 0),
                        "attachment_strategy": attachment.strategy,
                        "attached_image_count": attachment.attached_count,
                        "suppressed_image_count": attachment.suppressed_image_count,
                    }
                )
            if attachment.strategy != "sequential":
                warnings.append(
                    {
                        "type": "image_attachment_strategy_applied",
                        "manual_id": manual.manual_id,
                        "attachment_strategy": attachment.strategy,
                        "pic_count": attachment.pic_count,
                        "image_count": attachment.image_count,
                        "attached_image_count": attachment.attached_count,
                        "suppressed_image_count": attachment.suppressed_image_count,
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
            manual_record = manual.to_record(chunk_count=len(manual_chunks))
            manual_record.update(
                {
                    "attached_image_count": attachment.attached_count,
                    "unmatched_pic_count": unmatched_pic_count,
                    "pic_coverage_ratio": round(
                        (len(manual.image_ids) / manual.pic_count), 4
                    ) if manual.pic_count else 1.0,
                    "attachment_strategy": attachment.strategy,
                    "suppressed_image_count": attachment.suppressed_image_count,
                }
            )
            manual_records.append(manual_record)

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
    english_summary_segments = build_english_summary_segments(chunks)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manuals.json", manual_records)
    write_jsonl(output_dir / "chunks.jsonl", [chunk.to_record() for chunk in chunks])
    write_jsonl(output_dir / "images.jsonl", [image.to_record() for image in image_records])
    write_json(output_dir / "english_summary_segments.json", english_summary_segments)
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
        "chunk_quality": summarize_chunk_quality(chunks, max_chunk_chars=max_chunk_chars),
        "manual_quality": summarize_manual_quality(manuals, chunks),
        "english_summary_segments": english_summary_segments,
        "english_summary_segment_quality": summarize_english_summary_segment_quality(english_summary_segments),
        "sqlite": sqlite_summary,
        "warnings": warnings,
    }
    write_json(output_dir / "build_summary.json", summary)
    return summary


def summarize_chunk_quality(chunks: list[KnowledgeChunk], *, max_chunk_chars: int) -> dict[str, Any]:
    if not chunks:
        return {
            "avg_char_count": 0.0,
            "median_char_count": 0,
            "max_char_count": 0,
            "min_char_count": 0,
            "near_limit_ratio": 0.0,
            "oversized_chunk_ratio": 0.0,
            "with_image_ratio": 0.0,
            "low_clean_score_ratio": 0.0,
            "overlap_context_ratio": 0.0,
            "heading_only_ratio": 0.0,
            "long_title_ratio": 0.0,
            "chunk_type_counts": {},
            "semantic_type_counts": {},
            "domain_label_counts": {},
            "step_count_distribution": {},
        }

    char_counts = sorted(chunk.char_count for chunk in chunks)
    chunk_type_counts: Counter[str] = Counter()
    semantic_type_counts: Counter[str] = Counter()
    domain_label_counts: Counter[str] = Counter()
    step_count_distribution: Counter[str] = Counter()
    with_image = 0
    low_clean_score = 0
    overlap_context = 0
    near_limit = 0
    oversized = 0
    heading_only = 0
    long_title = 0

    for chunk in chunks:
        metadata = chunk.metadata or {}
        chunk_type = str(metadata.get("chunk_type") or metadata.get("semantic_type") or "general")
        semantic_type = str(metadata.get("semantic_type") or "general")
        domain_label = str(metadata.get("domain_label") or "")
        step_count = int(metadata.get("step_count") or 0)
        clean_score = float(metadata.get("clean_score") or 0.0)

        chunk_type_counts[chunk_type] += 1
        semantic_type_counts[semantic_type] += 1
        if domain_label:
            domain_label_counts[domain_label] += 1
        if step_count <= 0:
            step_count_distribution["0"] += 1
        elif step_count == 1:
            step_count_distribution["1"] += 1
        elif step_count <= 3:
            step_count_distribution["2-3"] += 1
        else:
            step_count_distribution["4+"] += 1
        if chunk.image_ids:
            with_image += 1
        if clean_score < 0.7:
            low_clean_score += 1
        if bool(metadata.get("has_overlap_context")):
            overlap_context += 1
        if chunk.char_count >= int(max_chunk_chars * 0.85):
            near_limit += 1
        if chunk.char_count > max_chunk_chars:
            oversized += 1
        if len(chunk.title) > 60:
            long_title += 1
        stripped_lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        heading_body = stripped_lines[0].lstrip("# ").strip() if stripped_lines else ""
        if (
            not chunk.image_ids
            and len(stripped_lines) == 1
            and stripped_lines[0].startswith("#")
            and len(heading_body) <= 20
            and not any(mark in heading_body for mark in ("。", "！", "？", ".", "!", "?", "：", ":"))
        ):
            heading_only += 1

    return {
        "avg_char_count": round(sum(char_counts) / len(char_counts), 2),
        "median_char_count": _median(char_counts),
        "max_char_count": max(char_counts),
        "min_char_count": min(char_counts),
        "near_limit_ratio": round(near_limit / len(chunks), 4),
        "oversized_chunk_ratio": round(oversized / len(chunks), 4),
        "with_image_ratio": round(with_image / len(chunks), 4),
        "low_clean_score_ratio": round(low_clean_score / len(chunks), 4),
        "overlap_context_ratio": round(overlap_context / len(chunks), 4),
        "heading_only_ratio": round(heading_only / len(chunks), 4),
        "long_title_ratio": round(long_title / len(chunks), 4),
        "chunk_type_counts": dict(chunk_type_counts.most_common()),
        "semantic_type_counts": dict(semantic_type_counts.most_common()),
        "domain_label_counts": dict(domain_label_counts.most_common()),
        "step_count_distribution": dict(step_count_distribution),
    }


def summarize_manual_quality(manuals: list[ManualDocument], chunks: list[KnowledgeChunk]) -> dict[str, Any]:
    chunk_count_by_manual: Counter[str] = Counter(chunk.manual_id for chunk in chunks)
    image_chunk_count_by_manual: Counter[str] = Counter(
        chunk.manual_id for chunk in chunks if chunk.image_ids
    )
    parse_mode_counts: Counter[str] = Counter(manual.parse_mode for manual in manuals)
    avg_chunk_chars_by_manual: dict[str, float] = {}
    attachment_outliers: list[dict[str, Any]] = []
    for manual in manuals:
        manual_chunks = [chunk.char_count for chunk in chunks if chunk.manual_id == manual.manual_id]
        avg_chunk_chars_by_manual[manual.manual_id] = round(
            sum(manual_chunks) / len(manual_chunks), 2
        ) if manual_chunks else 0.0
        if manual.pic_count:
            coverage_ratio = round(len(manual.image_ids) / manual.pic_count, 4)
            if coverage_ratio < 0.7 or abs(manual.pic_count - len(manual.image_ids)) >= 20:
                attachment_outliers.append(
                    {
                        "manual_id": manual.manual_id,
                        "pic_count": manual.pic_count,
                        "image_count": len(manual.image_ids),
                        "pic_coverage_ratio": coverage_ratio,
                        "parse_mode": manual.parse_mode,
                    }
                )

    top_chunk_manuals = [
        {
            "manual_id": manual_id,
            "chunk_count": count,
            "image_chunk_count": int(image_chunk_count_by_manual.get(manual_id, 0)),
            "avg_chunk_chars": avg_chunk_chars_by_manual.get(manual_id, 0.0),
        }
        for manual_id, count in chunk_count_by_manual.most_common(8)
    ]
    return {
        "top_chunk_manuals": top_chunk_manuals,
        "manual_chunk_counts": dict(chunk_count_by_manual),
        "parse_mode_counts": dict(parse_mode_counts),
        "attachment_outliers": sorted(
            attachment_outliers,
            key=lambda item: (item["pic_coverage_ratio"], -item["pic_count"]),
        )[:8],
    }


def build_english_summary_segments(chunks: list[KnowledgeChunk]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        if not chunk.manual_id.startswith("汇总英文手册"):
            continue
        metadata = chunk.metadata or {}
        sub_manual_id = str(metadata.get("sub_manual_id") or chunk.manual_id)
        domain_label = str(metadata.get("domain_segment_label") or metadata.get("section_domain_label") or "")
        segment_index = metadata.get("domain_segment_index")
        if not domain_label or not isinstance(segment_index, int) or segment_index < 0:
            continue
        record = grouped.setdefault(
            sub_manual_id,
            {
                "sub_manual_id": sub_manual_id,
                "manual_id": chunk.manual_id,
                "domain_label": domain_label,
                "domain_segment_index": segment_index,
                "chunk_count": 0,
                "image_chunk_count": 0,
                "avg_char_count": 0.0,
                "_char_total": 0,
                "titles": [],
                "section_indexes": [],
            },
        )
        record["chunk_count"] += 1
        record["_char_total"] += chunk.char_count
        if chunk.image_ids:
            record["image_chunk_count"] += 1
        if len(record["titles"]) < 3 and chunk.title not in record["titles"]:
            record["titles"].append(chunk.title)
        record["section_indexes"].append(chunk.section_index)

    segments: list[dict[str, Any]] = []
    for sub_manual_id, record in sorted(grouped.items(), key=lambda item: item[1]["domain_segment_index"]):
        section_indexes = record.pop("section_indexes")
        char_total = record.pop("_char_total")
        record["avg_char_count"] = round(char_total / record["chunk_count"], 2) if record["chunk_count"] else 0.0
        record["section_range"] = [
            min(section_indexes) if section_indexes else 0,
            max(section_indexes) if section_indexes else 0,
        ]
        segments.append(record)
    return segments


def summarize_english_summary_segment_quality(segments: list[dict[str, Any]]) -> dict[str, Any]:
    if not segments:
        return {
            "segment_count": 0,
            "short_segment_ratio": 0.0,
            "singleton_segment_ratio": 0.0,
            "domain_segment_counts": {},
        }
    short_segments = sum(1 for item in segments if int(item.get("chunk_count", 0)) <= 2)
    singleton_segments = sum(1 for item in segments if int(item.get("chunk_count", 0)) == 1)
    domain_segment_counts = Counter(str(item.get("domain_label") or "") for item in segments if item.get("domain_label"))
    return {
        "segment_count": len(segments),
        "short_segment_ratio": round(short_segments / len(segments), 4),
        "singleton_segment_ratio": round(singleton_segments / len(segments), 4),
        "domain_segment_counts": dict(domain_segment_counts.most_common()),
    }


def _median(values: list[int]) -> int:
    if not values:
        return 0
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return values[midpoint]
    return int((values[midpoint - 1] + values[midpoint]) / 2)


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
