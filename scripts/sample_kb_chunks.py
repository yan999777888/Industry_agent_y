#!/usr/bin/env python3
"""Sample built knowledge chunks for manual quality review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.config import settings


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def select_chunk_samples(
    records: list[dict[str, Any]],
    *,
    chunk_type: str = "",
    manual_id: str = "",
    domain_label: str = "",
    has_image: bool | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    normalized_chunk_type = chunk_type.strip().lower()
    normalized_manual_id = manual_id.strip()
    normalized_domain_label = domain_label.strip().lower()

    for record in records:
        metadata = record.get("metadata") or {}
        record_chunk_type = str(metadata.get("chunk_type") or metadata.get("semantic_type") or "").lower()
        if normalized_chunk_type and record_chunk_type != normalized_chunk_type:
            continue
        if normalized_manual_id and str(record.get("manual_id", "")) != normalized_manual_id:
            continue
        if normalized_domain_label and str(metadata.get("domain_label") or "").lower() != normalized_domain_label:
            continue
        if has_image is not None and bool(record.get("image_ids")) != has_image:
            continue
        filtered.append(
            {
                "chunk_id": record.get("chunk_id", ""),
                "manual_id": record.get("manual_id", ""),
                "product_name": record.get("product_name", ""),
                "title": record.get("title", ""),
                "char_count": record.get("char_count", 0),
                "image_ids": record.get("image_ids", []),
                "metadata": metadata,
                "text_preview": str(record.get("text", ""))[:240],
            }
        )
        if len(filtered) >= limit:
            break
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample chunk records for manual review.")
    parser.add_argument("--chunks", type=Path, default=settings.processed_dir / "chunks.jsonl")
    parser.add_argument("--chunk-type", default="", help="Filter by chunk_type / semantic_type")
    parser.add_argument("--manual-id", default="", help="Filter by manual_id")
    parser.add_argument("--domain-label", default="", help="Filter by metadata.domain_label")
    parser.add_argument("--has-image", choices=("yes", "no"), default="", help="Filter chunks with or without images")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    records = load_chunk_records(args.chunks)
    has_image = None
    if args.has_image == "yes":
        has_image = True
    elif args.has_image == "no":
        has_image = False

    samples = select_chunk_samples(
        records,
        chunk_type=args.chunk_type,
        manual_id=args.manual_id,
        domain_label=args.domain_label,
        has_image=has_image,
        limit=args.limit,
    )
    print(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
