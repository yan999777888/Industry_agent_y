"""Analyze generated submission answers for quality risks.

This script does not use labels or public-question-specific rules.  It reports
format and behavior risks that usually correlate with low automatic scores:
fallback answers, very long answers, missing picture markers, likely question
echo, and incomplete multi-question handling.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


FALLBACK_RE = re.compile(r"无法回答|无法准确|补充.*产品|根据现有资料")
PIC_RE = re.compile(r"<PIC>")
IMAGE_SUFFIX_RE = re.compile(r'";\[')
QUESTION_ECHO_RE = re.compile(r"请问|怎么|如何|是否|能不能|是什么|有哪些")
QUESTION_SPLIT_RE = re.compile(r'"\s*,\s*"|\n+|[？?]\s*["”]?\s*[,，]?\s*["“]?')


def analyze_submission(
    csv_path: Path,
    *,
    debug_path: Path | None = None,
    sample_limit: int = 8,
) -> dict[str, Any]:
    with csv_path.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"empty submission file: {csv_path}")

    debug_by_id = _load_debug_by_id(debug_path) if debug_path else {}
    lengths = [len(row.get("ret", "")) for row in rows]
    metrics = Counter()
    examples: dict[str, list[dict[str, str]]] = {
        "fallback": [],
        "too_long": [],
        "question_echo": [],
        "multi_question_maybe_incomplete": [],
    }

    for row in rows:
        row_id = str(row.get("id", ""))
        answer = row.get("ret", "")
        debug = debug_by_id.get(row_id, {})
        question = str(debug.get("question", ""))
        if FALLBACK_RE.search(answer):
            metrics["fallback"] += 1
            _append_example(examples["fallback"], row_id, question, answer, sample_limit)
        if PIC_RE.search(answer):
            metrics["pic_marker"] += 1
        if IMAGE_SUFFIX_RE.search(answer):
            metrics["image_suffix"] += 1
        if len(answer) > 900:
            metrics["too_long"] += 1
            _append_example(examples["too_long"], row_id, question, answer, sample_limit)
        if QUESTION_ECHO_RE.search(answer):
            metrics["question_echo"] += 1
            _append_example(examples["question_echo"], row_id, question, answer, sample_limit)
        if _looks_like_multi_question(question) and _looks_incomplete_for_multi_question(answer):
            metrics["multi_question_maybe_incomplete"] += 1
            _append_example(examples["multi_question_maybe_incomplete"], row_id, question, answer, sample_limit)

    return {
        "file": str(csv_path),
        "rows": len(rows),
        "length": {
            "avg": round(sum(lengths) / len(lengths), 1),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "metrics": dict(metrics),
        "risk_summary": _risk_summary(metrics, len(rows)),
        "examples": examples,
    }


def _load_debug_by_id(debug_path: Path | None) -> dict[str, dict[str, Any]]:
    if debug_path is None or not debug_path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with debug_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            result[str(item.get("id", ""))] = item
    return result


def _looks_like_multi_question(question: str) -> bool:
    if not question:
        return False
    parts = [part.strip(' "\'“”') for part in QUESTION_SPLIT_RE.split(question) if part.strip(' "\'“”')]
    return len(parts) >= 2


def _looks_incomplete_for_multi_question(answer: str) -> bool:
    if "问题1" in answer and "问题2" in answer:
        return False
    sentence_count = len([part for part in re.split(r"[。！？!?]", answer) if part.strip()])
    return sentence_count <= 2 or FALLBACK_RE.search(answer) is not None


def _append_example(
    examples: list[dict[str, str]],
    row_id: str,
    question: str,
    answer: str,
    sample_limit: int,
) -> None:
    if len(examples) >= sample_limit:
        return
    examples.append(
        {
            "id": row_id,
            "question": question[:160],
            "answer": answer[:240],
        }
    )


def _risk_summary(metrics: Counter[str], total: int) -> list[str]:
    summary: list[str] = []
    if metrics["too_long"] / total >= 0.15:
        summary.append("答案过长比例偏高，可能导致自动评分认为回答不聚焦。")
    if metrics["fallback"] / total >= 0.05:
        summary.append("拒答/fallback 比例偏高，说明检索证据或路由仍有漏召回。")
    if metrics["question_echo"] / total >= 0.08:
        summary.append("疑似问题回显比例偏高，提交清洗或回答模板还需压缩。")
    if metrics["multi_question_maybe_incomplete"] > 0:
        summary.append("存在多问题疑似漏答，需检查复杂问题拆解和提交清洗。")
    if not summary:
        summary.append("未发现明显格式风险，下一步应重点抽查语义正确性和图片相关性。")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated submission quality risks.")
    parser.add_argument("--submission", type=Path, default=Path("submission/submission_generated.csv"))
    parser.add_argument("--debug", type=Path, default=Path("submission/submission_generated_debug.jsonl"))
    parser.add_argument("--sample-limit", type=int, default=8)
    args = parser.parse_args()

    report = analyze_submission(args.submission, debug_path=args.debug, sample_limit=args.sample_limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
