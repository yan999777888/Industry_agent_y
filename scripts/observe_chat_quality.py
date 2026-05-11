#!/usr/bin/env python3
"""Run a tagged end-to-end observation set against the local /chat API."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_CASES = Path("tests/fixtures/quality_observation_cases.json")
DEFAULT_OUTPUT = Path("data/processed/quality_observation_report.json")


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("quality observation cases must be a list")
    return payload


def build_payload(case: dict) -> dict:
    images = list(case.get("images", []))
    for path_value in case.get("image_paths", []):
        image_path = Path(path_value)
        images.append(base64.b64encode(image_path.read_bytes()).decode("utf-8"))
    return {
        "question": case["question"],
        "images": images,
        "session_id": case.get("session_id"),
    }


def _get_nested_value(payload: dict, path: str):
    current = payload
    for raw_part in path.split("."):
        part = raw_part.strip()
        if not part:
            continue
        if isinstance(current, list):
            if not part.isdigit():
                return None
            index = int(part)
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def call_chat(base_url: str, payload: dict) -> dict:
    request = Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
            payload["_http_status"] = response.status
            return payload
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        try:
            payload = json.loads(detail)
        except json.JSONDecodeError:
            payload = {"detail": detail}
        payload["_http_status"] = exc.code
        return payload


def evaluate_case(case: dict, response: dict) -> dict:
    data = response.get("data", {})
    answer = str(data.get("answer", ""))
    sources = list(data.get("sources", []) or [])
    image_ids = list(data.get("image_ids", []) or [])
    confidence = float(data.get("confidence", 0.0) or 0.0)
    retrieval_debug = data.get("retrieval_debug", {}) or {}
    detail_text = json.dumps(response.get("detail", ""), ensure_ascii=False)

    issues: list[str] = []
    details: list[str] = []
    http_status = int(response.get("_http_status", 200))
    if http_status != int(case.get("expect_http_status", 200)):
        issues.append("http_status")
        details.append(f"http status {http_status} != {case.get('expect_http_status', 200)}")

    if case.get("expect_http_status") is None and response.get("code") != 0:
        issues.append("response_code")
        details.append(f"unexpected response code: {response.get('code')}")

    for term in case.get("expect_contains", []):
        if term not in answer:
            issues.append("answer_alignment")
            details.append(f"missing answer term: {term}")

    for term in case.get("expect_not_contains", []):
        if term in answer:
            issues.append("answer_noise")
            details.append(f"unexpected answer term: {term}")

    for source in case.get("expect_sources_contains", []):
        if source not in sources:
            issues.append("source_routing")
            details.append(f"missing source: {source}")

    for term in case.get("expect_error_contains", []):
        if term not in detail_text:
            issues.append("error_detail")
            details.append(f"missing error detail term: {term}")

    for path, expected in case.get("expect_debug_equals", {}).items():
        actual = _get_nested_value(retrieval_debug, str(path))
        if actual != expected:
            issues.append("debug_alignment")
            details.append(f"debug mismatch at {path}: {actual!r} != {expected!r}")

    for path, expected_terms in case.get("expect_debug_contains", {}).items():
        actual = _get_nested_value(retrieval_debug, str(path))
        values = expected_terms if isinstance(expected_terms, list) else [expected_terms]
        haystack = json.dumps(actual, ensure_ascii=False) if isinstance(actual, (list, dict)) else str(actual)
        for term in values:
            if str(term) not in haystack:
                issues.append("debug_alignment")
                details.append(f"missing debug term at {path}: {term}")

    min_image_ids = case.get("min_image_ids")
    if min_image_ids is not None and len(image_ids) < int(min_image_ids):
        issues.append("image_binding")
        details.append(f"image_ids too few: {len(image_ids)} < {min_image_ids}")

    max_image_ids = case.get("max_image_ids")
    if max_image_ids is not None and len(image_ids) > int(max_image_ids):
        issues.append("image_binding")
        details.append(f"image_ids too many: {len(image_ids)} > {max_image_ids}")

    min_confidence = case.get("min_confidence")
    if min_confidence is not None and confidence < float(min_confidence):
        issues.append("low_confidence")
        details.append(f"confidence too low: {confidence:.2f} < {float(min_confidence):.2f}")

    max_confidence = case.get("max_confidence")
    if max_confidence is not None and confidence > float(max_confidence):
        issues.append("confidence_range")
        details.append(f"confidence too high: {confidence:.2f} > {float(max_confidence):.2f}")

    unique_issues: list[str] = []
    for issue in issues:
        if issue not in unique_issues:
            unique_issues.append(issue)

    return {
        "id": case.get("id", ""),
        "category": case.get("category", "uncategorized"),
        "question": case.get("question", ""),
        "ok": not unique_issues,
        "issues": unique_issues,
        "details": details,
        "confidence": round(confidence, 4),
        "sources": sources,
        "image_count": len(image_ids),
    }


def summarize_records(records: list[dict]) -> dict:
    total = len(records)
    passed = sum(1 for record in records if record.get("ok"))
    categories: dict[str, dict[str, int]] = {}
    issues: dict[str, int] = {}

    for record in records:
        category = str(record.get("category", "uncategorized"))
        bucket = categories.setdefault(category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if record.get("ok"):
            bucket["passed"] += 1

        for issue in record.get("issues", []):
            issues[str(issue)] = issues.get(str(issue), 0) + 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "categories": categories,
        "issue_buckets": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tagged quality observation cases against /chat.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    records: list[dict] = []
    for index, case in enumerate(cases, start=1):
        payload = build_payload(case)
        response = call_chat(args.base_url, payload)
        record = evaluate_case(case, response)
        records.append(record)
        status = "OK" if record["ok"] else "OBSERVE"
        print(
            f"[{index}/{len(cases)}] {status} category={record['category']} "
            f"id={record['id']} confidence={record['confidence']}"
        )
        if record["issues"]:
            print(f"  issues={record['issues']}")
            print(f"  details={record['details'][:3]}")

    summary = summarize_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved observation report to {args.output}")
    print(f"Passed: {summary['passed']}/{summary['total']}")
    if summary["issue_buckets"]:
        print(f"Issue buckets: {summary['issue_buckets']}")


if __name__ == "__main__":
    main()
