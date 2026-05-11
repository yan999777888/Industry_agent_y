#!/usr/bin/env python3
"""Run a fixed regression suite against the local /chat API."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_CASES = Path("tests/fixtures/regression_cases.json")


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("regression cases must be a list")
    return payload


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


def check_case(case: dict, response: dict) -> tuple[bool, list[str]]:
    data = response.get("data", {})
    answer = str(data.get("answer", ""))
    sources = list(data.get("sources", []) or [])
    image_ids = list(data.get("image_ids", []) or [])
    detail_text = json.dumps(response.get("detail", ""), ensure_ascii=False)

    failures: list[str] = []
    expected_http_status = case.get("expect_http_status")
    if expected_http_status is not None:
        if int(response.get("_http_status", 200)) != int(expected_http_status):
            failures.append(
                f"unexpected http status: {response.get('_http_status', 200)} != {expected_http_status}"
            )
    elif response.get("_http_status", 200) != 200:
        failures.append(f"unexpected http status: {response.get('_http_status')}")

    for term in case.get("expect_contains", []):
        if term not in answer:
            failures.append(f"missing answer term: {term}")
    for term in case.get("expect_not_contains", []):
        if term in answer:
            failures.append(f"unexpected answer term: {term}")
    for source in case.get("expect_sources_contains", []):
        if source not in sources:
            failures.append(f"missing source: {source}")
    for term in case.get("expect_error_contains", []):
        if term not in detail_text:
            failures.append(f"missing error detail term: {term}")

    for path, expected in case.get("expect_debug_equals", {}).items():
        actual = _get_nested_value(data.get("retrieval_debug", {}), str(path))
        if actual != expected:
            failures.append(f"debug mismatch at {path}: {actual!r} != {expected!r}")

    for path, expected_terms in case.get("expect_debug_contains", {}).items():
        actual = _get_nested_value(data.get("retrieval_debug", {}), str(path))
        if isinstance(expected_terms, list):
            values = expected_terms
        else:
            values = [expected_terms]
        haystack = json.dumps(actual, ensure_ascii=False) if isinstance(actual, (list, dict)) else str(actual)
        for term in values:
            if str(term) not in haystack:
                failures.append(f"missing debug term at {path}: {term}")

    min_image_ids = case.get("min_image_ids")
    if min_image_ids is not None and len(image_ids) < int(min_image_ids):
        failures.append(f"image_ids too few: {len(image_ids)} < {min_image_ids}")

    max_image_ids = case.get("max_image_ids")
    if max_image_ids is not None and len(image_ids) > int(max_image_ids):
        failures.append(f"image_ids too many: {len(image_ids)} > {max_image_ids}")

    if expected_http_status is None and response.get("code") != 0:
        failures.append(f"unexpected response code: {response.get('code')}")

    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed regression suite against /chat.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    passed = 0
    for index, case in enumerate(cases, start=1):
        payload = build_payload(case)
        response = call_chat(args.base_url, payload)
        data = response.get("data", {})
        answer = str(data.get("answer", ""))
        ok, failures = check_case(case, response)
        if ok:
            passed += 1
        status = "OK" if ok else "FAIL"
        print(f"[{index}/{len(cases)}] {status} case={case.get('id', index)}")
        if not ok:
            print(f"  question={case['question']}")
            print(f"  failures={failures}")
            print(f"  answer={answer[:240]}")

    print(f"Passed: {passed}/{len(cases)}")


if __name__ == "__main__":
    main()
