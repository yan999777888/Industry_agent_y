#!/usr/bin/env python3
"""Run a fixed regression suite against the local /chat API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed regression suite against /chat.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    passed = 0
    for index, case in enumerate(cases, start=1):
        payload = {
            "question": case["question"],
            "images": case.get("images", []),
            "session_id": case.get("session_id"),
        }
        response = call_chat(args.base_url, payload)
        data = response.get("data", {})
        answer = str(data.get("answer", ""))
        expectations = case.get("expect_contains", [])
        ok = response.get("code") == 0 and all(term in answer for term in expectations)
        if ok:
            passed += 1
        status = "OK" if ok else "FAIL"
        print(f"[{index}/{len(cases)}] {status} case={case.get('id', index)}")
        if not ok:
            print(f"  question={case['question']}")
            print(f"  expect_contains={expectations}")
            print(f"  answer={answer[:240]}")

    print(f"Passed: {passed}/{len(cases)}")


if __name__ == "__main__":
    main()
