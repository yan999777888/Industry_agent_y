#!/usr/bin/env python3
"""Run a small API-based evaluation set for the /chat endpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_QUESTIONS = [
    "我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？",
    "我想更换健身追踪器的表带，有其他尺寸可选吗？",
    "洗碗机安装有什么要求？",
    "VR头显使用时有哪些安全注意事项？",
    "可编程温控器的默认密码是多少？",
    "请问这台设备能不能直接连接火星基地网络？",
]


def post_chat(base_url: str, question: str, timeout: int) -> dict:
    payload = {
        "question": question,
        "images": [],
        "session_id": None,
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_questions(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_QUESTIONS
    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        questions.append(line)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate /chat with a small question set.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=Path, default=None, help="Optional newline-separated question file.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/eval_chat_results.jsonl"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    with args.output.open("w", encoding="utf-8") as file:
        for index, question in enumerate(questions, start=1):
            started = time.time()
            record = {
                "id": index,
                "question": question,
                "ok": False,
                "elapsed_sec": 0.0,
            }
            try:
                result = post_chat(args.base_url, question, args.timeout)
                data = result.get("data", {})
                record.update(
                    {
                        "ok": result.get("code") == 0,
                        "answer": data.get("answer", ""),
                        "confidence": data.get("confidence", 0.0),
                        "image_ids": data.get("image_ids", []),
                        "sources": data.get("sources", []),
                        "references": data.get("references", []),
                    }
                )
                ok_count += 1 if record["ok"] else 0
                print(f"[{index}/{len(questions)}] OK confidence={record.get('confidence')} question={question}")
            except (HTTPError, URLError, TimeoutError) as exc:
                record["error"] = str(exc)
                print(f"[{index}/{len(questions)}] ERR question={question}: {exc}")
            finally:
                record["elapsed_sec"] = round(time.time() - started, 3)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved results to {args.output}")
    print(f"Success: {ok_count}/{len(questions)}")


if __name__ == "__main__":
    main()
