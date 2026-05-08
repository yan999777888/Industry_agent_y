"""Quick smoke test for /chat endpoint."""

import requests
import json

URL = "http://localhost:8000"

# ── Test cases ───────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "电钻怎么使用？",
    "可编程温控器的默认密码是多少？",
    "VR头显使用时有哪些安全注意事项？",
]


def test_health():
    resp = requests.get(f"{URL}/health")
    resp.raise_for_status()
    print(f"[Health] {resp.json()}")


def test_chat(question: str, session_id: str | None = None):
    payload = {
        "question": question,
        "session_id": session_id,
    }
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    resp = requests.post(f"{URL}/chat", json=payload)
    resp.raise_for_status()
    result = resp.json()

    data = result.get("data", {})
    print(f"\nAnswer:\n{data.get('answer', '')}\n")
    print(f"Session:    {data.get('session_id')}")
    print(f"Image IDs:  {data.get('image_ids', [])}")
    print(f"Sources:    {data.get('sources', [])}")

    refs = data.get("references", [])
    if refs:
        print("References:")
        for ref in refs:
            print(f"  - [{ref.get('title', '')}] {ref.get('text_snippet', '')[:60]}...")

    return data.get("session_id")


def main():
    print("=== Health Check ===")
    test_health()

    # Single-turn tests
    for q in TEST_QUESTIONS:
        test_chat(q)

    # Multi-turn test
    print(f"\n{'#'*60}")
    print("Multi-turn test")
    print(f"{'#'*60}")
    sid = test_chat("电钻的电池怎么充电？")
    test_chat("充电时有什么注意事项？", session_id=sid)


if __name__ == "__main__":
    main()
