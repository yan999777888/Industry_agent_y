"""Parse submission debug JSONL and show image retrieval stats."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/submission_test_debug.jsonl"

with open(path) as f:
    for line in f:
        r = json.loads(line)
        qid = r.get("id", "?")
        question = str(r.get("question", ""))[:80]
        resp = r.get("response", {})
        data = resp.get("data", {})

        answer = data.get("answer", "")[:200]
        image_ids = data.get("image_ids", [])
        images = data.get("images", [])
        sources = data.get("sources", [])
        refs = data.get("references", [])
        elapsed = r.get("elapsed_sec", 0)

        titles = [ref.get("title", "")[:50] for ref in refs[:3]]

        print(f"--- ID {qid} ({elapsed:.0f}s) ---")
        print(f"Q: {question}")
        print(f"A: {answer}...")
        print(f"Images ({len(images)}): {image_ids[:5]}")
        for img in images[:3]:
            print(f"  - {img.get('image_id', '?')} ({img.get('file_name', '?')})")
        print(f"Sources: {sources}")
        print(f"Top chunks: {titles}")
        print()
