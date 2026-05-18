"""Generate submission for English questions only (QIDs 241-380)."""
import csv
import json
import sys
import time
from pathlib import Path

import requests

API_URL = "http://localhost:8000/chat"
INPUT_CSV = "submission/question_public.csv"
OUTPUT_CSV = "submission/submission_en_only.csv"

HEADERS = {"Content-Type": "application/json"}


def main():
    # Read questions
    questions = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["id"])
            if 241 <= qid <= 436:
                questions.append((row["id"], row["question"]))

    print(f"English questions to process: {len(questions)}")

    results = []
    for idx, (qid, question) in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] QID={qid}...", end=" ", flush=True)
        try:
            resp = requests.post(
                API_URL,
                json={"question": question, "session_id": f"en_{qid}"},
                headers=HEADERS,
                timeout=120,
            )
            data = resp.json()
            answer = data.get("answer", "") or data.get("ret", "")
            # Get image_ids if present
            img_ids = data.get("image_ids", [])
            if img_ids:
                answer += ",[" + ",".join(json.dumps(i) for i in img_ids) + "]"
            results.append((qid, answer))
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((qid, ""))
        time.sleep(0.5)  # rate limit

    # Write output CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ret"])
        for qid, answer in results:
            writer.writerow([qid, answer])

    print(f"\nDone! Saved to {OUTPUT_CSV}")
    print(f"Successful: {sum(1 for _, a in results if a)}/{len(results)}")


if __name__ == "__main__":
    main()
