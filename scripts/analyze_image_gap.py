"""Analyze why only 805 of 2608 images have descriptions."""
import json
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb"

with open(KB_DIR / "images.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

total = len(records)
with_chunks = [r for r in records if r.get("chunk_ids")]
without_chunks = [r for r in records if not r.get("chunk_ids")]

print(f"Total images: {total}")
print(f"With chunk_ids: {len(with_chunks)}")
print(f"Without chunk_ids: {len(without_chunks)}")

# Breakdown of images without chunks
no_chunk_no_ref = [r for r in without_chunks if not r.get("referenced_by")]
no_chunk_with_ref = [r for r in without_chunks if r.get("referenced_by")]
print(f"\nWithout chunks breakdown:")
print(f"  No referenced_by either (orphan): {len(no_chunk_no_ref)}")
print(f"  Have referenced_by but no chunk_ids: {len(no_chunk_with_ref)}")

# Show some examples
print("\nOrphan images (no chunk, no reference):")
for r in no_chunk_no_ref[:10]:
    print(f"  {r['image_id']} exists={r.get('exists')}")

print("\nHave reference but no chunk_ids:")
for r in no_chunk_with_ref[:10]:
    print(f"  {r['image_id']} ref={r['referenced_by']} exists={r.get('exists')}")

# Show images WITH chunks that we would generate descriptions for
print(f"\nImages that WOULD get descriptions: {len(with_chunks)}")
print("Sample:")
for r in with_chunks[:5]:
    print(f"  {r['image_id']} chunks={r['chunk_ids']}")

# How many no-chunk images have referenced_by naming an actual manual?
import sqlite3
conn = sqlite3.connect(str(KB_DIR / "index.sqlite"))
manual_ids = set(r[0] for r in conn.execute("SELECT DISTINCT manual_id FROM manuals").fetchall())
conn.close()

no_chunk_ref_match = [r for r in no_chunk_with_ref
                      if any(ref in manual_ids for ref in r.get("referenced_by", []))]
print(f"\nNo-chunk images with manual reference matching known manuals: {len(no_chunk_ref_match)}")
print(f"These could be recoverable if we scan manuals for PIC references.")
