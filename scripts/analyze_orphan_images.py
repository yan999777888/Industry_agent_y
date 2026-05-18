"""Analyze orphan images to see if they can be mapped to manuals."""
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb"

with open(KB_DIR / "images.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

orphans = [r for r in records if not r.get("chunk_ids") and not r.get("referenced_by")]

# Group by prefix pattern
prefix_groups = defaultdict(list)
for r in orphans:
    # Extract prefix (e.g., "Boat" from "Boat_01", "Manual04" from "Manual04_0")
    img_id = r["image_id"]
    # Try various patterns
    parts = img_id.rsplit("_", 1)
    prefix = parts[0] if parts else img_id
    prefix_groups[prefix].append(r)

print(f"Total orphans: {len(orphans)}")
print(f"Unique prefixes: {len(prefix_groups)}")

# Show top prefix groups
sorted_groups = sorted(prefix_groups.items(), key=lambda x: -len(x[1]))
print(f"\nTop orphan groups:")
for prefix, items in sorted_groups[:20]:
    print(f"  {prefix}: {len(items)} images (e.g., {items[0]['image_id']} - {items[-1]['image_id']})")

# Check against known manuals
conn = sqlite3.connect(str(KB_DIR / "index.sqlite"))
manual_ids = set(r[0] for r in conn.execute("SELECT manual_id FROM manuals").fetchall())
product_names = set(r[0] for r in conn.execute("SELECT DISTINCT product_name FROM chunks").fetchall())
conn.close()

print(f"\nKnown manual_ids: {sorted(manual_ids)}")
print(f"Known product_names: {sorted(product_names)}")

# Check if any orphan prefixes match manual_ids or product names
print(f"\nPrefix matching analysis:")
for prefix in sorted(prefix_groups.keys()):
    count = len(prefix_groups[prefix])
    if count < 5:
        continue
    matches_manual = prefix in manual_ids or any(m.startswith(prefix) for m in manual_ids)
    matches_product = prefix in product_names or any(p.startswith(prefix) for p in product_names)
    if not matches_manual and not matches_product:
        # Try manual number pattern
        manual_match = any(prefix.startswith(m.split("手册")[0]) if "手册" in m else False for m in manual_ids)
        if manual_match:
            matches_manual = True
    marker = ""
    if matches_manual:
        marker = " << MATCHES MANUAL"
    if matches_product:
        marker = " << MATCHES PRODUCT"
    if marker:
        print(f"  {prefix}: {count} images{marker}")
