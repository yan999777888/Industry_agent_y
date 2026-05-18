"""Verify image_ids in image chunks are correct (not img: prefixed)."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb" / "index.sqlite"
conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT chunk_id, image_ids, product_name FROM chunks "
    "WHERE chunk_id LIKE 'img:%' LIMIT 8"
).fetchall()

print("Image chunks sample:")
for r in rows:
    chunk_id = r[0]
    image_ids = r[1]
    product = r[2]
    print(f"  chunk_id={chunk_id} | product={product} | image_ids={image_ids}")

# Verify no image_ids contain "img:" prefix
bad = conn.execute(
    "SELECT chunk_id, image_ids FROM chunks "
    "WHERE chunk_id LIKE 'img:%' AND image_ids LIKE '%img:%'"
).fetchall()
print(f"\nImage chunks with img: prefix in image_ids: {len(bad)}")
if bad:
    for r in bad:
        print(f"  BAD: {r[0]} -> {r[1]}")

conn.close()
