"""Normalize product names in image chunks to match regular chunks."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb" / "index.sqlite"
conn = sqlite3.connect(str(DB_PATH))

# Show before
before = conn.execute(
    "SELECT product_name, COUNT(*) FROM chunks WHERE chunk_id LIKE 'img:%' "
    "GROUP BY product_name ORDER BY COUNT(*) DESC LIMIT 10"
).fetchall()
print("Before:")
for name, cnt in before:
    print(f"  [{name}]: {cnt}")

# Normalize: remove '手册' suffix
conn.execute(
    "UPDATE chunks SET product_name = REPLACE(product_name, '手册', '') "
    "WHERE chunk_id LIKE 'img:%' AND product_name LIKE '%手册'"
)
conn.commit()

# Show after
after = conn.execute(
    "SELECT product_name, COUNT(*) FROM chunks WHERE chunk_id LIKE 'img:%' "
    "GROUP BY product_name ORDER BY COUNT(*) DESC LIMIT 10"
).fetchall()
print("\nAfter:")
for name, cnt in after:
    print(f"  [{name}]: {cnt}")

# Verify all match
regular = set(r[0] for r in conn.execute(
    "SELECT DISTINCT product_name FROM chunks WHERE chunk_id NOT LIKE 'img:%'"
).fetchall())
image = set(r[0] for r in conn.execute(
    "SELECT DISTINCT product_name FROM chunks WHERE chunk_id LIKE 'img:%'"
).fetchall())
only_img = image - regular
only_reg = regular - image
print(f"\nMismatch: img-only={sorted(only_img)}, reg-only={sorted(only_reg)}")

conn.close()
print("Done.")
