"""Check product name consistency between regular and image chunks."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb" / "index.sqlite"
conn = sqlite3.connect(str(DB_PATH))

regular = conn.execute(
    "SELECT DISTINCT product_name FROM chunks WHERE chunk_id NOT LIKE 'img:%'"
).fetchall()
image = conn.execute(
    "SELECT DISTINCT product_name FROM chunks WHERE chunk_id LIKE 'img:%'"
).fetchall()

regular_names = {r[0] for r in regular}
image_names = {r[0] for r in image}

print("Regular product names:")
for n in sorted(regular_names):
    print(f"  [{n}]")

print(f"\nImage product names:")
for n in sorted(image_names):
    print(f"  [{n}]")

only_img = image_names - regular_names
only_reg = regular_names - image_names
print(f"\nIn image but NOT in regular: {sorted(only_img)}")
print(f"In regular but NOT in image: {sorted(only_reg)}")
print(f"\nTotal: regular={len(regular_names)}, image={len(image_names)}, mismatch={len(only_img)}")

# Check counts per product in image chunks
counts = conn.execute(
    "SELECT product_name, COUNT(*) FROM chunks WHERE chunk_id LIKE 'img:%' GROUP BY product_name ORDER BY COUNT(*) DESC LIMIT 20"
).fetchall()
print("\nImage chunk count by product:")
for name, cnt in counts:
    in_reg = "Y" if name in regular_names else "N"
    print(f"  [{name}]: {cnt} (in regular: {in_reg})")

conn.close()
