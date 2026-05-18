"""Quick verification of image chunks in the database."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb" / "index.sqlite"

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
img = conn.execute("SELECT COUNT(*) FROM chunks WHERE chunk_id LIKE 'img:%'").fetchone()[0]
total_vec = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
img_vec = conn.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id LIKE 'img:%'").fetchone()[0]

print(f"Total chunks: {total} (including {img} image chunks)")
print(f"Total vectors: {total_vec} (including {img_vec} image vectors)")

join_test = conn.execute("""
    SELECT c.chunk_id, c.product_name, substr(c.text, 1, 80) as preview,
           length(cv.vector) as vec_bytes
    FROM chunk_vectors cv
    JOIN chunks c ON c.chunk_id = cv.chunk_id
    WHERE c.chunk_id LIKE 'img:%'
    LIMIT 3
""").fetchall()
print("\nJOIN test (vectors + chunks):")
for r in join_test:
    print(f"  {r[0]:30s} [{r[1]:15s}] vec={r[3]}B  {r[2]}...")

print("\nDone.")
conn.close()
