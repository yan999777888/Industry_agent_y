"""Build image-description chunks by extracting context around each image.

For each image referenced in the knowledge base:
1. Start with the owning chunk's title + text
2. If the chunk text is too short, pull in neighboring chunks from the same manual
3. Prepend product name for disambiguation

Output: data/processed/kb/image_chunks.jsonl (one record per unique image)
"""

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "kb"
DB_PATH = KB_DIR / "index.sqlite"
IMAGES_PATH = KB_DIR / "images.jsonl"


def clean_text(text: str) -> str:
    text = re.sub(r"\[\[PIC[^\]]*\]\]", " ", text)
    text = re.sub(r"#+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_image_index() -> dict:
    """image_id -> list of chunk_ids (from images.jsonl)"""
    mapping: dict[str, list[str]] = {}
    with open(IMAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cids = rec.get("chunk_ids", [])
            if cids:
                mapping[rec["image_id"]] = cids
    return mapping


def load_chunks_dict(db) -> dict:
    """chunk_id -> {title, text, product_name, manual_name}"""
    rows = db.execute(
        "SELECT chunk_id, product_name, title, text FROM chunks"
    ).fetchall()
    chunks: dict[str, dict] = {}
    for r in rows:
        chunks[r["chunk_id"]] = {
            "title": r["title"] or "",
            "text": r["text"] or "",
            "product": r["product_name"] or "",
        }
    return chunks


def get_neighbor_chunks(chunk_id: str, chunks_db, n: int = 2) -> list:
    """Get n previous and n next chunks from the same product."""
    # Find the product and position of this chunk
    row = chunks_db.execute(
        "SELECT product_name, rowid FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    if not row:
        return []
    product, rowid = row["product_name"], row["rowid"]

    neighbors = chunks_db.execute(
        """SELECT title, text FROM chunks
           WHERE product_name = ? AND rowid BETWEEN ? AND ?
           ORDER BY rowid ASC""",
        (product, max(1, rowid - n), rowid + n),
    ).fetchall()
    return [{"title": r["title"] or "", "text": r["text"] or ""} for r in neighbors if r["title"] or r["text"]]


def build_description(
    img_id: str,
    chunk_ids: list[str],
    chunks: dict[str, dict],
    db,
) -> str:
    """Build a rich description for an image from its owning chunks."""
    descriptions: list[str] = []

    for cid in chunk_ids[:3]:  # max 3 chunks per image
        ch = chunks.get(cid, {})
        title = clean_text(ch.get("title", ""))
        text = clean_text(ch.get("text", ""))
        product = ch.get("product", "")
        combined = f"{title}. {text}".strip(" .")

        if len(combined) >= 80:
            descriptions.append(combined)
        else:
            # Short chunk: pull in neighbors
            neighbors = get_neighbor_chunks(cid, db, n=2)
            neighbor_texts = []
            for nb in neighbors:
                nb_combined = f"{clean_text(nb['title'])}. {clean_text(nb['text'])}".strip(" .")
                if nb_combined and nb_combined != combined:
                    neighbor_texts.append(nb_combined)
            expanded = combined
            if neighbor_texts:
                expanded += " | " + " ".join(neighbor_texts[:3])
            descriptions.append(expanded)

    # Join all chunk descriptions
    full = " ".join(descriptions)

    # Prepend product for disambiguation
    if chunk_ids:
        product = chunks.get(chunk_ids[0], {}).get("product", "")
    else:
        product = ""
    if product and product not in full[:100]:
        full = f"[{product}] {full}"

    return full[:500]  # keep reasonable for embedding


def main():
    out_path = KB_DIR / "image_chunks.jsonl"
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])

    print("Loading image index...")
    img_to_chunks = load_image_index()
    print(f"Images with chunk refs: {len(img_to_chunks)}")

    print("Loading chunks from DB...")
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    chunks = load_chunks_dict(db)

    records = []
    for img_id, chunk_ids in sorted(img_to_chunks.items()):
        desc = build_description(img_id, chunk_ids, chunks, db)
        records.append({
            "image_id": img_id,
            "description": desc,
            "chunk_count": len(chunk_ids),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} image descriptions to {out_path}")

    # Show samples
    print("\n--- Short descriptions (were CHUNK text short, now expanded) ---")
    short_before = [r for r in records if r["chunk_count"] == 1 and len(r["description"]) < 150]
    for r in short_before[:8]:
        print(f"{r['image_id']}: {r['description'][:150]}")

    print("\n--- Long descriptions (good chunk text) ---")
    long_ones = [r for r in records if len(r["description"]) >= 200]
    for r in long_ones[:5]:
        print(f"{r['image_id']}: {r['description'][:200]}")

    db.close()


if __name__ == "__main__":
    main()
