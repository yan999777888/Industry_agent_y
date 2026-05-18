"""Pre-compute semantic embeddings for all image descriptions using DashScope text-embedding-v4.

Stores embeddings in a simple NPY file + image_id order JSON for runtime lookup.

Usage:
    DASHSCOPE_API_KEY=sk-xxx python scripts/embed_image_descriptions.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
    from openai import OpenAI
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install numpy openai")
    sys.exit(1)

KB_DIR = PROJECT_ROOT / "data" / "processed" / "kb"
IMAGES_PATH = KB_DIR / "images.jsonl"
OUTPUT_NPY = KB_DIR / "image_desc_embeddings.npy"
OUTPUT_IDS = KB_DIR / "image_desc_ids.json"
BATCH_SIZE = 10  # text-embedding-v4 batch limit is 10

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")


def main():
    if not DASHSCOPE_API_KEY:
        print("ERROR: DASHSCOPE_API_KEY required")
        sys.exit(1)

    # Load descriptions
    image_ids: list[str] = []
    descriptions: list[str] = []
    with open(IMAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            desc = rec.get("description", "").strip()
            if desc:
                image_ids.append(rec["image_id"])
                descriptions.append(desc)

    print(f"Loaded {len(descriptions)} image descriptions")

    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    # Embed in batches
    all_embeddings: list[list[float]] = []
    total_batches = (len(descriptions) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(descriptions), BATCH_SIZE):
        batch = descriptions[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    dimensions=1024,
                    encoding_format="float",
                )
                # Sort by index to ensure correct order
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                embeddings = [d.embedding for d in sorted_data]
                all_embeddings.extend(embeddings)
                print(f"  Batch {batch_num}/{total_batches}: {len(embeddings)} embeddings OK")
                break
            except Exception as e:
                print(f"  Batch {batch_num}/{total_batches} attempt {attempt+1}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  FAILED batch starting at {batch_idx}")
                    return

    # Save
    emb_array = np.array(all_embeddings, dtype=np.float32)
    np.save(str(OUTPUT_NPY), emb_array)
    with open(OUTPUT_IDS, "w", encoding="utf-8") as f:
        json.dump(image_ids, f, ensure_ascii=False)

    print(f"\nSaved {len(image_ids)} embeddings to:")
    print(f"  {OUTPUT_NPY} shape={emb_array.shape}")
    print(f"  {OUTPUT_IDS}")
    print(f"  Sample: {image_ids[0]} -> {emb_array[0][:5]}...")


if __name__ == "__main__":
    main()
