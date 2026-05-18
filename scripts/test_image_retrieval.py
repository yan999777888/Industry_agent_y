"""Test that image chunks participate in vector retrieval."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import sqlite3
from industry_agent.rag.vector_store import SQLiteVectorSearcher, VectorSearchConfig

config = VectorSearchConfig(enabled=True, embedding_model="BAAI/bge-m3", dimensions=1024)
searcher = SQLiteVectorSearcher(config=config)

queries = [
    "如何清洁洗碗机过滤器",
    "How to replace the spark plug",
    "空调遥控器怎么设置定时",
    "how to clean the oven",
    "冰箱温度设置",
]

for q in queries:
    results = searcher.search(q, limit=10)
    img_chunks = [r for r in results if str(r.get("chunk_id", "")).startswith("img:")]
    print(f"\nQuery: {q}")
    print(f"  Total results: {len(results)}, Image chunks: {len(img_chunks)}")
    for r in img_chunks[:3]:
        img_id = r.get("image_ids", "[]")
        score = r.get("_vector_score", 0)
        text = str(r.get("text", ""))[:100]
        print(f"    {r['chunk_id']} score={score:.4f} img={img_id}")
        print(f"    text: {text}...")
