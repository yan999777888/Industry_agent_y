"""Quick test for Milvus vector search."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.rag.vector_store import MilvusVectorSearcher

searcher = MilvusVectorSearcher()
results = searcher.search("什么是气动调节阀", limit=5)
print(f"Results: {len(results)}")
for r in results:
    cid = r.get("chunk_id", "?")
    score = r.get("_vector_score", 0)
    title = (r.get("title") or "")[:60]
    print(f"  chunk_id={cid} score={score} title={title}")
