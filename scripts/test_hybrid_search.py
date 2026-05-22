"""Test hybrid retriever with Milvus backend."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.rag.factory import create_retriever

retriever = create_retriever("hybrid")
results = retriever.search("什么是气动调节阀", limit=5)
print(f"Hybrid results: {len(results)}")
for r in results:
    cid = r.get("chunk_id", "?")
    rrf = r.get("_rrf_score", "N/A")
    vec = r.get("_vector_score", "N/A")
    score = r.get("_score", "N/A")
    title = (r.get("title") or "")[:60]
    print(f"  chunk_id={cid} _rrf={rrf} _vec={vec} _score={score} title={title}")
