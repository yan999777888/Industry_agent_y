"""Build Milvus vector index from chunks.jsonl.

Reads chunks.jsonl, embeds all chunks with the configured embedding model
(BAAI/bge-m3), and inserts vectors into a Milvus Lite collection.

Usage:
    python scripts/build_milvus_index.py

The collection is created at the path specified by MILVUS_URI (default: ./milvus.db).
Set VECTOR_BACKEND=milvus (or INDUSTRY_AGENT_VECTOR_BACKEND=milvus) after building
to tell the retriever to use Milvus instead of SQLite chunk_vectors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.config import settings
from industry_agent.kb.models import KnowledgeChunk
from industry_agent.rag.vector_store import build_milvus_vector_index, MilvusClient

CHUNKS_PATH = settings.processed_dir / "chunks.jsonl"


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(KnowledgeChunk(**json.loads(line)))
    return chunks


def main() -> int:
    # 1. Load chunks
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  Loaded {len(chunks)} chunks")

    # 2. Check Milvus installation
    try:
        from pymilvus import MilvusClient
    except ImportError:
        print("ERROR: pymilvus not installed. Run: pip install pymilvus[milvus-lite]")
        return 1

    # 3. Build index
    print(f"\nBuilding Milvus vector index...")
    print(f"  URI: {settings.milvus_uri}")
    print(f"  Collection: {settings.milvus_collection}")
    print(f"  Embedding model: {settings.embedding_model}")
    print(f"  Vector backend: {settings.vector_backend}")
    print(f"  DashScope enabled: {settings.dashscope_enabled}")
    print()

    start = time.time()
    result = build_milvus_vector_index(
        chunks,
        milvus_uri=settings.milvus_uri,
        milvus_token=settings.milvus_token,
        collection_name=settings.milvus_collection,
    )
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Status: {result.get('status', '?')}")
    print(f"  Chunks indexed: {result.get('chunk_count', 0)}")
    print(f"  Collection: {result.get('collection', '?')}")
    print(f"  Dimensions: {result.get('dimensions', 0)}")

    if result.get("status") == "built":
        print(f"\nNext: set VECTOR_BACKEND=milvus (or INDUSTRY_AGENT_VECTOR_BACKEND=milvus)")
        print("  then restart the API service to use Milvus for vector search.")
        return 0
    else:
        print(f"\nBuild failed or was skipped.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
