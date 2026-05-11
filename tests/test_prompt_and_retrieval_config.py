from __future__ import annotations

import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.agent.prompts import (
    CUSTOMER_SERVICE_RULES,
    MANUAL_QA_RULES,
    build_customer_service_system_prompt,
    build_manual_qa_system_prompt,
)
from industry_agent.kb.models import KnowledgeChunk
from industry_agent.rag.embedding import EmbeddingManager
from industry_agent.rag.factory import VectorOnlyRetriever, create_retriever
from industry_agent.rag.hybrid_retriever import reciprocal_rank_fusion
from industry_agent.rag.retriever import SQLiteRetriever, analyze_query
from industry_agent.rag.vector_store import (
    DisabledVectorSearcher,
    HashingEmbeddingModel,
    SQLiteVectorSearcher,
    VectorSearchConfig,
    build_chunk_vector_index,
    describe_vector_retrieval,
)


class PromptTemplateTests(unittest.TestCase):
    def test_manual_qa_prompt_contains_anti_hallucination_rules(self) -> None:
        result = build_manual_qa_system_prompt("[参考1] 产品：电钻\n只能使用指定充电器。")

        self.assertTrue(result.has_context)
        self.assertEqual(result.rule_count, len(MANUAL_QA_RULES))
        self.assertIn("只允许基于【参考资料】回答", result.content)
        self.assertIn("不得使用常识补全、猜测或编造", result.content)
        self.assertIn("资料无法回答", result.content)
        self.assertIn("只能使用指定充电器", result.content)

    def test_manual_qa_prompt_uses_no_context_fallback(self) -> None:
        result = build_manual_qa_system_prompt("")

        self.assertFalse(result.has_context)
        self.assertIn("（未找到相关资料）", result.content)

    def test_customer_service_prompt_contains_generation_rules(self) -> None:
        result = build_customer_service_system_prompt("结论：可申请退款。\n处理步骤：先提交售后申请。")

        self.assertTrue(result.has_context)
        self.assertEqual(result.rule_count, len(CUSTOMER_SERVICE_RULES))
        self.assertIn("只允许基于【客服策略骨架】回答", result.content)
        self.assertIn("不得编造平台政策", result.content)
        self.assertIn("结论：可申请退款。", result.content)


class RetrievalConfigTests(unittest.TestCase):
    def test_vector_retrieval_reports_not_built_for_missing_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status = describe_vector_retrieval(db_path=Path(tmpdir) / "missing.sqlite")

        self.assertTrue(status["enabled"])
        self.assertEqual(status["embedding_model"], "BAAI/bge-small-zh-v1.5")
        self.assertEqual(status["status"], "not_built")

    def test_retriever_reports_hybrid_lexical_strategy(self) -> None:
        retriever = SQLiteRetriever(vector_searcher=DisabledVectorSearcher())
        status = retriever.retrieval_status()

        self.assertEqual(status["strategy"], "hybrid_lexical_with_optional_vector")
        self.assertEqual(status["channels"], ["like", "fts5_bm25", "vector"])
        self.assertEqual(status["lexical_channels"], ["like", "fts5_bm25"])
        self.assertIn("vector", status)

    def test_vector_score_can_participate_in_rerank_when_configured(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("电钻充电注意事项")
        row = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "vector_hit",
                "title": "电池充电注意事项",
                "text": "充电前请确认电池组和充电器状态。",
                "product_name": "电钻",
                "image_ids": "[]",
                "metadata": '{"clean_score": 1.0, "semantic_type": "procedure"}',
                "fts_hit": 0,
                "fts_rank": None,
                "_vector_score": 0.8,
                "_retrieval_channels": ["vector"],
            },
            analysis,
        )

        self.assertIn("vector", row["_retrieval_channels"])
        self.assertEqual(row["_retrieval_strategy"], "like+fts+vector_optional")
        self.assertGreater(row["_score"], 20.0)

    def test_hashing_embedding_is_normalized(self) -> None:
        vector = HashingEmbeddingModel(dimensions=64).embed("电钻 充电 指示灯 flashing battery")
        norm = sum(value * value for value in vector) ** 0.5

        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_embedding_manager_uses_current_hashing_backend(self) -> None:
        manager = EmbeddingManager(model_name="hashing-ngram-v1", dimensions=64)
        vector = manager.encode_query("电钻充电指示灯")

        self.assertEqual(manager.dimension, 64)
        self.assertEqual(len(vector), 64)

    def test_rrf_prefers_documents_seen_in_multiple_rank_lists(self) -> None:
        merged = reciprocal_rank_fusion(
            [
                [{"chunk_id": "a"}, {"chunk_id": "b"}],
                [{"chunk_id": "b"}, {"chunk_id": "c"}],
            ]
        )

        self.assertEqual(merged[0]["chunk_id"], "b")
        self.assertIn("_rrf_score", merged[0])

    def test_retriever_factory_defaults_to_hybrid_strategy(self) -> None:
        retriever = create_retriever("hybrid")
        status = retriever.retrieval_status()

        self.assertEqual(status["strategy"], "hybrid_rrf")
        self.assertEqual(status["channels"], ["sqlite", "vector", "rrf"])

    def test_retriever_factory_supports_vector_only_mode(self) -> None:
        retriever = create_retriever("vector")

        self.assertIsInstance(retriever, VectorOnlyRetriever)

    def test_build_vector_index_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "index.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.executescript(
                    """
                    CREATE TABLE chunks (
                      chunk_id TEXT PRIMARY KEY,
                      manual_id TEXT NOT NULL,
                      product_name TEXT NOT NULL,
                      source_path TEXT NOT NULL,
                      title TEXT NOT NULL,
                      text TEXT NOT NULL,
                      image_ids TEXT NOT NULL,
                      section_index INTEGER NOT NULL,
                      chunk_index INTEGER NOT NULL,
                      char_count INTEGER NOT NULL,
                      metadata TEXT NOT NULL
                    );
                    """
                )
                chunks = [
                    KnowledgeChunk(
                        chunk_id="a",
                        manual_id="m",
                        product_name="电钻",
                        source_path="Knowledge_base/电钻手册.txt",
                        title="电池充电",
                        text="充电时红灯闪烁表示正在充电。",
                        image_ids=[],
                        section_index=0,
                        chunk_index=0,
                        char_count=20,
                        metadata={"semantic_type": "troubleshooting"},
                    ),
                    KnowledgeChunk(
                        chunk_id="b",
                        manual_id="m",
                        product_name="洗碗机",
                        source_path="Knowledge_base/洗碗机手册.txt",
                        title="安装要求",
                        text="安装洗碗机前请检查进水和排水条件。",
                        image_ids=[],
                        section_index=1,
                        chunk_index=1,
                        char_count=20,
                        metadata={"semantic_type": "procedure"},
                    ),
                ]
                conn.executemany(
                    """
                    INSERT INTO chunks (
                      chunk_id, manual_id, product_name, source_path, title, text,
                      image_ids, section_index, chunk_index, char_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.chunk_id,
                            chunk.manual_id,
                            chunk.product_name,
                            chunk.source_path,
                            chunk.title,
                            chunk.text,
                            "[]",
                            chunk.section_index,
                            chunk.chunk_index,
                            chunk.char_count,
                            "{}",
                        )
                        for chunk in chunks
                    ],
                )
                summary = build_chunk_vector_index(
                    conn,
                    chunks,
                    config=VectorSearchConfig(
                        embedding_model="hashing-ngram-v1",
                        dimensions=384,
                        index_path=db_path,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            self.assertEqual(summary["chunk_count"], 2)
            status = describe_vector_retrieval(db_path=db_path)
            self.assertEqual(status["status"], "ready")
            self.assertEqual(status["chunk_count"], 2)

            rows = SQLiteVectorSearcher(
                db_path,
                config=VectorSearchConfig(
                    embedding_model="hashing-ngram-v1",
                    dimensions=384,
                    index_path=db_path,
                ),
            ).search("电钻红灯闪烁怎么充电", limit=2)

        self.assertTrue(rows)
        self.assertEqual(rows[0]["chunk_id"], "a")
        self.assertIn("_vector_score", rows[0])


if __name__ == "__main__":
    unittest.main()
