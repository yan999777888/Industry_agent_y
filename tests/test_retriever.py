from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.rag.retriever import SQLiteRetriever, analyze_query, extract_keywords


class RetrieverAnalysisTests(unittest.TestCase):
    def test_analyze_query_extracts_models_products_and_phrases(self) -> None:
        analysis = analyze_query("我的DCB107或DCB112型号电钻指示灯闪烁时，这些闪烁标识代表什么含义？")
        self.assertIn("电钻", analysis.products)
        self.assertIn("DCB107", analysis.models)
        self.assertIn("DCB112", analysis.models)
        self.assertIn("指示灯", analysis.keywords)
        self.assertIn("闪烁", analysis.keywords)
        self.assertIn("标识", analysis.keywords)

    def test_analyze_query_expands_synonyms(self) -> None:
        analysis = analyze_query("腕带有别的大小吗，红灯闪了怎么办？")
        self.assertIn("表带", analysis.keywords)
        self.assertIn("尺寸", analysis.keywords)
        self.assertIn("指示灯", analysis.keywords)
        self.assertIn("健身追踪器", analysis.products)

    def test_analyze_query_supports_pin_reset_synonyms(self) -> None:
        analysis = analyze_query("健身追踪器 pin码 忘了怎么重置？")
        self.assertIn("密码", analysis.keywords)
        self.assertIn("设备锁", analysis.keywords)

    def test_extract_keywords_for_long_question_reduces_noise(self) -> None:
        keywords = extract_keywords("这台设备充满电以后红灯还在闪，是过热还是故障？")
        self.assertIn("充电", keywords)
        self.assertIn("红灯", keywords)
        self.assertIn("故障", keywords)
        self.assertNotIn("这台", keywords)

    def test_extract_keywords_filters_english_stopwords(self) -> None:
        keywords = extract_keywords("How to find the approval label of emission control certificate of the boat?")
        self.assertIn("APPROVAL", keywords)
        self.assertIn("LABEL", keywords)
        self.assertIn("EMISSION", keywords)
        self.assertNotIn("HOW", keywords)
        self.assertNotIn("THE", keywords)


class RetrieverScoringTests(unittest.TestCase):
    def test_score_row_prefers_matching_title_intent(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("电钻充电时有什么注意事项？")
        strong = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "a",
                "title": "电池充电注意事项",
                "text": "充电前请确认电池组和充电器状态。",
                "product_name": "电钻",
                "image_ids": "[]",
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        weak = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "b",
                "title": "保养说明",
                "text": "请定期清洁设备外壳。",
                "product_name": "电钻",
                "image_ids": "[]",
                "fts_hit": 0,
                "fts_rank": None,
            },
            analysis,
        )
        self.assertGreater(strong["_score"], weak["_score"])

    def test_score_row_penalizes_expansion_only_generic_titles(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("可编程温控器密码忘了怎么重置？")
        generic = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "generic",
                "title": "设置时间",
                "text": "按菜单键进入设备菜单。",
                "product_name": "可编程温控器",
                "image_ids": "[]",
                "fts_hit": 1,
                "fts_rank": -0.1,
            },
            analysis,
        )
        focused = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "focused",
                "title": "设备锁和 PIN 码重置",
                "text": "可以在设备锁设置中重置 PIN 码。",
                "product_name": "可编程温控器",
                "image_ids": "[]",
                "fts_hit": 1,
                "fts_rank": -0.1,
            },
            analysis,
        )
        self.assertGreater(focused["_score"], generic["_score"])

    def test_score_row_prefers_english_phrase_alignment_inside_summary_manual(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("How to find the approval label of emission control certificate of the boat?")
        focused = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "focused",
                "title": "Approval label of emission control certificate",
                "text": "These labels are attached to each engine unit and to the inside of the engine compartment.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        generic = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "generic",
                "title": "Camera Care",
                "text": "Do not drop the camera or subject it to physical shock.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        self.assertGreater(focused["_score"], generic["_score"])

    def test_score_row_prefers_ereader_voice_chunk_over_airfryer_voice_control(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("Can this eReader record voice?")
        focused = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "focused",
                "title": "Voice Recording Select the Record in the main menu",
                "text": "This E Reader supports voice recording.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        generic = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "generic",
                "title": "Voice control",
                "text": "Use the NutriU app and voice assistant to control the air fryer.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        self.assertGreater(focused["_score"], generic["_score"])

    def test_score_row_uses_english_domain_metadata_to_reduce_cross_product_noise(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("What is the correct procedure for cleaning the side brush of a vacuum cleaner?")
        focused = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "focused",
                "title": "Cleaning the side brush",
                "text": "Remove hair and debris from the vacuum side brush.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "metadata": '{"domain_label": "vacuum", "clean_score": 1.0}',
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        noisy = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "noisy",
                "title": "Cleaning cooking surfaces",
                "text": "Clean the grill cooking surfaces with a brush.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "metadata": '{"domain_label": "grill", "clean_score": 0.7}',
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )

        self.assertGreater(focused["_score"], noisy["_score"])

    def test_score_row_prefers_procedure_metadata_for_how_to_queries(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("How do I clean the side brush of a vacuum cleaner?")
        procedure = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "procedure",
                "title": "Cleaning the side brush",
                "text": "1. Remove the side brush. 2. Clean any hair and debris from the vacuum.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "metadata": '{"domain_label": "vacuum", "clean_score": 1.0, "semantic_type": "procedure", "is_procedure": true}',
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        warning = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "warning",
                "title": "Cleaning Safety",
                "text": "WARNING: Do not clean the product while it is connected to power.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "metadata": '{"domain_label": "vacuum", "clean_score": 1.0, "semantic_type": "safety_warning", "is_warning_only": true}',
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )

        self.assertGreater(procedure["_score"], warning["_score"])

    def test_score_row_prefers_troubleshooting_metadata_for_fault_queries(self) -> None:
        retriever = SQLiteRetriever()
        analysis = analyze_query("指示灯一直闪烁是什么故障？")
        troubleshooting = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "trouble",
                "title": "指示灯闪烁故障",
                "text": "如果指示灯持续闪烁，请检查电池和连接状态。",
                "product_name": "测试产品",
                "image_ids": "[]",
                "metadata": '{"clean_score": 1.0, "semantic_type": "troubleshooting"}',
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )
        generic = retriever._score_row(  # type: ignore[attr-defined]
            {
                "chunk_id": "generic",
                "title": "指示灯位置",
                "text": "指示灯位于设备正面。",
                "product_name": "测试产品",
                "image_ids": "[]",
                "metadata": '{"clean_score": 1.0, "semantic_type": "general"}',
                "fts_hit": 1,
                "fts_rank": -0.2,
            },
            analysis,
        )

        self.assertGreater(troubleshooting["_score"], generic["_score"])


class RetrieverIntegrationTests(unittest.TestCase):
    def test_search_prefers_boating_battery_switch_chunk_when_query_mentions_sailing(self) -> None:
        retriever = SQLiteRetriever()
        if not retriever.db_path.exists():
            self.skipTest("retrieval index not available")

        rows = retriever.search("How do I use the battery conversion feature before sailing?", limit=5)

        self.assertTrue(rows)
        self.assertIn("Battery switches", rows[0]["title"])


if __name__ == "__main__":
    unittest.main()
