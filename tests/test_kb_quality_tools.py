from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.kb.build_index import (
    build_english_summary_segments,
    summarize_english_summary_segment_quality,
    summarize_chunk_quality,
    summarize_manual_quality,
)
from industry_agent.kb.models import KnowledgeChunk, ManualDocument
from scripts.sample_kb_chunks import load_chunk_records, select_chunk_samples


class BuildSummaryStatsTests(unittest.TestCase):
    def test_summarize_chunk_quality_reports_core_ratios_and_counts(self) -> None:
        chunks = [
            KnowledgeChunk(
                chunk_id="a",
                manual_id="m1",
                product_name="产品A",
                source_path="Knowledge_base/a.txt",
                title="安装",
                text="1. 打开盖子。\n2. 安装电池。",
                image_ids=["img1"],
                section_index=0,
                chunk_index=0,
                char_count=40,
                metadata={
                    "chunk_type": "procedure",
                    "semantic_type": "procedure",
                    "clean_score": 1.0,
                    "has_overlap_context": True,
                    "step_count": 2,
                    "domain_label": "",
                },
            ),
            KnowledgeChunk(
                chunk_id="b",
                manual_id="m2",
                product_name="产品B",
                source_path="Knowledge_base/b.txt",
                title="参数",
                text="尺寸: 120 mm\n重量: 1.2 kg",
                image_ids=[],
                section_index=0,
                chunk_index=1,
                char_count=28,
                metadata={
                    "chunk_type": "specification",
                    "semantic_type": "specification",
                    "clean_score": 0.65,
                    "has_overlap_context": False,
                    "step_count": 0,
                    "domain_label": "camera",
                },
            ),
        ]

        summary = summarize_chunk_quality(chunks, max_chunk_chars=100)

        self.assertEqual(summary["chunk_type_counts"]["procedure"], 1)
        self.assertEqual(summary["chunk_type_counts"]["specification"], 1)
        self.assertEqual(summary["semantic_type_counts"]["procedure"], 1)
        self.assertEqual(summary["domain_label_counts"]["camera"], 1)
        self.assertEqual(summary["step_count_distribution"]["2-3"], 1)
        self.assertEqual(summary["step_count_distribution"]["0"], 1)
        self.assertEqual(summary["with_image_ratio"], 0.5)
        self.assertEqual(summary["low_clean_score_ratio"], 0.5)
        self.assertEqual(summary["oversized_chunk_ratio"], 0.0)
        self.assertEqual(summary["heading_only_ratio"], 0.0)
        self.assertEqual(summary["long_title_ratio"], 0.0)

    def test_summarize_manual_quality_reports_top_manuals(self) -> None:
        manuals = [
            ManualDocument("m1", "产品A", PROJECT_ROOT / "Knowledge_base" / "a.txt", "", ["img1"], 1, "json"),
            ManualDocument("m2", "产品B", PROJECT_ROOT / "Knowledge_base" / "b.txt", "", ["img1", "img2"], 12, "tail-recovery"),
        ]
        chunks = [
            KnowledgeChunk("a", "m1", "产品A", "Knowledge_base/a.txt", "安装", "文本A", [], 0, 0, 30, {}),
            KnowledgeChunk("b", "m1", "产品A", "Knowledge_base/a.txt", "安装2", "文本B", ["img1"], 0, 1, 40, {}),
            KnowledgeChunk("c", "m2", "产品B", "Knowledge_base/b.txt", "参数", "文本C", [], 0, 2, 20, {}),
        ]

        summary = summarize_manual_quality(manuals, chunks)

        self.assertEqual(summary["top_chunk_manuals"][0]["manual_id"], "m1")
        self.assertEqual(summary["top_chunk_manuals"][0]["chunk_count"], 2)
        self.assertEqual(summary["top_chunk_manuals"][0]["image_chunk_count"], 1)
        self.assertEqual(summary["parse_mode_counts"]["json"], 1)
        self.assertEqual(summary["parse_mode_counts"]["tail-recovery"], 1)
        self.assertEqual(summary["attachment_outliers"][0]["manual_id"], "m2")

    def test_build_english_summary_segments_groups_sub_manuals(self) -> None:
        chunks = [
            KnowledgeChunk(
                chunk_id="a",
                manual_id="汇总英文手册",
                product_name="汇总英文",
                source_path="Knowledge_base/汇总英文手册.txt",
                title="Camera Care",
                text="Text A",
                image_ids=[],
                section_index=10,
                chunk_index=0,
                char_count=100,
                metadata={
                    "domain_label": "camera",
                    "section_domain_label": "camera",
                    "domain_segment_label": "camera",
                    "domain_segment_index": 0,
                    "sub_manual_id": "汇总英文手册:camera:0",
                },
            ),
            KnowledgeChunk(
                chunk_id="b",
                manual_id="汇总英文手册",
                product_name="汇总英文",
                source_path="Knowledge_base/汇总英文手册.txt",
                title="CF Card",
                text="Text B",
                image_ids=["img1"],
                section_index=12,
                chunk_index=1,
                char_count=140,
                metadata={
                    "domain_label": "camera",
                    "section_domain_label": "camera",
                    "domain_segment_label": "camera",
                    "domain_segment_index": 0,
                    "sub_manual_id": "汇总英文手册:camera:0",
                },
            ),
            KnowledgeChunk(
                chunk_id="c",
                manual_id="汇总英文手册",
                product_name="汇总英文",
                source_path="Knowledge_base/汇总英文手册.txt",
                title="Anchoring",
                text="Text C",
                image_ids=[],
                section_index=30,
                chunk_index=2,
                char_count=120,
                metadata={
                    "domain_label": "boat",
                    "section_domain_label": "boat",
                    "domain_segment_label": "boat",
                    "domain_segment_index": 1,
                    "sub_manual_id": "汇总英文手册:boat:1",
                },
            ),
        ]

        summary = build_english_summary_segments(chunks)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["sub_manual_id"], "汇总英文手册:camera:0")
        self.assertEqual(summary[0]["chunk_count"], 2)
        self.assertEqual(summary[0]["image_chunk_count"], 1)
        self.assertEqual(summary[0]["section_range"], [10, 12])
        self.assertEqual(summary[1]["sub_manual_id"], "汇总英文手册:boat:1")

    def test_summarize_english_summary_segment_quality_reports_ratios(self) -> None:
        summary = summarize_english_summary_segment_quality(
            [
                {"sub_manual_id": "a", "domain_label": "camera", "chunk_count": 5},
                {"sub_manual_id": "b", "domain_label": "camera", "chunk_count": 2},
                {"sub_manual_id": "c", "domain_label": "boat", "chunk_count": 1},
            ]
        )

        self.assertEqual(summary["segment_count"], 3)
        self.assertEqual(summary["short_segment_ratio"], 0.6667)
        self.assertEqual(summary["singleton_segment_ratio"], 0.3333)
        self.assertEqual(summary["domain_segment_counts"]["camera"], 2)


class ChunkSamplingScriptTests(unittest.TestCase):
    def test_load_and_select_chunk_samples_support_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chunks.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"chunk_id":"a","manual_id":"m1","product_name":"A","title":"安装","text":"步骤文本","char_count":20,"image_ids":["img1"],"metadata":{"chunk_type":"procedure","semantic_type":"procedure"}}',
                        '{"chunk_id":"b","manual_id":"m2","product_name":"B","title":"参数","text":"参数文本","char_count":18,"image_ids":[],"metadata":{"chunk_type":"specification","semantic_type":"specification"}}',
                    ]
                ),
                encoding="utf-8",
            )
            records = load_chunk_records(path)

        image_samples = select_chunk_samples(records, chunk_type="procedure", has_image=True, limit=5)
        spec_samples = select_chunk_samples(records, manual_id="m2", limit=5)
        domain_samples = select_chunk_samples(records, domain_label="procedure", limit=5)

        self.assertEqual(len(records), 2)
        self.assertEqual(len(image_samples), 1)
        self.assertEqual(image_samples[0]["chunk_id"], "a")
        self.assertEqual(len(spec_samples), 1)
        self.assertEqual(spec_samples[0]["chunk_id"], "b")
        self.assertEqual(len(domain_samples), 0)

    def test_select_chunk_samples_supports_domain_label_filter(self) -> None:
        records = [
            {
                "chunk_id": "a",
                "manual_id": "m1",
                "product_name": "A",
                "title": "Title A",
                "text": "Text A",
                "char_count": 12,
                "image_ids": [],
                "metadata": {"chunk_type": "general", "semantic_type": "general", "domain_label": "camera"},
            },
            {
                "chunk_id": "b",
                "manual_id": "m1",
                "product_name": "A",
                "title": "Title B",
                "text": "Text B",
                "char_count": 12,
                "image_ids": [],
                "metadata": {"chunk_type": "general", "semantic_type": "general", "domain_label": "boat"},
            },
        ]

        samples = select_chunk_samples(records, domain_label="camera", limit=5)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["chunk_id"], "a")


if __name__ == "__main__":
    unittest.main()
