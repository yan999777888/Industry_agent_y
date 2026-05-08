from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.kb.chunker import chunk_manual
from industry_agent.kb.models import ManualDocument
from industry_agent.kb.parser import normalize_manual_text


class KnowledgeCleaningTests(unittest.TestCase):
    def test_normalize_manual_text_decodes_unicode_and_common_ocr_glue(self) -> None:
        cleaned = normalize_manual_text(
            "# Safety \\u00b7Keep theback-up battery away. "
            "The cord canaccidentallychokethechild and cause electricalshock. "
            "Use a bristle brush tocleanany grill surface. Turn off power at the main power sup ply."
        )

        self.assertIn("·Keep the back-up battery", cleaned)
        self.assertIn("can accidentally choke the child", cleaned)
        self.assertIn("electrical shock", cleaned)
        self.assertIn("to clean any grill surface", cleaned)
        self.assertIn("main power supply", cleaned)
        self.assertNotIn("\\u00b7", cleaned)

    def test_chunk_manual_skips_toc_like_sections(self) -> None:
        manual = ManualDocument(
            manual_id="测试手册",
            product_name="测试产品",
            source_path=PROJECT_ROOT / "Knowledge_base" / "测试手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# Contents\n"
            "Safety information..........................10\n"
            "Installation.................................20\n"
            "Operation....................................30\n"
            "Maintenance..................................40\n\n"
            "# Installation\n"
            "Install the battery before first use."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Installation")
        self.assertNotIn("Contents", chunks[0].text)

    def test_chunk_manual_skips_low_value_heading_fragments(self) -> None:
        manual = ManualDocument(
            manual_id="测试手册",
            product_name="测试产品",
            source_path=PROJECT_ROOT / "Knowledge_base" / "测试手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = "# Warranty\n\n# Cleaning\n1. Unplug the product.\n2. Wipe the surface."

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Cleaning")

    def test_chunk_manual_tags_english_summary_domain(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )

        chunks = chunk_manual(
            manual,
            "# Voice Recording\nSelect the Record in the main menu to enter voice record mode.",
            project_root=PROJECT_ROOT,
        )

        self.assertEqual(chunks[0].metadata["domain_label"], "ereader")
        self.assertGreaterEqual(chunks[0].metadata["clean_score"], 0.8)

    def test_chunk_manual_tags_additional_english_summary_domains(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )

        cases = {
            "lawn_mower": "# Cleaning under the Mower Deck\nDisengage the blade-control switch and engage the parking brake.",
            "coffee_machine": "# COFFEE PREPARATION\nFill the water tank and press the espresso button.",
            "toothbrush": "# Brush Head\nThe toothbrush brush head and pressure sensor provide brushing feedback.",
            "fax": "# Product Safety Guide\nConnect the telephone line cord near the telephone wall jack for fax models.",
        }

        for expected_domain, marked_text in cases.items():
            with self.subTest(expected_domain=expected_domain):
                chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)
                self.assertEqual(chunks[0].metadata["domain_label"], expected_domain)

    def test_chunk_manual_tags_semantic_types(self) -> None:
        manual = ManualDocument(
            manual_id="测试手册",
            product_name="测试产品",
            source_path=PROJECT_ROOT / "Knowledge_base" / "测试手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )

        cases = {
            "procedure": "# Installing the Battery\n1. Open the cover.\n2. Insert the battery.",
            "safety_warning": "# WARNING\nDo not expose the battery to fire or water.",
            "troubleshooting": "# Indicator Light\nIf the indicator is flashing, remove the battery and recharge it.",
            "parts_list": "# Package Contents\nThe package includes the base, handset, and accessories.",
            "specification": "# Specifications\nDimensions: 120 mm. Weight: 1.2 kg.",
        }

        for expected_type, marked_text in cases.items():
            with self.subTest(expected_type=expected_type):
                chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)
                self.assertEqual(chunks[0].metadata["semantic_type"], expected_type)


if __name__ == "__main__":
    unittest.main()
