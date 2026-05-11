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
from industry_agent.kb.parser import attach_image_markers, normalize_manual_text


class KnowledgeCleaningTests(unittest.TestCase):
    def test_normalize_manual_text_decodes_unicode_and_common_ocr_glue(self) -> None:
        cleaned = normalize_manual_text(
            "# Safety \\u00b7Keep theback-up battery away. "
            "The cord canaccidentallychokethechild and cause electricalshock. "
            "Use a bristle brush tocleanany grill surface. Turn off power at the main power sup ply."
        )

        self.assertIn("· Keep the back-up battery", cleaned)
        self.assertIn("can accidentally choke the child", cleaned)
        self.assertIn("electrical shock", cleaned)
        self.assertIn("to clean any grill surface", cleaned)
        self.assertIn("main power supply", cleaned)
        self.assertNotIn("\\u00b7", cleaned)

    def test_normalize_manual_text_fixes_additional_english_ocr_noise(self) -> None:
        cleaned = normalize_manual_text(
            "# Preventing Serious Injury or Death: Ioprevent tire, excessiveheat, chemical leakage, and explosions, tollow the safeguardsbelow:\n"
            "-Do not use paint thinner or other oraanic solvents to clean the eauioment.\n"
            "If the surrounding is dusty, humid, oroily, thedustonthe power outlet may becomemoist andshort-circuit theoutlet tocausea fire."
        )

        self.assertIn("To prevent tire, excessive heat", cleaned)
        self.assertIn("follow the safeguards below", cleaned)
        self.assertIn("- Do not use paint thinner or other organic solvents to clean the equipment.", cleaned)
        self.assertIn("or oily, the dust on the power outlet may become moist", cleaned)
        self.assertIn("the outlet to cause a fire", cleaned)

    def test_attach_image_markers_keeps_sequential_binding_for_normal_manuals(self) -> None:
        result = attach_image_markers("A<PIC>B<PIC>C", ["img1", "img2"])

        self.assertEqual(result.strategy, "sequential")
        self.assertEqual(result.attached_image_ids, ["img1", "img2"])
        self.assertEqual(result.unmatched_pic_count, 0)
        self.assertIn("[[PIC:img1]]", result.marked_text)

    def test_attach_image_markers_suppresses_extreme_pic_image_mismatch(self) -> None:
        text = "<PIC>" * 150
        image_ids = [f"img{i}" for i in range(20)]

        result = attach_image_markers(text, image_ids)

        self.assertEqual(result.strategy, "suppress_misaligned")
        self.assertEqual(result.attached_image_ids, [])
        self.assertEqual(result.attached_count, 0)
        self.assertEqual(result.suppressed_image_count, 20)
        self.assertEqual(result.unmatched_pic_count, 150)
        self.assertNotIn("[[PIC:img0]]", result.marked_text)
        self.assertIn("[[PIC_MISSING]]", result.marked_text)

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
                self.assertIn("domain_segment_index", chunks[0].metadata)
                self.assertEqual(chunks[0].metadata["domain_segment_index"], 0)
                self.assertEqual(chunks[0].metadata["sub_manual_id"], f"汇总英文手册:{expected_domain}:0")

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

    def test_chunk_manual_adds_overlap_for_procedure_sections(self) -> None:
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
            "# 安装步骤\n"
            "1. 打开电池仓并确认卡扣已经完全松开。\n"
            "2. 将电池沿导轨缓慢推入，直到听到卡扣声。\n"
            "3. 轻拉电池确认已经锁定，然后再开始充电。"
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT, max_chars=55)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("2.", chunks[0].text)
        self.assertIn("2.", chunks[1].text)
        self.assertEqual(int(chunks[1].metadata["step_count"]), 2)
        self.assertTrue(bool(chunks[1].metadata["has_overlap_context"]))

    def test_chunk_manual_overlap_never_exceeds_max_chars(self) -> None:
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
            "# Slope Safety\n"
            "1. " + ("A" * 500) + "\n"
            "2. " + ("B" * 500) + "\n"
            "3. " + ("C" * 500)
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT, max_chars=700)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(chunk.char_count <= 700 for chunk in chunks))

    def test_chunk_manual_binds_picture_with_neighbor_text(self) -> None:
        manual = ManualDocument(
            manual_id="测试手册",
            product_name="测试产品",
            source_path=PROJECT_ROOT / "Knowledge_base" / "测试手册.txt",
            text="",
            image_ids=["drill0_17"],
            pic_count=1,
            parse_mode="test",
        )
        marked_text = (
            "# 指示灯说明\n"
            "请先观察设备顶部的指示灯区域。\n"
            "[[PIC:drill0_17]]\n"
            "如果红灯持续闪烁，请先断开电源并重新插入电池。"
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT, max_chars=120)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].image_ids, ["drill0_17"])
        self.assertIn("请先观察设备顶部的指示灯区域", chunks[0].text)
        self.assertIn("如果红灯持续闪烁", chunks[0].text)
        self.assertEqual(chunks[0].metadata["image_count"], 1)
        self.assertEqual(chunks[0].metadata["chunk_type"], "troubleshooting")

    def test_chunk_manual_filters_english_toc_sections_more_aggressively(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# Introduction Item Check List. 3\n"
            "Handling Precautions.8\n"
            "Nomenclature....10\n"
            "Conventions Used in this Manual..16\n\n"
            "# Safety Warnings\n"
            "Do not expose the battery to fire or water."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Safety Warnings")
        self.assertNotIn("Item Check List", chunks[0].text)

    def test_chunk_manual_filters_single_line_english_toc_sections(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# Flash Photography 91 Using the Built-in Flash.92 Using Flash Units..98\n\n"
            "# Camera Care\n"
            "Do not drop the camera or expose it to strong magnetic fields."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Camera Care")

    def test_chunk_manual_shortens_long_english_titles(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = "# Camera Care ● This camera is a precision instrument. Do not drop it or subject it to physical shock."

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Camera Care")

    def test_chunk_manual_shortens_colon_style_english_titles(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = "# Preventing Serious Injury or Death: To prevent fire, excessive heat, chemical leakage, and explosions."

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Preventing Serious Injury or Death")

    def test_chunk_manual_shortens_follow_style_english_titles(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = "# Safety Warnings Follow these safeguards and use the equipment properly to prevent injury."

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Safety Warnings")

    def test_chunk_manual_infers_english_domain_on_neighboring_generic_section(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# Camera Care\n"
            "Do not drop the camera or subject it to strong magnetic fields.\n\n"
            "# Handling Precautions\n"
            "Keep the equipment dry and store it in a clean place.\n\n"
            "# CF Card\n"
            "Do not drop the memory card or subject it to vibration."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[1].title, "Handling Precautions")
        self.assertEqual(chunks[1].metadata["domain_label"], "camera")
        self.assertTrue(bool(chunks[1].metadata["domain_inferred"]))
        self.assertEqual(chunks[1].metadata["domain_segment_index"], 0)
        self.assertEqual(chunks[1].metadata["sub_manual_id"], "汇总英文手册:camera:0")

    def test_chunk_manual_splits_english_domain_segments_when_product_changes(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# Camera Care\n"
            "Do not drop the camera or touch the lens contacts.\n\n"
            "# Handling Precautions\n"
            "Store the equipment in a dry location.\n\n"
            "# Anchoring\n"
            "Always anchor from the bow and select an anchor suitable for your boat."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].metadata["domain_label"], "camera")
        self.assertEqual(chunks[1].metadata["domain_label"], "camera")
        self.assertEqual(chunks[2].metadata["domain_label"], "boat")
        self.assertEqual(chunks[0].metadata["domain_segment_index"], 0)
        self.assertEqual(chunks[1].metadata["domain_segment_index"], 0)
        self.assertEqual(chunks[2].metadata["domain_segment_index"], 1)
        self.assertEqual(chunks[0].metadata["sub_manual_id"], "汇总英文手册:camera:0")
        self.assertEqual(chunks[2].metadata["sub_manual_id"], "汇总英文手册:boat:1")

    def test_chunk_manual_smooths_generic_short_english_domain_runs(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# Camera Care\n"
            "Do not drop the camera or touch the lens contacts.\n\n"
            "# Select [Language]\n"
            "The language screen will appear.\n\n"
            "# CF Card\n"
            "Do not drop the memory card or subject it to vibration."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 3)
        self.assertEqual([chunk.metadata["domain_label"] for chunk in chunks], ["camera", "camera", "camera"])
        self.assertTrue(bool(chunks[1].metadata["domain_inferred"]))
        self.assertEqual(chunks[1].metadata["sub_manual_id"], "汇总英文手册:camera:0")

    def test_chunk_manual_infers_ambiguous_boat_sections_from_neighbors(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# Navigation and anchor lights switch\n"
            "Use the switch to control the navigation lights on the boat.\n\n"
            "# Remote control levers\n"
            "Move the levers smoothly during docking and low-speed operation.\n\n"
            "# Wet storage compartment\n"
            "Open the compartment to store wet items while on board."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual([chunk.metadata["domain_label"] for chunk in chunks], ["boat", "boat", "boat"])
        self.assertTrue(bool(chunks[1].metadata["domain_inferred"]))

    def test_chunk_manual_filters_long_index_style_sections(self) -> None:
        manual = ManualDocument(
            manual_id="汇总英文手册",
            product_name="汇总英文",
            source_path=PROJECT_ROOT / "Knowledge_base" / "汇总英文手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = (
            "# R Remote control lever checks................... 87 "
            "Remote control levers.............................. 32 "
            "Required equipment................................. 13 "
            "Reverse RPM control............................... 49\n\n"
            "# Wet storage compartment\n"
            "Open the hatch to access the wet storage compartment."
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Wet storage compartment")

    def test_chunk_manual_strips_layout_codes_and_isolated_markers(self) -> None:
        manual = ManualDocument(
            manual_id="发电机手册",
            product_name="发电机",
            source_path=PROJECT_ROOT / "Knowledge_base" / "发电机手册.txt",
            text="",
            image_ids=["generator_01"],
            pic_count=1,
            parse_mode="test",
        )
        marked_text = (
            "# 废气有毒\n"
            "①\n"
            "切勿在密闭空间内启动发动机。\n"
            "[[PIC:generator_01]]\n"
            "AE01019"
        )

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertNotIn("①", chunks[0].text)
        self.assertNotIn("AE01019", chunks[0].text)
        self.assertIn("切勿在密闭空间内启动发动机", chunks[0].text)

    def test_chunk_manual_normalizes_titles_with_layout_codes_and_page_suffix(self) -> None:
        manual = ManualDocument(
            manual_id="测试手册",
            product_name="测试产品",
            source_path=PROJECT_ROOT / "Knowledge_base" / "测试手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = "# 产品识别码 AE00012 17\n请记录该识别码。"

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "产品识别码")
        self.assertNotIn("AE00012", chunks[0].text)

    def test_chunk_manual_filters_heading_only_chunk(self) -> None:
        manual = ManualDocument(
            manual_id="测试手册",
            product_name="测试产品",
            source_path=PROJECT_ROOT / "Knowledge_base" / "测试手册.txt",
            text="",
            image_ids=[],
            pic_count=0,
            parse_mode="test",
        )
        marked_text = "# 显示\n\n# 使用方法\n1. 打开设备。"

        chunks = chunk_manual(manual, marked_text, project_root=PROJECT_ROOT)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "使用方法")


if __name__ == "__main__":
    unittest.main()
