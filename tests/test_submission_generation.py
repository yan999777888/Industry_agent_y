from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_submission import normalize_submission_answer, rows_from_debug_records


class SubmissionGenerationTests(unittest.TestCase):
    def test_normalize_submission_answer_strips_internal_sections(self) -> None:
        raw = (
            "结论：根据现有资料无法回答此问题。\n\n"
            "操作/说明：请补充产品名称。\n\n"
            "相关图片：\n- Manual16_51"
        )
        normalized = normalize_submission_answer(
            raw,
            question="洗碗机安装有什么要求？",
            sources=["洗碗机"],
        )
        self.assertNotIn("相关图片", normalized)
        self.assertNotIn("Manual16_51", normalized)
        self.assertIn("说明书内容", normalized)

    def test_normalize_submission_answer_uses_customer_service_fallback(self) -> None:
        normalized = normalize_submission_answer(
            "根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。",
            question="我想退款，退款多久能到账？",
            sources=["customer_service_policy"],
        )
        self.assertIn("订单号", normalized)
        self.assertNotIn("根据现有资料无法回答此问题", normalized)

    def test_normalize_submission_answer_preserves_partial_multi_question_content(self) -> None:
        raw = (
            "问题1：发票问题通常需要确认订单号、开票类型、抬头信息以及当前开票状态。\n"
            "问题2：根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。"
        )
        normalized = normalize_submission_answer(
            raw,
            question="发票类型是什么？多久能收到？",
            sources=["customer_service_policy"],
        )
        self.assertIn("发票问题通常需要确认订单号", normalized)
        self.assertNotIn("根据现有资料无法回答此问题", normalized)
        self.assertNotIn("您好，相关情况需要结合订单信息", normalized)

    def test_normalize_submission_answer_does_not_submit_question_echo_only(self) -> None:
        normalized = normalize_submission_answer(
            "问题1：Can this eReader record voice? If so, how do I operate this feature?\n"
            "问题2：根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。",
            question='"Can this eReader record voice? If so, how do I operate this feature?"',
            sources=[],
        )
        self.assertIn("说明书内容", normalized)
        self.assertNotEqual(normalized.strip("。"), "Can this eReader record voice? If so, how do I operate this feature?")

    def test_normalize_submission_answer_compresses_customer_service_duplicates(self) -> None:
        raw = (
            "问题1：发票问题通常需要确认订单号、开票类型、抬头信息以及当前开票状态。"
            " 建议先准备订单号、开票类型、发票抬头、税号和接收邮箱等信息。"
            " 一般建议先确认订单是否满足开票条件，再提交或修改开票信息，必要时联系人工客服协助处理。\n"
            "问题2：发票问题通常需要确认订单号、开票类型、抬头信息以及当前开票状态。"
            " 建议先准备订单号、开票类型、发票抬头、税号和接收邮箱等信息。"
            " 一般建议先确认订单是否满足开票条件，再提交或修改开票信息，必要时联系人工客服协助处理。"
        )
        normalized = normalize_submission_answer(
            raw,
            question="公司发票抬头怎么开，写错了能重开吗？",
            sources=["customer_service_policy"],
        )
        self.assertEqual(normalized.count("发票问题通常需要确认订单号"), 1)
        self.assertLessEqual(normalized.count("建议先准备订单号"), 1)

    def test_normalize_submission_answer_appends_pic_markers_and_image_ids(self) -> None:
        normalized = normalize_submission_answer(
            "安装电池：取下电池盖，装入新电池，确认正负极正确。",
            question="如何给空调遥控器安装电池？",
            sources=["空调"],
            image_ids=["Manual01_2", "Manual01_3"],
        )

        self.assertIn("<PIC><PIC>", normalized)
        self.assertIn('";["Manual01_2", "Manual01_3"]', normalized)

    def test_normalize_submission_answer_strips_internal_extractive_tail(self) -> None:
        normalized = normalize_submission_answer(
            "Battery switches are located in the battery compartment. "
            "The answer is extracted from the retrieved manual evidence. "
            "Please follow the original manual for safety-critical operation.",
            question="How do I use the battery switches?",
            sources=["汇总英文"],
        )

        self.assertIn("Battery switches", normalized)
        self.assertNotIn("retrieved manual evidence", normalized)
        self.assertNotIn("safety-critical", normalized)

    def test_normalize_submission_answer_uses_references_when_model_refuses(self) -> None:
        normalized = normalize_submission_answer(
            "根据现有资料无法准确回答此问题。请补充产品名称、型号、故障现象或上传更清晰的图片后再试。",
            question="如何给空调遥控器安装电池？",
            sources=["空调"],
            image_ids=["Manual01_2"],
            references=[
                {
                    "title": "安装须知",
                    "text_snippet": "# 安装须知 使用符合空调额定参数的标准断路器和保险丝，否则可能导致触电或产品故障。",
                },
                {
                    "title": "安装电池",
                    "text_snippet": "# 安装电池 使用遥控器前，请先安装电池，适用电池型号为 7 号。1 取下电池盖。2 装入新电池，确保电池正、负极安装正确。",
                }
            ],
        )

        self.assertIn("安装电池", normalized)
        self.assertIn("取下电池盖", normalized)
        self.assertIn("<PIC>", normalized)
        self.assertNotIn("无法准确定位", normalized)

    def test_rows_from_debug_records_reuses_existing_raw_responses(self) -> None:
        rows = rows_from_debug_records(
            [
                {
                    "id": "72",
                    "question": "如何给空调遥控器安装电池？",
                    "response": {
                        "data": {
                            "answer": "根据现有资料无法准确回答此问题。",
                            "sources": ["空调"],
                            "image_ids": ["Manual01_2"],
                            "references": [
                                {
                                    "title": "安装电池",
                                    "text_snippet": "安装电池 使用遥控器前，请先安装电池。1 取下电池盖。2 装入新电池。",
                                }
                            ],
                        }
                    },
                }
            ],
            "fallback",
        )

        self.assertEqual(rows[0]["id"], "72")
        self.assertIn("安装电池", rows[0]["ret"])
        self.assertIn('";["Manual01_2"]', rows[0]["ret"])


if __name__ == "__main__":
    unittest.main()
