from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_submission import normalize_submission_answer


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


if __name__ == "__main__":
    unittest.main()
