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


if __name__ == "__main__":
    unittest.main()
