from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.agent.question_splitter import split_complex_question
from industry_agent.agent.service import AgentService, ChatRequest


class DummyAgentService(AgentService):
    """Test double that avoids real retrieval and LLM calls."""

    def __init__(self) -> None:
        self.image_index = {}
        self.model = "dummy"
        self.base_url = "http://dummy"
        self.http_client = None

    def generate_response(self, query: str, history=None, image_input=None):  # type: ignore[override]
        image_id = "img_a" if "退换货" in query else "img_b" if "运费" in query else "img_c"
        return {
            "answer": f"回答({query})",
            "image_ids": [image_id],
            "images": [{"image_id": image_id, "file_name": f"{image_id}.png", "path": f"Knowledge_base/插图/{image_id}.png", "exists": True}],
            "sources": ["测试产品"],
            "references": [{"chunk_id": "chunk_1", "title": "测试标题", "text_snippet": query[:50], "product_name": "测试产品", "score": "99"}],
            "confidence": 0.8,
            "retrieval_debug": {"query": query},
        }

    def _call_llm(self, messages):  # type: ignore[override]
        return messages[0]["content"]

    def _merge_subquestion_answers(self, *, original_question, sub_questions, sub_results):  # type: ignore[override]
        lines = []
        for index, (sub_question, result) in enumerate(zip(sub_questions, sub_results), start=1):
            lines.append(f"问题{index}：{sub_question.normalized_text}")
            lines.append(result["answer"])
        return "\n".join(lines)


class QuestionSplitterTests(unittest.TestCase):
    def test_split_quoted_multiline_question(self) -> None:
        question = '"请问你们家的商品支持7天无理由退换货吗？",\n"需要自己承担运费吗？"'
        sub_questions = split_complex_question(question)
        self.assertEqual(len(sub_questions), 2)
        self.assertEqual(sub_questions[0].normalized_text, "你们家的商品支持7天无理由退换货吗？")
        self.assertEqual(sub_questions[1].normalized_text, "需要自己承担运费吗？")

    def test_split_plain_multi_question(self) -> None:
        question = "我想取消订单，但是订单已经付款了，能全额退款吗？多久能到账？"
        sub_questions = split_complex_question(question)
        self.assertEqual(len(sub_questions), 2)
        self.assertIn("退款", sub_questions[0].normalized_text)
        self.assertIn("到账", sub_questions[1].normalized_text)


class AgentFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DummyAgentService()

    def test_agent_merges_multi_question_answers(self) -> None:
        response = self.agent.chat(
            ChatRequest(question='"请问你们家的商品支持7天无理由退换货吗？",\n"需要自己承担运费吗？"')
        )
        self.assertIn("问题1", response.answer)
        self.assertIn("问题2", response.answer)
        self.assertEqual(response.image_ids, ["img_a", "img_b"])
        self.assertEqual(response.confidence, 0.8)
        self.assertEqual(len(response.retrieval_debug["sub_questions"]), 2)

    def test_agent_keeps_single_question_shape(self) -> None:
        response = self.agent.chat(ChatRequest(question="洗碗机安装有什么要求？"))
        self.assertIn("问题1", response.answer)
        self.assertEqual(response.image_ids, ["img_c"])
        self.assertEqual(len(response.retrieval_debug["sub_questions"]), 1)


if __name__ == "__main__":
    unittest.main()
