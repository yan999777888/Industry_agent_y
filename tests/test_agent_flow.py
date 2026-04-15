from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.agent.question_splitter import split_complex_question
from industry_agent.agent.service import (
    AgentService,
    ChatRequest,
    _filter_evidence_for_query,
    _merge_retrieval_candidates,
)
from industry_agent.agent.context_manager import ContextManager
from industry_agent.agent.customer_service_policy import CustomerServicePolicy
from industry_agent.agent.image_understanding import ImageObservation, ImageUnderstandingResult, ImageUnderstander
from industry_agent.agent.question_router import QuestionRouter
from industry_agent.agent.response_formatter import format_customer_service_answer, format_manual_answer
from industry_agent.agent.session_store import InMemorySessionStore

ONE_BY_ONE_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/w8AAgMBgN8L1n4AAAAASUVORK5CYII="
)


class StubImageUnderstander:
    def analyze_images(self, images, *, question=""):
        has_image_input = bool(images)
        if not has_image_input:
            return ImageUnderstandingResult(has_image_input=False)
        return ImageUnderstandingResult(
            has_image_input=True,
            observations=[
                ImageObservation(
                    image_index=1,
                    format="PNG",
                    mime_type="image/png",
                    file_size=68,
                    width=1,
                    height=1,
                    summary="上传图片1：格式 PNG（image/png），尺寸 1x1，大小 68B",
                    visual_summary="图片里是设备指示灯区域，红灯闪烁",
                    source="stub",
                )
            ],
            combined_summary="上传图片1：格式 PNG（image/png），尺寸 1x1，大小 68B。视觉描述：图片里是设备指示灯区域，红灯闪烁",
            retrieval_hint="指示灯 红灯 闪烁",
            retrieval_terms=["指示灯", "红灯", "闪烁"],
            visual_features={
                "component_terms": ["指示灯"],
                "status_terms": ["红灯", "闪烁"],
                "issue_terms": [],
                "other_terms": [],
            },
            used_vision_model="stub-vision",
        )


class DummyAgentService(AgentService):
    """Test double that avoids real retrieval and LLM calls."""

    def __init__(self) -> None:
        self.image_index = {}
        self.model = "dummy"
        self.base_url = "http://dummy"
        self.http_client = None
        self.session_store = InMemorySessionStore()
        self.context_manager = ContextManager()
        self.question_router = QuestionRouter()
        self.customer_service_policy = CustomerServicePolicy()
        self.image_understander = ImageUnderstander(base_url=self.base_url, http_client=None, vision_model="")
        self.queries: list[str] = []

    def generate_response(self, query: str, history=None, image_input=None, dialog_summary=None, image_context=None, image_terms=None, image_features=None):  # type: ignore[override]
        self.queries.append(query)
        image_id = "img_b" if "运费" in query else "img_a" if "退换货" in query else "img_c"
        return {
            "answer": f"回答({query})",
            "image_ids": [image_id],
            "images": [{"image_id": image_id, "file_name": f"{image_id}.png", "path": f"Knowledge_base/插图/{image_id}.png", "exists": True}],
            "sources": ["测试产品"],
            "references": [{"chunk_id": "chunk_1", "title": "测试标题", "text_snippet": query[:50], "product_name": "测试产品", "score": "99"}],
            "confidence": 0.8,
            "retrieval_debug": {
                "query": query,
                "image_terms": image_terms or [],
                "image_features": image_features or {},
            },
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
        self.assertEqual(response.image_ids, [])
        self.assertIn("customer_service_policy", response.sources)
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )
        self.assertEqual(
            response.retrieval_debug["sub_results"][1]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )
        self.assertEqual(len(response.retrieval_debug["sub_questions"]), 2)

    def test_agent_keeps_single_question_shape(self) -> None:
        response = self.agent.chat(ChatRequest(question="洗碗机安装有什么要求？"))
        self.assertIn("问题1", response.answer)
        self.assertEqual(response.image_ids, ["img_c"])
        self.assertEqual(len(response.retrieval_debug["sub_questions"]), 1)

    def test_smalltalk_bypasses_retrieval(self) -> None:
        response = self.agent.chat(ChatRequest(question="你好"))

        self.assertIn("工业产品客服智能体", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertEqual(response.sources, [])
        self.assertEqual(response.retrieval_debug["route"], "smalltalk")
        self.assertEqual(self.agent.queries, [])

    def test_english_smalltalk_bypasses_retrieval(self) -> None:
        response = self.agent.chat(ChatRequest(question="hello"))

        self.assertIn("工业产品客服智能体", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertEqual(response.retrieval_debug["intent"], "greeting")
        self.assertEqual(self.agent.queries, [])

    def test_customer_service_route_bypasses_manual_retrieval(self) -> None:
        response = self.agent.chat(ChatRequest(question="我想退款，退款多久能到账？"))

        self.assertIn("订单号", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertEqual(self.agent.queries, [])
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )
        self.assertIn("customer_service_policy", response.sources)

    def test_mixed_question_uses_different_routes_per_subquestion(self) -> None:
        response = self.agent.chat(
            ChatRequest(question='"请问支持退款吗？",\n"电钻怎么充电？"')
        )

        self.assertIn("问题1", response.answer)
        self.assertEqual(len(response.retrieval_debug["sub_results"]), 2)
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )
        self.assertEqual(
            response.retrieval_debug["sub_results"][1]["retrieval_debug"]["route_decision"]["route"],
            "manual_rag",
        )
        self.assertEqual(len(self.agent.queries), 1)
        self.assertIn("电钻怎么充电？", self.agent.queries[0])

    def test_follow_up_inherits_product_context(self) -> None:
        session_id = "s_drill"
        self.agent.chat(ChatRequest(question="电钻的电池怎么充电？", session_id=session_id))
        response = self.agent.chat(ChatRequest(question="充电时有什么注意事项？", session_id=session_id))

        self.assertIn("电钻", self.agent.queries[-1])
        self.assertTrue(response.retrieval_debug["session"]["is_follow_up"])
        self.assertEqual(response.retrieval_debug["session"]["inherited_product"], "电钻")
        self.assertIn("电钻", response.retrieval_debug["sub_results"][0]["retrieval_debug"]["resolved_query"])

    def test_follow_up_resolves_pronoun_reference(self) -> None:
        session_id = "s_tracker"
        self.agent.chat(ChatRequest(question="我想更换健身追踪器的表带", session_id=session_id))
        response = self.agent.chat(ChatRequest(question="这个还有其他尺寸吗？", session_id=session_id))

        self.assertIn("健身追踪器", self.agent.queries[-1])
        self.assertEqual(response.retrieval_debug["session"]["inherited_product"], "健身追踪器")
        self.assertIn("健身追踪器", response.retrieval_debug["session"]["resolved_question"])

    def test_agent_uses_uploaded_image_hint(self) -> None:
        self.agent.image_understander = StubImageUnderstander()
        response = self.agent.chat(
            ChatRequest(
                question="这个指示灯是什么意思？",
                images=[ONE_BY_ONE_PNG_BASE64],
            )
        )

        self.assertIn("指示灯", self.agent.queries[-1])
        self.assertTrue(response.retrieval_debug["session"]["image_understanding"]["has_image_input"])
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["image_terms"],
            ["指示灯", "红灯", "闪烁"],
        )
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["image_features"]["component_terms"],
            ["指示灯"],
        )
        self.assertEqual(
            response.retrieval_debug["session"]["image_understanding"]["used_vision_model"],
            "stub-vision",
        )


class ImageUnderstandingTests(unittest.TestCase):
    def test_image_understander_parses_png_base64(self) -> None:
        understander = ImageUnderstander(base_url="http://dummy", http_client=None, vision_model="")
        result = understander.analyze_images([ONE_BY_ONE_PNG_BASE64], question="这是什么部件？")

        self.assertTrue(result.has_image_input)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.observations[0].format, "PNG")
        self.assertEqual(result.observations[0].width, 1)
        self.assertEqual(result.observations[0].height, 1)
        self.assertIn("上传图片1", result.combined_summary)
        self.assertEqual(result.retrieval_hint, "")
        self.assertEqual(result.retrieval_terms, [])

    def test_image_understander_accepts_data_url(self) -> None:
        understander = ImageUnderstander(base_url="http://dummy", http_client=None, vision_model="")
        result = understander.analyze_images([f"data:image/png;base64,{ONE_BY_ONE_PNG_BASE64}"])

        self.assertTrue(result.has_image_input)
        self.assertEqual(result.observations[0].mime_type, "image/png")

    def test_image_understander_filters_noisy_hint_terms(self) -> None:
        understander = ImageUnderstander(base_url="http://dummy", http_client=None, vision_model="")
        result = understander._build_retrieval_hint(  # type: ignore[attr-defined]
            question="这个指示灯是什么意思？",
            observations=[
                ImageObservation(
                    image_index=1,
                    format="PNG",
                    mime_type="image/png",
                    file_size=68,
                    visual_summary="该图像显示了一个电子设备，图片里有一个指示灯和一个按钮，红灯闪烁。",
                )
            ],
        )
        self.assertIn("红灯", result)
        self.assertIn("按钮", result)
        self.assertNotIn("图片", result)
        self.assertNotIn("设备", result)

    def test_image_understander_exposes_structured_retrieval_terms(self) -> None:
        understander = ImageUnderstander(base_url="http://dummy", http_client=None, vision_model="")
        features = understander._extract_visual_features(  # type: ignore[attr-defined]
            question="这个按钮怎么用？",
            observations=[
                ImageObservation(
                    image_index=1,
                    format="PNG",
                    mime_type="image/png",
                    file_size=68,
                    visual_summary="图片里有一个按钮、一个红灯指示灯和充电接口。",
                )
            ],
        )
        self.assertIn("指示灯", features["component_terms"])
        self.assertIn("红灯", features["status_terms"])
        self.assertIn("充电", features["status_terms"])


class ResponseFormatterTests(unittest.TestCase):
    def test_format_manual_answer_injects_image_section(self) -> None:
        answer = format_manual_answer("请先连接充电器，再观察指示灯状态。", image_ids=["drill0_17"])
        self.assertIn("结论：", answer)
        self.assertIn("相关图片：", answer)
        self.assertIn("drill0_17", answer)

    def test_format_customer_service_answer_is_plain_text(self) -> None:
        answer = format_customer_service_answer("  这类问题更适合按通用客服流程处理。  ")
        self.assertEqual(answer, "这类问题更适合按通用客服流程处理。")

    def test_format_manual_answer_splits_plain_text_into_distinct_sections(self) -> None:
        answer = format_manual_answer(
            "先连接充电器。观察红灯是否闪烁。请勿在潮湿环境中操作。",
            image_ids=[],
        )
        self.assertIn("结论：\n- 先连接充电器。", answer)
        self.assertIn("操作/说明：\n- 观察红灯是否闪烁。", answer)
        self.assertIn("注意事项：\n- 请勿在潮湿环境中操作。", answer)

    def test_format_manual_answer_normalizes_section_labels(self) -> None:
        answer = format_manual_answer(
            "结论：可正常充电\n操作：先插电源，再连接设备\n注意：请勿遮挡散热孔",
            image_ids=["img_1"],
        )
        self.assertIn("操作/说明：", answer)
        self.assertIn("注意事项：", answer)
        self.assertIn("相关图片：\n- img_1", answer)


class RetrievalFusionTests(unittest.TestCase):
    def test_merge_retrieval_candidates_prefers_multi_variant_hits(self) -> None:
        rows = _merge_retrieval_candidates(
            [
                (
                    "text_only",
                    [
                        {"chunk_id": "a", "title": "普通章节", "text": "普通说明", "product_name": "电钻", "image_ids": "[]", "_score": 20.0},
                        {"chunk_id": "b", "title": "按钮说明", "text": "按钮位置", "product_name": "电钻", "image_ids": "[]", "_score": 19.0},
                    ],
                ),
                (
                    "multimodal_fused",
                    [
                        {"chunk_id": "b", "title": "按钮说明", "text": "按钮位置", "product_name": "电钻", "image_ids": "[]", "_score": 19.0},
                    ],
                ),
            ]
        )
        self.assertEqual(rows[0]["chunk_id"], "b")
        self.assertEqual(rows[0]["_variant_hits"], 2)

    def test_filter_evidence_for_query_uses_image_overlap_and_diversity(self) -> None:
        filtered = _filter_evidence_for_query(
            [
                {
                    "chunk_id": "a",
                    "title": "设置时间",
                    "text": "按菜单键进入设置界面。",
                    "product_name": "可编程温控器",
                    "image_ids": "[]",
                    "_score": 18.0,
                    "_variant_hits": 1,
                },
                {
                    "chunk_id": "b",
                    "title": "充电指示灯说明",
                    "text": "红灯闪烁表示正在充电。",
                    "product_name": "可编程温控器",
                    "image_ids": "[\"img_1\"]",
                    "_score": 17.5,
                    "_variant_hits": 2,
                },
                {
                    "chunk_id": "c",
                    "title": "充电指示灯说明",
                    "text": "另一段重复标题内容。",
                    "product_name": "可编程温控器",
                    "image_ids": "[\"img_2\"]",
                    "_score": 17.0,
                    "_variant_hits": 1,
                },
            ],
            query="这个指示灯是什么意思？",
            image_terms=["指示灯", "红灯", "闪烁"],
        )
        self.assertEqual(filtered[0]["chunk_id"], "b")
        self.assertEqual(len(filtered), 2)
        self.assertNotEqual(filtered[0]["title"], filtered[1]["title"])

    def test_filter_evidence_for_query_prefers_component_status_alignment(self) -> None:
        filtered = _filter_evidence_for_query(
            [
                {
                    "chunk_id": "a",
                    "title": "指示灯状态说明",
                    "text": "红灯闪烁表示正在充电。",
                    "product_name": "电钻",
                    "image_ids": "[\"img_1\"]",
                    "_score": 16.0,
                    "_variant_hits": 1,
                },
                {
                    "chunk_id": "b",
                    "title": "电池安装步骤",
                    "text": "安装电池时请卡紧卡扣。",
                    "product_name": "电钻",
                    "image_ids": "[\"img_2\"]",
                    "_score": 17.2,
                    "_variant_hits": 1,
                },
            ],
            query="这个指示灯是什么意思？",
            image_terms=["指示灯", "红灯", "闪烁"],
            image_features={
                "component_terms": ["指示灯"],
                "status_terms": ["红灯", "闪烁"],
                "issue_terms": [],
                "other_terms": [],
            },
        )
        self.assertEqual(filtered[0]["chunk_id"], "a")


if __name__ == "__main__":
    unittest.main()
