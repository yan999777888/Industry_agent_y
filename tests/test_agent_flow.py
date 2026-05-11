from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.agent.question_splitter import split_complex_question
from industry_agent.agent.service import (
    AgentService,
    ChatRequest,
    _build_extractive_manual_answer,
    _filter_evidence_for_query,
    _merge_retrieval_candidates,
)
from industry_agent.agent.context_manager import ContextManager
from industry_agent.agent.customer_service_kb import CustomerServiceKnowledgeBase
from industry_agent.agent.customer_service_policy import CustomerServicePolicy
from industry_agent.agent.image_understanding import ImageObservation, ImageUnderstandingResult, ImageUnderstander
from industry_agent.agent.orchestrator import AgentOrchestrator
from industry_agent.agent.question_router import QuestionRouter
from industry_agent.agent.response_formatter import (
    format_customer_service_answer,
    format_manual_answer,
    format_multi_question_answer,
)
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


class RescueRetriever:
    def search(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "chunk_battery_1",
                "title": "安装电池",
                "text": "使用遥控器前，请先安装电池。1 取下电池盖。2 装入新电池，确保电池正、负极安装正确。",
                "product_name": "空调",
                "image_ids": "[]",
                "_score": 18.0,
            }
        ]

    def retrieval_status(self) -> dict[str, str]:
        return {"mode": "stub"}


class FakeFallbackLLMClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def chat(self, messages, **kwargs) -> str:
        return "根据现有资料无法准确回答此问题。"


class FakeEnglishMixedLLMClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def chat(self, messages, **kwargs) -> str:
        return "您需要在发动机停机且船只处于水平状态下检查机油液位，并通过油尺确认液位在最低和最高标记之间。"


class FakeLooseManualLLMClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def chat(self, messages, **kwargs) -> str:
        return "结论：请根据说明书操作。操作/说明：按提示继续。注意事项：注意安全。"


class EnglishRescueRetriever:
    def search(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "chunk_en_oil_1",
                "title": "To check the engine oil level",
                "text": (
                    "To check the engine oil level: With the engine stopped, place the boat in a precisely "
                    "level position and check the dipstick."
                ),
                "product_name": "汇总英文",
                "image_ids": "[]",
                "_score": 22.0,
            },
            {
                "chunk_id": "chunk_en_oil_2",
                "title": "Engine oil level check",
                "text": "Make sure that the engine oil level is between the minimum and maximum level marks on the dipstick.",
                "product_name": "汇总英文",
                "image_ids": "[]",
                "_score": 20.0,
            },
        ]

    def retrieval_status(self) -> dict[str, str]:
        return {"mode": "stub"}


class ImageGroundingRetriever:
    def search(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        return [
            {
                "chunk_id": "chunk_light",
                "title": "充电指示灯说明",
                "text": "红灯闪烁表示正在充电，绿灯常亮表示已充满。",
                "product_name": "电钻",
                "image_ids": "[\"drill0_17\"]",
                "_score": 22.0,
                "_variant_hits": 2,
            },
            {
                "chunk_id": "chunk_install",
                "title": "电池安装步骤",
                "text": "安装电池时请按压卡扣并确认电池锁定到位。",
                "product_name": "电钻",
                "image_ids": "[\"drill0_03\"]",
                "_score": 18.5,
                "_variant_hits": 1,
            },
            {
                "chunk_id": "chunk_clean",
                "title": "日常清洁",
                "text": "清洁前请先断电，并使用干布擦拭外壳。",
                "product_name": "电钻",
                "image_ids": "[]",
                "_score": 16.0,
                "_variant_hits": 1,
            },
        ]

    def retrieval_status(self) -> dict[str, str]:
        return {"mode": "stub"}


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
        self.customer_service_kb = CustomerServiceKnowledgeBase()
        self.image_understander = ImageUnderstander(base_url=self.base_url, http_client=None, vision_model="")
        self.queries: list[str] = []
        self.llm_messages: list[list[dict[str, str]]] = []

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
        self.llm_messages.append(messages)
        if messages and messages[0].get("role") == "system" and "【客服策略骨架】" in messages[0].get("content", ""):
            return "LLM 调用失败: dummy customer service fallback"
        return messages[0]["content"]

    def _merge_subquestion_answers(self, *, original_question, sub_questions, sub_results):  # type: ignore[override]
        if len(sub_results) == 1:
            return str(sub_results[0]["answer"])
        return format_multi_question_answer(
            [
                (sub_question.normalized_text, str(result["answer"]))
                for sub_question, result in zip(sub_questions, sub_results)
            ]
        )


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

    def test_split_outer_quoted_single_question_with_inner_quotes(self) -> None:
        question = '"How to set the camera model to "P" model?"'
        sub_questions = split_complex_question(question)
        self.assertEqual(len(sub_questions), 1)
        self.assertEqual(sub_questions[0].normalized_text, 'How to set the camera model to "P" model?')

    def test_split_mixed_clauses_by_question_mark_and_commas(self) -> None:
        question = "请问支持退款吗，电钻怎么充电？"
        sub_questions = split_complex_question(question)
        self.assertEqual(len(sub_questions), 2)
        self.assertIn("退款", sub_questions[0].normalized_text)
        self.assertIn("充电", sub_questions[1].normalized_text)


class AgentFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DummyAgentService()

    def test_agent_merges_multi_question_answers(self) -> None:
        response = self.agent.chat(
            ChatRequest(question='"请问你们家的商品支持7天无理由退换货吗？",\n"需要自己承担运费吗？"')
        )
        self.assertIn("结论：", response.answer)
        self.assertIn("处理步骤：", response.answer)
        self.assertNotIn("问题1：", response.answer)
        self.assertNotIn("问题2：", response.answer)
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
        self.assertNotIn("问题1：", response.answer)
        self.assertIn("回答(洗碗机安装有什么要求？)", response.answer)
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

        self.assertIn("原支付渠道", response.answer)
        self.assertIn("结论：", response.answer)
        self.assertIn("处理步骤：", response.answer)
        self.assertNotIn("建议一次性准备订单号", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertEqual(self.agent.queries, [])
        self.assertEqual(len(self.agent.llm_messages), 1)
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )
        self.assertIn("customer_service_policy", response.sources)
        self.assertIn("customer_service_kb", response.sources)
        self.assertGreaterEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["customer_service_kb"]["hit_count"],
            1,
        )
        self.assertTrue(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["customer_service_generation"]["used_llm"]
        )
        self.assertTrue(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["customer_service_generation"]["used_policy_fallback"]
        )

    def test_size_exchange_question_routes_to_customer_service(self) -> None:
        response = self.agent.chat(ChatRequest(question="我想把商品换成更大的尺寸，差价怎么处理？"))

        self.assertIn("customer_service_policy", response.sources)
        self.assertEqual(response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["route"], "customer_service")
        self.assertEqual(self.agent.queries, [])

    def test_customer_service_answer_is_direct_for_seven_day_refund(self) -> None:
        response = self.agent.chat(
            ChatRequest(question='"请问你们家的商品支持7天无理由退换货吗？","需要自己承担运费吗？"')
        )

        self.assertIn("通常可以申请退货", response.answer)
        self.assertIn("运费通常由买家承担", response.answer)
        self.assertEqual(self.agent.queries, [])

    def test_invoice_type_question_routes_to_customer_service_with_direct_answer(self) -> None:
        response = self.agent.chat(
            ChatRequest(question='"请问你们的商品能开发票吗？发票类型是什么？","多久能收到呢？"')
        )

        self.assertIn("电子发票、普通发票还是专用发票", response.answer)
        self.assertIn("customer_service_policy", response.sources)
        self.assertEqual(self.agent.queries, [])

    def test_human_damage_uses_paid_repair_wording(self) -> None:
        response = self.agent.chat(
            ChatRequest(question="如果是人为损坏的，能维修吗？维修费用怎么算？")
        )

        self.assertIn("不能按免费保修处理", response.answer)
        self.assertIn("付费检测或付费维修", response.answer)
        self.assertEqual(self.agent.queries, [])

    def test_packaging_damage_question_stays_on_customer_service_route(self) -> None:
        response = self.agent.chat(
            ChatRequest(question="我收到商品后发现外包装破损了，这会影响退换货吗？")
        )

        self.assertIn("先别急着定性为不能退换", response.answer)
        self.assertIn("customer_service_policy", response.sources)
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )

    def test_customer_service_follow_up_inherits_policy_context(self) -> None:
        session_id = "s_policy_followup"
        self.agent.chat(ChatRequest(question="我想退款，退款多久能到账？", session_id=session_id))
        query_count = len(self.agent.queries)
        response = self.agent.chat(ChatRequest(question="那需要准备什么材料？", session_id=session_id))

        self.assertIn("订单号", response.answer)
        self.assertIn("customer_service_policy", response.sources)
        self.assertEqual(len(self.agent.queries), query_count)
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["reason"],
            "inherit_customer_service_context",
        )

    def test_expanded_customer_service_topic_address_change(self) -> None:
        response = self.agent.chat(ChatRequest(question="我想修改收货地址，还来得及吗？"))

        self.assertIn("新地址", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertIn("customer_service_policy", response.sources)

    def test_platform_service_questions_route_to_customer_service(self) -> None:
        response = self.agent.chat(ChatRequest(question="请问你们支持以旧换新服务吗？"))

        self.assertIn("以旧换新", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertEqual(self.agent.queries, [])
        self.assertIn("customer_service_policy", response.sources)

    def test_trial_questions_route_to_customer_service(self) -> None:
        response = self.agent.chat(
            ChatRequest(question="试用期间商品出现故障，还能延长试用期限吗？故障商品可以更换吗？")
        )

        self.assertEqual(self.agent.queries, [])
        self.assertIn("customer_service_policy", response.sources)
        self.assertEqual(
            response.retrieval_debug["sub_results"][0]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )

    def test_manual_copy_request_routes_to_customer_service(self) -> None:
        response = self.agent.chat(ChatRequest(question="商品能提供纸质版说明书吗？电子版在哪里？"))

        self.assertIn("电子版", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertEqual(self.agent.queries, [])
        self.assertIn("customer_service_policy", response.sources)

    def test_mixed_question_uses_different_routes_per_subquestion(self) -> None:
        response = self.agent.chat(
            ChatRequest(question='"请问支持退款吗？",\n"电钻怎么充电？"')
        )

        self.assertIn("结论：", response.answer)
        self.assertNotIn("问题1：", response.answer)
        self.assertNotIn("问题2：", response.answer)
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

    def test_same_turn_customer_service_follow_up_inherits_previous_subquestion_context(self) -> None:
        response = self.agent.chat(
            ChatRequest(question='"请问你们的商品能开发票吗？发票类型是什么？",\n"多久能收到呢？"')
        )

        self.assertEqual(response.image_ids, [])
        self.assertEqual(len(self.agent.queries), 0)
        self.assertEqual(
            response.retrieval_debug["sub_results"][1]["retrieval_debug"]["route_decision"]["route"],
            "customer_service",
        )
        self.assertEqual(
            response.retrieval_debug["sub_results"][1]["retrieval_debug"]["route_decision"]["reason"],
            "inherit_current_turn_customer_service_context",
        )

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

    def test_session_reset_clears_inherited_context(self) -> None:
        session_id = "s_reset"
        self.agent.chat(ChatRequest(question="电钻的电池怎么充电？", session_id=session_id))
        reset_response = self.agent.chat(ChatRequest(question="清空上下文", session_id=session_id))
        response = self.agent.chat(ChatRequest(question="充电时有什么注意事项？", session_id=session_id))

        self.assertIn("已清空", reset_response.answer)
        self.assertEqual(reset_response.retrieval_debug["route"], "session_control")
        self.assertNotIn("电钻", self.agent.queries[-1])
        self.assertFalse(response.retrieval_debug["session"]["is_follow_up"])

    def test_unresolved_topic_switch_asks_for_product_name(self) -> None:
        session_id = "s_switch_unknown"
        self.agent.chat(ChatRequest(question="电钻的电池怎么充电？", session_id=session_id))
        query_count = len(self.agent.queries)
        response = self.agent.chat(ChatRequest(question="换个产品怎么安装？", session_id=session_id))

        self.assertIn("请补充新的产品名称或型号", response.answer)
        self.assertEqual(response.retrieval_debug["route"], "clarification")
        self.assertEqual(len(self.agent.queries), query_count)

    def test_explicit_topic_switch_updates_session_product(self) -> None:
        session_id = "s_switch_explicit"
        self.agent.chat(ChatRequest(question="电钻的电池怎么充电？", session_id=session_id))
        response = self.agent.chat(ChatRequest(question="换个产品，洗碗机安装有什么要求？", session_id=session_id))
        session = self.agent.session_store.get(session_id)

        self.assertIn("洗碗机", self.agent.queries[-1])
        self.assertTrue(response.retrieval_debug["session"]["topic_switched"])
        self.assertIsNotNone(session)
        self.assertEqual(session.current_product, "洗碗机")

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


class QuestionRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = QuestionRouter()

    def test_router_prefers_manual_for_product_operation_question(self) -> None:
        decision = self.router.route("健身追踪器的表带怎么更换？")

        self.assertEqual(decision.route, "manual_rag")
        self.assertEqual(decision.reason, "prefer_manual_rag_with_manual_signal")

    def test_router_prefers_manual_for_howto_even_with_ambiguous_service_term(self) -> None:
        decision = self.router.route("这个配件怎么安装？")

        self.assertEqual(decision.route, "manual_rag")
        self.assertGreaterEqual(decision.manual_score, decision.service_score)

    def test_router_keeps_size_exchange_on_customer_service_route(self) -> None:
        decision = self.router.route("我想把商品换成更大的尺寸，差价怎么处理？")

        self.assertEqual(decision.route, "customer_service")


class OrchestratorSmokeTests(unittest.TestCase):
    def test_orchestrator_handles_smalltalk_without_retrieval(self) -> None:
        orchestrator = AgentOrchestrator()

        response = orchestrator.chat(ChatRequest(question="hello"))

        self.assertIn("工业产品客服智能体", response.answer)
        self.assertEqual(response.image_ids, [])
        self.assertEqual(response.retrieval_debug["route"], "smalltalk")


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
        answer = format_customer_service_answer("  结论：\n  - 这类问题更适合按通用客服流程处理。  ")
        self.assertEqual(answer, "结论：\n- 这类问题更适合按通用客服流程处理。")

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

    def test_format_manual_answer_compact_mode_flattens_sections(self) -> None:
        answer = format_manual_answer(
            "结论：可正常充电\n操作：先插电源，再连接设备\n注意：请勿遮挡散热孔",
            image_ids=[],
            compact=True,
        )

        self.assertNotIn("结论：", answer)
        self.assertNotIn("操作/说明：", answer)
        self.assertIn("可正常充电。", answer)
        self.assertIn("先插电源，再连接设备。", answer)

    def test_format_multi_question_answer_merges_without_rewriting(self) -> None:
        answer = format_multi_question_answer(
            [
                ("电钻怎么充电？", "结论：\n- 使用充电器。"),
                ("有哪些注意事项？", "回答：注意事项：\n- 请勿潮湿操作。"),
            ]
        )
        self.assertIn("结论：", answer)
        self.assertIn("结论：\n- 使用充电器。", answer)
        self.assertIn("注意事项：", answer)
        self.assertNotIn("问题1：", answer)
        self.assertNotIn("问题2：", answer)
        self.assertNotIn("回答：", answer)

    def test_format_multi_question_answer_merges_related_images_without_label_noise(self) -> None:
        answer = format_multi_question_answer(
            [
                ("怎么安装电池？", "结论：\n- 取下电池盖。\n\n相关图片：\n- Manual01_2"),
                ("还要注意什么？", "注意事项：\n- 确认正负极方向。\n\n相关图片：\n- Manual01_3"),
            ]
        )

        self.assertIn("相关图片：", answer)
        self.assertIn("Manual01_2、Manual01_3", answer)
        self.assertNotIn("Manual01_2。", answer)

    def test_build_extractive_manual_answer_uses_evidence_for_english_query(self) -> None:
        answer = _build_extractive_manual_answer(
            query="How to find the approval label of emission control certificate of the boat?",
            evidence_chunks=[
                {
                    "chunk_id": "chunk_en_1",
                    "title": "Approval label of emission control certificate",
                    "text": "These labels are attached to each engine unit and to the inside of the engine compartment.",
                    "product_name": "汇总英文",
                    "image_ids": "[]",
                }
            ],
            image_ids=[],
        )
        self.assertIn("Approval label of emission control certificate", answer)
        self.assertNotIn("根据现有资料无法准确回答此问题", answer)

    def test_build_extractive_manual_answer_deduplicates_repeated_title_and_text(self) -> None:
        answer = _build_extractive_manual_answer(
            query="How to find the approval label of emission control certificate of the boat?",
            evidence_chunks=[
                {
                    "chunk_id": "chunk_en_1",
                    "title": "Approval label of emission control certificate",
                    "text": "These labels are attached to each engine unit and to the inside of the engine compartment.",
                    "product_name": "汇总英文",
                    "image_ids": "[]",
                },
                {
                    "chunk_id": "chunk_en_2",
                    "title": "Approval label of Emission control certificate",
                    "text": "This label is attached to the electrical box and the exhaust side of the crankcase.",
                    "product_name": "汇总英文",
                    "image_ids": "[]",
                },
            ],
            image_ids=[],
        )
        self.assertEqual(answer.count("Approval label of emission control certificate"), 1)
        self.assertIn("engine unit", answer)

    def test_agent_service_uses_extractive_rescue_for_chinese_fallback(self) -> None:
        with patch("industry_agent.agent.service.LLMClient", FakeFallbackLLMClient):
            agent = AgentService(
                retriever=RescueRetriever(),
                llm_backend="openai_compatible",
                base_url="https://api.example.com/v1",
                model="demo-model",
            )

        result = agent.generate_response(query="如何给空调遥控器安装电池？")

        self.assertIn("安装电池", result["answer"])
        self.assertIn("取下电池盖", result["answer"])
        self.assertNotIn("根据现有资料无法准确回答此问题", result["answer"])

    def test_agent_service_prefers_extractive_answer_for_english_query_when_llm_returns_chinese(self) -> None:
        with patch("industry_agent.agent.service.LLMClient", FakeEnglishMixedLLMClient):
            agent = AgentService(
                retriever=EnglishRescueRetriever(),
                llm_backend="openai_compatible",
                base_url="https://api.example.com/v1",
                model="demo-model",
            )

        result = agent.generate_response(
            query="When I am sailing, how do I check the engine oil level to ensure continued sailing?"
        )

        self.assertIn("engine oil level", result["answer"])
        self.assertNotRegex(result["answer"], r"[\u4e00-\u9fff]")

    def test_agent_service_prefers_extractive_answer_for_manual_procedure_queries(self) -> None:
        with patch("industry_agent.agent.service.LLMClient", FakeLooseManualLLMClient):
            agent = AgentService(
                retriever=RescueRetriever(),
                llm_backend="openai_compatible",
                base_url="https://api.example.com/v1",
                model="demo-model",
            )

        result = agent.generate_response(query="如何给空调遥控器安装电池？")

        self.assertIn("取下电池盖", result["answer"])
        self.assertIn("装入新电池", result["answer"])
        self.assertNotIn("请根据说明书操作", result["answer"])

    def test_agent_service_grounds_manual_images_to_selected_evidence(self) -> None:
        with patch("industry_agent.agent.service.LLMClient", FakeFallbackLLMClient):
            agent = AgentService(
                retriever=ImageGroundingRetriever(),
                llm_backend="openai_compatible",
                base_url="https://api.example.com/v1",
                model="demo-model",
            )

        result = agent.generate_response(query="这个指示灯是什么意思？")

        self.assertIn("红灯闪烁表示正在充电", result["answer"])
        self.assertEqual(result["image_ids"], ["drill0_17"])
        self.assertEqual(result["retrieval_debug"]["candidate_image_ids"], ["drill0_17", "drill0_03"])
        self.assertEqual(result["retrieval_debug"]["grounded_image_ids"], ["drill0_17"])
        self.assertNotIn("相关配图如下", result["answer"])


class CustomerServicePolicyTests(unittest.TestCase):
    def test_customer_service_kb_retrieves_specific_invoice_entry(self) -> None:
        kb = CustomerServiceKnowledgeBase()
        hits = kb.search("请问你们的商品能开发票吗？发票类型是什么？多久能收到呢？")

        self.assertTrue(hits)
        self.assertEqual(hits[0]["topic"], "invoice")
        self.assertTrue(any("发票类型" in hit["title"] or "发票处理" in hit["title"] for hit in hits))

    def test_customer_service_kb_prefers_data_file_entry_for_pickup_pending(self) -> None:
        kb = CustomerServiceKnowledgeBase()
        hits = kb.search("物流一直显示待揽收，是什么原因？")

        self.assertTrue(hits)
        self.assertEqual(hits[0]["entry_id"], "shipping::pickup_pending")
        self.assertEqual(hits[0]["source_type"], "data_file")
        self.assertIn("24 小时", hits[0]["content"])

    def test_customer_service_kb_keeps_policy_projection_fallback_for_unlisted_topic(self) -> None:
        kb = CustomerServiceKnowledgeBase()
        hits = kb.search("刚买完就降价了，可以申请价保吗？")

        self.assertTrue(hits)
        self.assertEqual(hits[0]["topic"], "price_protection")
        self.assertEqual(hits[0]["source_type"], "data_file")

    def test_customer_service_kb_retrieves_repeat_charge_data_entry(self) -> None:
        kb = CustomerServiceKnowledgeBase()
        hits = kb.search("我好像被重复扣款了，扣了两次怎么办？")

        self.assertTrue(hits)
        self.assertEqual(hits[0]["entry_id"], "payment_issue::repeat_charge")
        self.assertEqual(hits[0]["source_type"], "data_file")
        self.assertIn("支付流水", hits[0]["content"])

    def test_customer_service_kb_retrieves_shipping_signed_missing_data_entry(self) -> None:
        kb = CustomerServiceKnowledgeBase()
        hits = kb.search("物流显示已签收但我没收到，这种情况应该怎么处理？")

        self.assertTrue(hits)
        self.assertTrue(any(hit["entry_id"] == "shipping::signed_missing" for hit in hits))
        data_hit = next(hit for hit in hits if hit["entry_id"] == "shipping::signed_missing")
        self.assertEqual(data_hit["source_type"], "data_file")
        self.assertIn("承运商", data_hit["content"])

    def test_policy_uses_context_topics_for_short_follow_up(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("那需要准备什么材料？", context_topics=["refund_exchange"])

        self.assertIn("订单号", response.answer)
        self.assertIn("refund_exchange", response.matched_topics)

    def test_policy_uses_context_topics_for_short_timeline_follow_up(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("那多久能收到呢？", context_topics=["invoice"])

        self.assertIn("开票", response.answer)
        self.assertIn("时效/费用", response.answer)
        self.assertIn("invoice", response.matched_topics)

    def test_policy_covers_price_protection(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("刚买完就降价了，可以申请价保吗？")

        self.assertIn("价保", response.answer)
        self.assertIn("price_protection", response.matched_topics)

    def test_policy_answers_materials_intent_with_topic_specific_materials(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("修改收货地址需要准备什么材料？")

        self.assertIn("订单号", response.answer)
        self.assertIn("新地址", response.answer)
        self.assertIn("address_change", response.matched_topics)

    def test_policy_answers_timeline_intent_with_timing_hint(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("催发货一般多久能处理？")

        self.assertIn("时效/费用", response.answer)
        self.assertIn("delivery_delay", response.matched_topics)

    def test_policy_answers_fee_intent_with_fee_hint(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("上门安装需要收费吗？")

        self.assertIn("时效/费用", response.answer)
        self.assertIn("收费", response.answer)
        self.assertIn("installation_service", response.matched_topics)

    def test_policy_answers_eligibility_intent_with_conditions(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("刚买完就降价了，申请价保需要满足什么条件？")

        self.assertIn("取决于", response.answer)
        self.assertIn("price_protection", response.matched_topics)

    def test_policy_answers_process_intent_with_steps(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("支付失败但是扣款了，应该怎么申请核查？")

        self.assertIn("建议", response.answer)
        self.assertIn("payment_issue", response.matched_topics)

    def test_policy_answers_contact_intent_with_contact_hint(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("发票抬头填错了，这种情况应该联系谁处理？")

        self.assertIn("人工客服", response.answer)
        self.assertIn("invoice", response.matched_topics)

    def test_policy_refines_refund_reason_for_quality_issue(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("这个商品有质量问题，我想退款，需要我承担运费吗？")

        self.assertIn("质量问题", response.answer)
        self.assertIn("商家或平台承担", response.answer)
        self.assertIn("refund_exchange", response.matched_topics)
        self.assertNotIn("退换货和退款通常要结合订单状态", response.answer)

    def test_policy_refines_refund_arrival_by_payment_channel(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("退款多久到账？信用卡会原路返回吗？")

        self.assertIn("原路退回", response.answer)
        self.assertIn("发卡行", response.answer)
        self.assertIn(response.matched_topics[0], {"refund_exchange", "order_change"})

    def test_policy_refines_size_exchange_and_difference(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("我想把商品换成更大的尺寸，差价怎么处理？")

        self.assertIn("补差价", response.answer)
        self.assertIn("尺寸", response.answer)
        self.assertIn("refund_exchange", response.matched_topics)

    def test_policy_refines_shipping_status_for_signed_but_missing(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("物流显示已签收但我没收到，这种情况应该怎么处理？")

        self.assertIn("已签收但实际未收到", response.answer)
        self.assertIn("承运商", response.answer)
        self.assertIn("shipping", response.matched_topics)

    def test_policy_refines_village_or_overseas_shipping(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("你们的商品能送到乡镇吗？需要额外加运费吗？多久能到？")

        self.assertIn("乡镇", response.answer)
        self.assertIn("运费", response.answer)
        self.assertIn("shipping", response.matched_topics)

    def test_policy_handles_trial_extension_and_fault(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("试用期间商品出现故障，还能延长试用期限吗？故障商品可以更换吗？")

        self.assertIn("试用", response.answer)
        self.assertIn("故障", response.answer)
        self.assertIn("活动规则", response.answer)
        self.assertIn("platform_service", response.matched_topics)

    def test_policy_refines_after_sales_for_human_damage(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("设备进水了，还能走保修吗？")

        self.assertIn("人为损坏", response.answer)
        self.assertIn("付费维修", response.answer)
        self.assertIn("after_sales", response.matched_topics)

    def test_policy_refines_repair_delay_or_repeat_failure(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("我送修半个月了还没修好，而且返修后又是同样的故障，怎么办？")

        self.assertIn("复检", response.answer)
        self.assertIn("重新返修", response.answer)
        self.assertIn("after_sales", response.matched_topics)
        self.assertNotIn("售后、维修和保修问题通常需要确认", response.answer)

    def test_policy_refines_refund_rejected_status(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("我的退款申请被驳回了，我现在该怎么办？")

        self.assertIn("驳回原因", response.answer)
        self.assertIn("refund_exchange", response.matched_topics)

    def test_policy_refines_invoice_reissue_status(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("发票抬头填错了，而且已经开出来了，还能重开吗？")

        self.assertIn("重开", response.answer)
        self.assertIn("invoice", response.matched_topics)

    def test_policy_refines_invoice_after_issued(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("发票已经开出来了，还能改抬头吗？")

        self.assertIn("红冲", response.answer)
        self.assertIn("重开", response.answer)
        self.assertIn("invoice", response.matched_topics)

    def test_policy_avoids_generic_support_tail_for_specific_invoice_question(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("请问你们的商品能开发票吗？发票类型是什么？多久能收到呢？")

        self.assertIn("电子发票、普通发票还是专用发票", response.answer)
        self.assertIn("开票", response.answer)
        self.assertNotIn("如果你这边已经有订单号和开票截图", response.answer)
        self.assertNotIn("补充说明：", response.answer)

    def test_policy_can_cover_parallel_customer_service_topics(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("退款多久到账，以及发票抬头写错了还能重开吗？")

        self.assertIn("原支付渠道", response.answer)
        self.assertIn("重开", response.answer)
        self.assertIn("refund_exchange", response.matched_topics)
        self.assertIn("invoice", response.matched_topics)

    def test_policy_refines_installation_reschedule_status(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("上门安装已经约好了，但是师傅没来，可以改约吗？")

        self.assertIn("改约", response.answer)
        self.assertIn("installation_service", response.matched_topics)

    def test_policy_refines_address_after_shipment(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("订单已经发货了，还能改地址吗？")

        self.assertIn("拦截", response.answer)
        self.assertIn("改派", response.answer)
        self.assertIn("address_change", response.matched_topics)

    def test_policy_refines_shipping_lost_or_returned(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("物流显示退回了，但我没有申请退货，怎么办？")

        self.assertIn("物流异常", response.answer)
        self.assertIn("退回", response.answer)
        self.assertIn("shipping", response.matched_topics)

    def test_policy_refines_accessory_rejected_status(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("我申请补寄配件被驳回了，还能重新提交吗？")

        self.assertIn("驳回", response.answer)
        self.assertIn("accessory_request", response.matched_topics)

    def test_policy_refines_after_sales_rejected_or_disputed(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("我的售后申请被驳回了，我觉得判定不合理，能复核吗？")

        self.assertIn("驳回原因", response.answer)
        self.assertIn("人工复核", response.answer)
        self.assertIn("after_sales", response.matched_topics)

    def test_policy_refines_repeat_charge(self) -> None:
        policy = CustomerServicePolicy()
        response = policy.answer("我好像被重复扣款了，扣了两次怎么办？")

        self.assertIn("重复订单", response.answer)
        self.assertIn("成功支付", response.answer)
        self.assertIn("payment_issue", response.matched_topics)


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

    def test_filter_evidence_for_query_uses_fusion_score(self) -> None:
        filtered = _filter_evidence_for_query(
            [
                {
                    "chunk_id": "a",
                    "title": "普通充电说明",
                    "text": "充电前请确认电池状态。",
                    "product_name": "电钻",
                    "image_ids": "[]",
                    "_score": 18.0,
                    "_fusion_score": 18.0,
                    "_variant_hits": 1,
                },
                {
                    "chunk_id": "b",
                    "title": "电池充电注意事项",
                    "text": "充电时请使用指定充电器。",
                    "product_name": "电钻",
                    "image_ids": "[]",
                    "_score": 16.0,
                    "_fusion_score": 22.0,
                    "_variant_hits": 2,
                },
            ],
            query="电钻充电时有什么注意事项？",
        )

        self.assertEqual(filtered[0]["chunk_id"], "b")

    def test_filter_evidence_for_query_can_keep_high_overlap_cross_product_evidence(self) -> None:
        filtered = _filter_evidence_for_query(
            [
                {
                    "chunk_id": "a",
                    "title": "基本说明",
                    "text": "可用于日常操作。",
                    "product_name": "产品A",
                    "image_ids": "[]",
                    "_score": 18.0,
                    "_variant_hits": 1,
                },
                {
                    "chunk_id": "b",
                    "title": "默认密码说明",
                    "text": "默认密码是 1234，首次登录后建议修改。",
                    "product_name": "产品B",
                    "image_ids": "[]",
                    "_score": 17.6,
                    "_variant_hits": 2,
                },
            ],
            query="默认密码是多少？",
        )

        self.assertEqual(filtered[0]["chunk_id"], "b")
        self.assertIn("产品B", {item["product_name"] for item in filtered})

    def test_filter_evidence_for_query_keeps_same_product_lock_when_query_is_explicit(self) -> None:
        filtered = _filter_evidence_for_query(
            [
                {
                    "chunk_id": "a",
                    "title": "电钻默认密码说明",
                    "text": "电钻默认密码是 1234。",
                    "product_name": "电钻",
                    "image_ids": "[]",
                    "_score": 18.2,
                    "_variant_hits": 1,
                },
                {
                    "chunk_id": "b",
                    "title": "温控器默认密码说明",
                    "text": "温控器默认密码是 9999。",
                    "product_name": "可编程温控器",
                    "image_ids": "[]",
                    "_score": 17.9,
                    "_variant_hits": 2,
                },
            ],
            query="电钻的默认密码是多少？",
        )

        self.assertEqual([item["chunk_id"] for item in filtered], ["a"])


if __name__ == "__main__":
    unittest.main()
