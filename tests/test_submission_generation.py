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

    def test_normalize_submission_answer_merges_multi_question_content(self) -> None:
        raw = (
            "问题1：发票问题通常需要确认订单号、开票类型、抬头信息以及当前开票状态。\n"
            "问题2：根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。"
        )
        normalized = normalize_submission_answer(
            raw,
            question="发票类型是什么？多久能收到？",
            sources=["customer_service_policy"],
        )
        self.assertIn("通常支持按订单开票", normalized)
        self.assertIn("先确认订单号", normalized)
        self.assertNotIn("问题1：", normalized)
        self.assertNotIn("问题2：", normalized)
        self.assertNotIn("根据现有资料无法回答此问题", normalized)
        self.assertNotIn("您好，相关情况需要结合订单信息", normalized)

    def test_normalize_submission_answer_does_not_submit_question_echo_only(self) -> None:
        normalized = normalize_submission_answer(
            "问题1：Can this eReader record voice? If so, how do I operate this feature?\n"
            "问题2：根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。",
            question='"Can this eReader record voice? If so, how do I operate this feature?"',
            sources=[],
        )
        self.assertIn("manual evidence is not sufficient", normalized)
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
        self.assertEqual(normalized.count("先确认订单号"), 1)
        self.assertLessEqual(normalized.count("建议先准备订单号"), 1)

    def test_normalize_submission_answer_merges_multi_question_markers(self) -> None:
        normalized = normalize_submission_answer(
            "问题1：结论：可以开发票。\n处理步骤：建议先准备订单号和开票信息。\n"
            "问题2：结论：到账时间通常取决于开票系统和平台处理进度。",
            question='"请问你们的商品能开发票吗？发票类型是什么？","多久能收到呢？"',
            sources=["customer_service_policy"],
        )

        self.assertNotIn("问题1：", normalized)
        self.assertNotIn("问题2：", normalized)
        self.assertIn("可以开发票", normalized)
        self.assertNotIn("建议先准备订单号", normalized)

    def test_normalize_submission_answer_rewrites_customer_service_to_more_direct_style(self) -> None:
        normalized = normalize_submission_answer(
            "发票问题通常需要确认订单号、开票类型、抬头信息以及当前开票状态。"
            "建议先准备订单号、开票类型、发票抬头、税号和接收邮箱等信息。"
            "开票和补开发票的处理时效通常与订单状态、开票系统和平台规则有关，实际时间以平台处理进度为准。",
            question="请问你们的商品能开发票吗？发票类型是什么？多久能收到呢？",
            sources=["customer_service_policy"],
        )

        self.assertIn("通常支持按订单开票", normalized)
        self.assertIn("订单号", normalized)
        self.assertNotIn("通常需要确认", normalized)
        self.assertNotIn("通常与订单状态", normalized)

    def test_normalize_submission_answer_preserves_direct_customer_service_answer(self) -> None:
        normalized = normalize_submission_answer(
            "物流一直显示待揽收，一般表示包裹已打包完成，正在等待快递员揽件；通常会在24小时内更新首条物流轨迹。",
            question="物流一直显示待揽收，是什么原因？",
            sources=["customer_service_policy"],
        )

        self.assertIn("24小时内更新首条物流轨迹", normalized)
        self.assertNotIn("相关情况需要结合订单信息", normalized)
        self.assertNotIn("建议提供订单号", normalized)

    def test_normalize_submission_answer_still_rewrites_generic_customer_service_answer(self) -> None:
        normalized = normalize_submission_answer(
            "相关情况需要结合订单信息、商品状态和平台规则确认。建议提供订单号、商品名称、问题照片或聊天记录，以便继续判断处理方式。",
            question="我想退款，退款多久到账？",
            sources=["customer_service_policy"],
        )

        self.assertNotIn("相关情况需要结合订单信息", normalized)
        self.assertIn("原路退回", normalized)

    def test_normalize_submission_answer_does_not_inject_generic_refund_prefix(self) -> None:
        normalized = normalize_submission_answer(
            "退款一般原路退回原支付账户；如果是信用卡支付，到账时间通常要看发卡行处理进度。"
            "建议先确认订单是否已取消成功或售后审核通过，再查看退款状态。",
            question="我想了解一下退款多久到账，信用卡会原路返回吗？",
            sources=["customer_service_policy"],
        )

        self.assertNotIn("退货、换货或退款通常要先看订单状态", normalized)
        self.assertIn("原路退回", normalized)

    def test_normalize_submission_answer_strips_generic_invoice_support_tail(self) -> None:
        normalized = normalize_submission_answer(
            "通常支持按订单开票；具体支持电子发票、普通发票还是专用发票，要以订单开票入口为准。"
            "如果你这边已经有订单号和开票截图，建议一并提供，通常可以减少重复核对。",
            question="请问你们的商品能开发票吗？发票类型是什么？多久能收到呢？",
            sources=["customer_service_policy"],
        )

        self.assertIn("通常支持按订单开票", normalized)
        self.assertNotIn("开票截图", normalized)
        self.assertNotIn("减少重复核对", normalized)

    def test_normalize_submission_answer_emits_pic_markers_and_image_ids_for_manual_answers(self) -> None:
        normalized = normalize_submission_answer(
            "安装电池：取下电池盖，装入新电池，确认正负极正确。",
            question="如何给空调遥控器安装电池？",
            sources=["空调"],
            image_ids=["Manual01_2", "Manual01_3"],
        )

        self.assertIn("<PIC>", normalized)
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
        self.assertIn('";["Manual01_2"]', normalized)
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
        self.assertIn("<PIC>", rows[0]["ret"])
        self.assertIn('";["Manual01_2"]', rows[0]["ret"])

    def test_normalize_submission_answer_strips_html_and_duplicate_pic_noise(self) -> None:
        normalized = normalize_submission_answer(
            '"<IMG src=\\".png\\" alt=\\"设置设备锁界面\\" /> <text>设置设备锁</text> '
            '<PIC></PIC> <PIC> <PIC> <PIC> <PIC>。";["Manual16_1", "Manual16_2"]',
            question="我想给健身追踪器设置锁屏，该如何实现？",
            sources=["健身追踪器"],
            image_ids=["Manual16_1", "Manual16_2", "Manual16_3", "Manual16_4"],
        )

        self.assertNotIn("<IMG", normalized)
        self.assertNotIn("<text>", normalized)
        self.assertNotIn("</PIC>", normalized)
        self.assertIn('";["Manual16_1", "Manual16_2", "Manual16_3"]', normalized)
        self.assertEqual(normalized.count("<PIC>"), 3)

    def test_normalize_submission_answer_falls_back_when_only_image_tags_remain(self) -> None:
        normalized = normalize_submission_answer(
            '<IMG src="Manual16_31.png" alt="设置设备锁界面" />' * 20,
            question="我想给健身追踪器设置锁屏，该如何实现？",
            sources=["健身追踪器"],
            image_ids=["Manual16_31"],
            references=[
                {
                    "title": "设置设备锁",
                    "text_snippet": "设置设备锁 在健身追踪器应用中进入设备锁，按屏幕提示设置 PIN 码。",
                }
            ],
        )

        self.assertTrue(normalized.strip())
        self.assertIn("设置设备锁", normalized)

    def test_normalize_submission_answer_compresses_overlong_answer(self) -> None:
        raw = "。".join(f"第{i}步：安装时请确认排水口连接牢固并检查漏水情况" for i in range(80))
        normalized = normalize_submission_answer(
            raw,
            question="首次使用时，如何将洗碗机连接到排水口？",
            sources=["洗碗机"],
        )

        self.assertLessEqual(len(normalized), 560)

    def test_normalize_submission_answer_removes_leading_question_echo(self) -> None:
        normalized = normalize_submission_answer(
            "问题1：你们的售后维修服务范围是什么？\n"
            "售后、维修和保修问题通常需要确认商品型号、故障现象、购买时间和保修凭证。",
            question='"我想咨询一下，你们的售后维修服务范围是什么？"',
            sources=["customer_service_policy"],
        )

        self.assertNotIn("售后维修服务范围是什么", normalized)
        self.assertIn("先确认商品型号", normalized)
        self.assertNotIn("售后、维修和保修问题通常需要确认", normalized)

    def test_normalize_submission_answer_strips_loose_fallback_sentences(self) -> None:
        normalized = normalize_submission_answer(
            "根据现有资料，无法回答“如何调节化油器”的具体操作步骤。"
            "参考5 产品：吹风机 | 章节：吹风机部件。请以实际产品型号和说明书原文为准。"
            "吹风机使用前请确认进风口没有堵塞。",
            question="使用吹风机时，如何调节化油器？",
            sources=["吹风机"],
        )

        self.assertNotIn("无法回答", normalized)
        self.assertNotIn("参考5 产品", normalized)
        self.assertIn("吹风机使用前", normalized)

    def test_normalize_submission_answer_uses_english_fallback_for_english_question(self) -> None:
        normalized = normalize_submission_answer(
            "The provided reference materials do not contain information regarding this feature. "
            "They only cover a different product.",
            question="How the ship steers?",
            sources=[],
        )

        self.assertIn("manual evidence is not sufficient", normalized)
        self.assertNotIn("请补充", normalized)

    def test_normalize_submission_answer_strips_placeholder_and_markdown_noise(self) -> None:
        normalized = normalize_submission_answer(
            "### 问题1：与上一问处理思路一致，可按相同材料和流程继续处理。\n"
            "### 问题2：模型未返回有效回答。\n"
            "- 实际建议：先确认设备型号，再按说明书步骤操作。<PIC>",
            question="这个设备怎么操作？",
            sources=["设备说明书"],
        )

        self.assertNotIn("问题1", normalized)
        self.assertNotIn("问题2", normalized)
        self.assertNotIn("与上一问处理思路一致", normalized)
        self.assertNotIn("模型未返回有效回答", normalized)
        self.assertNotIn("###", normalized)
        self.assertNotIn("<PIC>", normalized)
        self.assertIn("先确认设备型号", normalized)

    def test_normalize_submission_answer_keeps_customer_service_answers_without_image_suffix(self) -> None:
        normalized = normalize_submission_answer(
            "通常支持按订单开票；具体支持电子发票、普通发票还是专用发票，要以订单开票入口为准。",
            question="请问你们的商品能开发票吗？发票类型是什么？",
            sources=["customer_service_policy"],
            image_ids=[],
        )

        self.assertNotIn("<PIC>", normalized)
        self.assertNotIn('";[', normalized)

    def test_normalize_submission_answer_strips_mixed_language_refusal_tail(self) -> None:
        normalized = normalize_submission_answer(
            '根据现有资料无法回答如何设置相机型号。建议您检查问题表述是否完整，或提供更多关于您想设置的具体功能细节。'
            '无 Based on the available references, I cannot provide specific steps to set the camera to "P" mode。'
            'The references only mention "P: Program AE" without detailing how to set it.',
            question='How to set the camera model to "P" model?',
            sources=["汇总英文"],
            references=[
                {
                    "title": "P: Program AE",
                    "text_snippet": 'P: Program AE is one of the shooting modes available on the camera mode dial.',
                }
            ],
        )

        self.assertNotIn("Based on the available references", normalized)
        self.assertNotIn("The references only mention", normalized)
        self.assertIn("Program AE", normalized)

    def test_normalize_submission_answer_rewrites_chinese_english_mixed_answer_from_references(self) -> None:
        normalized = normalize_submission_answer(
            "您需要在发动机停机且船只处于水平状态下检查机油液位，并通过油尺确认液位在最低和最高标记之间。",
            question="When I am sailing, how do I check the engine oil level to ensure continued sailing?",
            sources=["汇总英文"],
            references=[
                {
                    "title": "To check the engine oil level",
                    "text_snippet": "To check the engine oil level: With the engine stopped, place the boat in a precisely level position and check the dipstick.",
                },
                {
                    "title": "Engine oil level check",
                    "text_snippet": "Make sure that the engine oil level is between the minimum level mark and maximum level mark on the dipstick.",
                },
            ],
        )

        self.assertIn("engine oil level", normalized)
        self.assertNotRegex(normalized, r"[\u4e00-\u9fff]")

    def test_normalize_submission_answer_strips_english_internal_headings(self) -> None:
        normalized = normalize_submission_answer(
            "Direct Conclusion: Based on the references, the base station is a core component."
            " Details/Description: The base station is packaged with a power adapter."
            " Notes: Register up to 4 handsets.",
            question="What is the overview of the base station of a landline?",
            sources=["汇总英文"],
            references=[
                {
                    "title": "What is in the box Base station",
                    "text_snippet": "Base station. Power adapter. Line cord. Quick start guide.",
                },
                {
                    "title": "Register additional handsets",
                    "text_snippet": "You can register additional handsets to the base station. The base station can support up to 4 handsets.",
                },
            ],
        )

        self.assertNotIn("Direct Conclusion", normalized)
        self.assertNotIn("Details/Description", normalized)
        self.assertNotIn("Notes:", normalized)
        self.assertIn("base station", normalized)

    def test_normalize_submission_answer_falls_back_for_weak_english_reference_overlap(self) -> None:
        normalized = normalize_submission_answer(
            "Based on the provided references, there is no specific information given on what to do before the first use of the airfryer.",
            question="If this is the first time to use airfryer, What should I do before first use?",
            sources=["汇总英文"],
            references=[
                {"title": "Note", "text_snippet": "Make sure to connect your Airfryer to a 2.4 GHz home Wi-Fi network."},
                {"title": "Factory reset", "text_snippet": "For a factory reset of the Airfryer, press the temperature and time up buttons."},
            ],
        )

        self.assertIn("manual evidence is not sufficient", normalized)

    def test_normalize_submission_answer_fixes_colon_period_artifact(self) -> None:
        normalized = normalize_submission_answer(
            "活饵舱（Livewell）供水开关：。开启供水：按下按钮启动供水。",
            question="How do I turn on the water supply button on my boat?",
            sources=["汇总英文"],
        )

        self.assertNotIn("：。", normalized)
        self.assertIn("活饵舱（Livewell）供水开关：开启供水", normalized)

    def test_normalize_submission_answer_strips_positive_manual_lead_phrase(self) -> None:
        normalized = normalize_submission_answer(
            "根据现有资料，正确为健身追踪器充电的方法如下。将充电线连接到设备背面的充电接口。",
            question="我刚发现健身追踪器电量低，该如何正确为其充电？",
            sources=["健身追踪器"],
        )

        self.assertNotIn("根据现有资料", normalized)
        self.assertIn("健身追踪器充电方法如下", normalized)

    def test_normalize_submission_answer_strips_loose_manual_tag_noise(self) -> None:
        normalized = normalize_submission_answer(
            '["Manual 34 ", "Manual 34 ", "Manual 34 "] RIDING DOWNHILL OPERATION ON SURFACES OTHER THAN SNOW OR ICE。'
            "oven_03、oven_04、oven_05。",
            question="How should I ride downhill on surfaces other than snow or ice?",
            sources=["汇总英文"],
        )

        self.assertNotIn("Manual 34", normalized)
        self.assertNotIn("oven_03", normalized)
        self.assertIn("RIDING DOWNHILL OPERATION", normalized)

    def test_normalize_submission_answer_avoids_noisy_english_reference_fallback(self) -> None:
        normalized = normalize_submission_answer(
            "根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。",
            question="What are the steps to clean a snowmobile?",
            sources=["汇总英文"],
            references=[
                {
                    "title": "COLOR CODE B Black Br. Brown Ch. Chocolate",
                    "text_snippet": '", ["Manual 34 ", "Manual 34 ", "Manual 34 "]',
                },
                {
                    "title": "RIDING DOWNHILL",
                    "text_snippet": "# RIDING DOWNHILL When riding downhill, keep speed to a minimum.",
                },
            ],
        )

        self.assertIn("manual evidence is not sufficient", normalized)
        self.assertNotIn("RIDING DOWNHILL", normalized)
        self.assertNotIn("Manual 34", normalized)


if __name__ == "__main__":
    unittest.main()
