"""Lightweight customer-service policy responses for non-manual questions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyResponse:
    answer: str
    confidence: float
    matched_topics: list[str]


_POLICY_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "refund_exchange",
        ("退货", "换货", "退换货", "退款"),
        "这类问题通常需要结合订单状态、签收时间、商品是否使用以及平台规则判断。建议你先提供订单号、购买渠道、签收时间和商品当前状态；如果是未使用且仍在退换时效内，一般可优先申请退换或退款。",
    ),
    (
        "invoice",
        ("发票", "开发票", "抬头"),
        "发票问题通常需要确认订单号、开票类型和抬头信息。建议先提供订单号、开票类型（个人/企业）以及发票抬头；如果抬头填错，是否支持重开通常取决于平台规则和开票状态。",
    ),
    (
        "shipping",
        ("物流", "快递", "发货", "补发", "签收", "运费", "乡镇", "国外", "配送"),
        "物流和配送问题通常需要结合订单号、发货状态、收货地址和物流单号查询。建议你补充订单号、物流单号、收货地区和异常现象；如果是待揽收、补发超时、乡镇/海外配送或运费问题，一般都需要客服结合订单系统进一步确认。",
    ),
    (
        "complaint",
        ("投诉", "虚假宣传", "假货", "二手", "辱骂", "赔偿"),
        "如果涉及投诉、假货、虚假宣传、二手商品或服务态度问题，建议立即保留订单记录、聊天记录、商品照片和视频证据，并联系人工客服升级处理。若平台支持投诉单或售后工单，优先走官方投诉流程会更稳妥。",
    ),
    (
        "after_sales",
        ("售后", "维修", "保修", "保修卡", "保障卡", "维修费用"),
        "售后与维修问题通常需要确认商品型号、故障现象、是否人为损坏、购买时间以及保修凭证。建议先准备订单号、型号、故障描述和照片；如果是人为损坏或超出保修范围，通常可能需要自费维修，具体仍以售后审核结果为准。",
    ),
    (
        "quality_issue",
        ("划痕", "破损", "瑕疵", "少件", "少了一件", "保质期"),
        "商品破损、瑕疵、少件或保质期异常这类问题，建议尽快拍照留证，并提供订单号、签收时间和问题细节给客服。若问题发生在签收后较短时间内，通常更容易走补发、换货或售后处理流程。",
    ),
    (
        "order_change",
        ("取消订单", "订单", "到账", "信用卡", "原路返回"),
        "订单取消和退款到账问题通常取决于订单是否发货、支付方式以及平台规则。建议先提供订单号和支付方式；如果订单尚未发货，通常更容易申请取消，退款一般按原支付路径退回，但实际到账时间仍要以支付渠道处理时效为准。",
    ),
    (
        "manual_request",
        ("纸质版说明书", "电子版", "说明书"),
        "如果你需要纸质版或电子版说明书，建议先提供具体产品名称和型号。电子版通常可以优先通过产品页面、品牌官网或客服渠道获取；纸质版是否支持补寄，则通常要看商品包装配置和售后政策。",
    ),
)

_TOPIC_RE = re.compile(
    "|".join(re.escape(term) for _, terms, _ in _POLICY_RULES for term in terms)
)


class CustomerServicePolicy:
    """Generate conservative but helpful replies for generic service scenarios."""

    def answer(self, question: str) -> PolicyResponse:
        matched_topics: list[str] = []
        snippets: list[str] = []
        normalized = question.strip()

        for topic, terms, response in _POLICY_RULES:
            if any(term in normalized for term in terms):
                matched_topics.append(topic)
                snippets.append(response)

        if not snippets:
            answer = (
                "这类问题更偏向订单、售后或平台服务流程，当前说明书资料无法直接确认。"
                "建议你补充订单号、商品名称、购买渠道、问题现象以及相关照片或聊天记录，"
                "这样更方便进一步判断应该走退款、换货、补发、维修还是投诉处理。"
            )
            return PolicyResponse(answer=answer, confidence=0.58, matched_topics=[])

        answer = self._compose_answer(snippets)
        confidence = min(0.68 + 0.05 * len(matched_topics), 0.88)
        return PolicyResponse(answer=answer, confidence=confidence, matched_topics=matched_topics)

    def _compose_answer(self, snippets: list[str]) -> str:
        unique_snippets: list[str] = []
        seen: set[str] = set()
        for snippet in snippets:
            if snippet in seen:
                continue
            seen.add(snippet)
            unique_snippets.append(snippet)

        body = " ".join(unique_snippets[:2])
        return (
            "这类问题更适合按通用客服流程处理。"
            f"{body}"
            " 如果你愿意，我建议下一步优先补充订单号、商品名称或型号、购买渠道，以及异常照片或截图。"
        )
