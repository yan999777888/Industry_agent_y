"""Route user questions to the most suitable answering strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from industry_agent.rag.retriever import analyze_query


_CUSTOMER_SERVICE_TERMS: tuple[str, ...] = (
    "退货", "换货", "退换货", "退款", "运费", "物流", "快递", "发票", "发货",
    "补发", "签收", "售后", "投诉", "赔偿", "订单", "取消订单", "二手", "假货",
    "划痕", "破损", "瑕疵", "少了一件", "少件", "少发", "漏发", "补件", "开发票", "抬头", "税号", "发票类型", "专票", "普票", "保修卡",
    "保障卡", "保质期", "乡镇", "寄到国外", "国外", "配送", "上门取件", "到账",
    "信用卡", "原路返回", "包装盒", "包装破损", "外包装破损", "维修费用", "维修服务", "维修范围", "补偿",
    "虚假宣传", "联系客服", "人工客服", "智能客服", "售后服务", "改地址", "修改地址",
    "收货地址", "预约安装", "安装服务", "上门安装", "保价", "价保", "降价",
    "支付失败", "付款失败", "扣款", "重复扣款", "支付异常", "催发货", "不发货",
    "延迟发货", "发货慢", "保修期", "质保期", "配件", "附件", "补寄配件",
    "纸质版说明书", "电子版说明书", "电子版", "以旧换新", "优惠券", "试用装",
    "试用", "活动规则", "会员", "优惠", "价格咨询", "差价", "补差价", "尺寸差价",
    "尺码", "换大", "换小", "改尺码", "换尺寸",
    "终身维修", "保修", "质保", "维修政策", "服务政策",
)
_MANUAL_TERMS: tuple[str, ...] = (
    "安装", "充电", "指示灯", "闪烁", "故障", "表带", "尺寸", "更换", "连接",
    "设置", "说明书", "默认密码", "安全注意事项", "开机", "关机", "按键", "模式",
    "屏幕", "接口", "电池", "程序", "钻孔", "佩戴", "清洁", "保养", "组装",
)
_CUSTOMER_SERVICE_RE = re.compile(
    "|".join(re.escape(term) for term in sorted(_CUSTOMER_SERVICE_TERMS, key=len, reverse=True))
)
_MANUAL_RE = re.compile(
    "|".join(re.escape(term) for term in sorted(_MANUAL_TERMS, key=len, reverse=True))
)
_PLATFORM_SERVICE_PRIORITY_TERMS = (
    "纸质版说明书", "电子版说明书", "电子版", "以旧换新", "优惠券", "试用装",
    "试用", "试用期", "试用期间", "延长试用", "智能客服", "人工客服", "活动规则",
)
_HARD_SERVICE_TERMS = (
    "退款", "退货", "换货", "退换货", "到账", "订单", "取消订单", "物流", "快递",
    "发票", "抬头", "税号", "专票", "普票", "开发票", "改地址", "收货地址",
    "催发货", "延迟发货", "不发货", "支付失败", "付款失败", "扣款", "重复扣款",
    "少发", "少件", "漏发", "补件", "补寄", "投诉", "赔偿", "运费",
    "上门取件", "联系客服", "人工客服", "智能客服", "以旧换新", "优惠券",
)
_FORCED_SERVICE_TERMS = (
    "少发", "补寄", "运费", "包装", "客服", "售后服务", "发票", "物流",
    "快递", "退款", "退货", "换货", "投诉", "催发货", "订单", "改地址",
    "试用", "试用期", "试用期间", "延长试用",
)
_FORCED_SERVICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(退款|取消订单).*(到账|原路返回|信用卡)"),
    re.compile(r"发票.*(类型|抬头|税号|重开|补开|专票|普票|电子发票)"),
    re.compile(r"(包装|外包装).*(破损|损坏|影响退换)"),
    re.compile(r"(人为|进水|摔坏|磕碰|私拆).*(维修|保修|售后)"),
    re.compile(r"(少发|少件|漏发|缺件|补件|补寄)"),
    re.compile(r"(试用|试用期|试用期间).*(故障|延长|延期|换货)"),
    re.compile(r"(物流|快递|发货).*(没更新|停滞|未收到|签收|改派|改地址)"),
    re.compile(r"(更大|更小|尺码|尺寸).*(能换|换吗|差价|补差价|退差价)"),
    re.compile(r"(换货|更换).*(尺码|尺寸|大一号|小一号)"),
)
_MANUAL_INTENT_HINTS = (
    "怎么", "如何", "怎样", "步骤", "指示灯", "闪烁", "安装", "充电", "设置",
    "连接", "默认密码", "安全注意事项", "佩戴", "清洁", "保养", "拆卸",
    "组装", "开机", "关机", "更换", "说明书",
)
_STRONG_MANUAL_TERMS = (
    "指示灯", "闪烁", "安装", "充电", "设置", "连接", "默认密码",
    "安全注意事项", "佩戴", "清洁", "保养", "拆卸", "组装", "开机",
    "关机", "更换", "表带",
)


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float
    matched_terms: list[str] = field(default_factory=list)
    manual_score: int = 0
    service_score: int = 0
    reason: str = ""


class QuestionRouter:
    """Heuristic router that separates manual QA from generic customer service issues."""

    def route(self, question: str) -> RouteDecision:
        normalized = question.strip()
        analysis = analyze_query(normalized)
        service_terms = self._find_terms(_CUSTOMER_SERVICE_RE, normalized)
        manual_terms = self._find_terms(_MANUAL_RE, normalized)
        has_product_or_model = bool(analysis.products or analysis.models)
        has_platform_service_term = any(term in normalized for term in _PLATFORM_SERVICE_PRIORITY_TERMS)
        has_hard_service_term = any(term in normalized for term in _HARD_SERVICE_TERMS)
        forced_service_match = any(term in normalized for term in _FORCED_SERVICE_TERMS)
        forced_service_pattern = any(pattern.search(normalized) for pattern in _FORCED_SERVICE_PATTERNS)
        has_manual_intent_hint = any(term in normalized for term in _MANUAL_INTENT_HINTS)
        has_strong_manual_term = any(term in normalized for term in _STRONG_MANUAL_TERMS)
        has_explicit_howto = any(term in normalized for term in ("怎么", "如何", "怎样", "步骤"))

        service_score = len(service_terms) * 2
        manual_score = len(manual_terms) * 2

        if has_product_or_model:
            manual_score += 2

        if "说明书" in normalized and not any(term in normalized for term in ("纸质版", "电子版", "补寄", "提供")):
            manual_score += 2
        if "客服" in normalized and not (analysis.products or analysis.models):
            service_score += 2
        if has_platform_service_term:
            service_score += 2
        if forced_service_match:
            service_score += 4
        if forced_service_pattern:
            service_score += 6

        manual_rescue = (
            has_manual_intent_hint
            and not has_platform_service_term
            and not has_hard_service_term
            and (has_product_or_model or len(manual_terms) >= 2 or has_explicit_howto or has_strong_manual_term)
        )

        if has_product_or_model and has_strong_manual_term:
            manual_score += 3
        elif has_explicit_howto and has_strong_manual_term:
            manual_score += 2

        if (forced_service_match or forced_service_pattern) and not manual_rescue:
            service_score = max(service_score, manual_score + 2)

        if manual_rescue and manual_score >= max(4, service_score):
            return RouteDecision(
                route="manual_rag",
                confidence=min(0.64 + manual_score * 0.05, 0.95),
                matched_terms=manual_terms or analysis.products or analysis.models,
                manual_score=manual_score,
                service_score=service_score,
                reason="prefer_manual_rag_with_manual_signal",
            )

        if service_score >= max(4, manual_score + 2):
            return RouteDecision(
                route="customer_service",
                confidence=min(0.65 + service_score * 0.06, 0.95),
                matched_terms=service_terms,
                manual_score=manual_score,
                service_score=service_score,
                reason="matched_customer_service_terms",
            )

        return RouteDecision(
            route="manual_rag",
            confidence=min(0.6 + manual_score * 0.05, 0.95),
            matched_terms=manual_terms or analysis.products or analysis.models,
            manual_score=manual_score,
            service_score=service_score,
            reason="prefer_manual_rag",
        )

    def _find_terms(self, pattern: re.Pattern[str], question: str) -> list[str]:
        seen: set[str] = set()
        terms: list[str] = []
        for match in pattern.finditer(question):
            term = match.group(0)
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
        return terms
