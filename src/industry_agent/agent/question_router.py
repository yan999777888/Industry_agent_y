"""Route user questions to the most suitable answering strategy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from industry_agent.rag.retriever import analyze_query


_CUSTOMER_SERVICE_TERMS: tuple[str, ...] = (
    "退货", "换货", "退换货", "退款", "运费", "物流", "快递", "发票", "发货",
    "补发", "签收", "售后", "投诉", "赔偿", "订单", "取消订单", "二手", "假货",
    "划痕", "破损", "瑕疵", "少了一件", "少件", "开发票", "抬头", "保修卡",
    "保障卡", "保质期", "乡镇", "寄到国外", "国外", "配送", "上门取件", "到账",
    "信用卡", "原路返回", "包装盒", "包装破损", "维修费用", "维修服务", "补偿",
    "虚假宣传", "联系客服", "人工客服", "智能客服", "售后服务", "改地址", "修改地址",
    "收货地址", "预约安装", "安装服务", "上门安装", "保价", "价保", "降价",
    "支付失败", "付款失败", "扣款", "重复扣款", "支付异常", "催发货", "不发货",
    "延迟发货", "发货慢", "保修期", "质保期", "配件", "附件", "补寄配件",
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

        service_score = len(service_terms) * 2
        manual_score = len(manual_terms) * 2

        if analysis.products or analysis.models:
            manual_score += 2

        if "说明书" in normalized:
            manual_score += 2
        if "客服" in normalized and not (analysis.products or analysis.models):
            service_score += 2

        if service_score >= max(2, manual_score + 1):
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
