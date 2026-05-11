"""Retrieval-oriented customer-service knowledge base built from structured policy rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industry_agent.agent.customer_service_policy import TopicRule, _TOPIC_RULES
from industry_agent.rag.retriever import analyze_query

_QUERY_ALIAS_MAP: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("待揽收", "未揽收", "一直待揽收"), ("未揽收", "揽收延迟", "物流停滞")),
    (("物流没更新", "物流不更新", "一直没有物流", "物流停滞"), ("没更新", "不更新", "停滞", "卡住")),
    (("乡镇", "偏远地区"), ("乡镇", "偏远地区", "配送范围")),
    (("海外", "国外", "寄到国外"), ("海外", "国外", "国际配送")),
    (("已签收没收到", "签收了没收到", "签收未收到", "已签收但我没收到", "已签收但是没收到"), ("已签收", "未收到", "代签", "误投")),
    (("更大的尺寸", "更小的尺寸", "换大一号", "换小一号"), ("更大", "更小", "换尺寸", "尺码", "尺寸差价")),
    (("补差价", "退差价"), ("差价", "补差价", "退差价")),
    (("开发票", "发票类型"), ("发票", "发票类型", "电子发票", "专票", "普票")),
    (("重开发票", "发票开错"), ("重开", "开错", "抬头", "税号")),
    (("人为损坏", "进水", "摔坏", "磕碰"), ("人为损坏", "进水", "摔坏", "磕碰", "付费维修")),
    (("维修后又坏", "返修后又坏", "维修很久"), ("返修", "复检", "重新返修", "维修延期")),
)


_DEFAULT_KB_PATH = Path(__file__).with_name("customer_service_kb_data.json")


@dataclass(frozen=True)
class CustomerServiceKBEntry:
    entry_id: str
    topic: str
    scenario_name: str
    title: str
    terms: tuple[str, ...]
    overview: str
    process: str
    timeline: str
    fees: str
    eligibility: str
    materials: str
    contact: str
    source_type: str = "policy_projection"
    source_path: str = ""

    @property
    def content(self) -> str:
        parts = [part for part in (
            self.overview,
            self.process,
            self.timeline,
            self.fees,
            self.eligibility,
            self.materials,
            self.contact,
        ) if part]
        return "\n".join(parts)


class CustomerServiceKnowledgeBase:
    """Small in-memory retriever for customer-service knowledge snippets."""

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or _DEFAULT_KB_PATH
        self.entries = _load_entries(self.data_path)

    def search(self, query: str, *, context_topics: list[str] | None = None, limit: int = 4) -> list[dict[str, Any]]:
        analysis = analyze_query(query)
        compact_query = "".join(query.split())
        expanded_terms = _expand_query_terms(query, analysis.keywords, analysis.phrases)
        detail_flags = {
            "timeline": any(term in compact_query for term in ("多久到账", "多久能到", "多久能收到", "多久", "时效")),
            "fees": any(term in compact_query for term in ("运费", "费用", "收费", "多少钱", "差价", "补差价", "退差价")),
            "materials": any(term in compact_query for term in ("材料", "资料", "凭证", "截图", "证明", "准备什么", "需要什么", "提供什么")),
            "contact": any(term in compact_query for term in ("联系谁", "怎么联系", "客服", "人工", "转人工")),
            "eligibility": any(term in compact_query for term in ("支持", "可以", "能否", "能不能", "还能", "影响")),
        }
        scored: list[dict[str, Any]] = []
        for entry in self.entries:
            score = 0.0
            matched_terms: list[str] = []
            scenario_bonus = 0.0
            for term in entry.terms:
                if term and (term in compact_query or term in expanded_terms):
                    matched_terms.append(term)
                    score += 3.5 if entry.scenario_name else 2.0
                    scenario_bonus = max(scenario_bonus, 1.5 if entry.scenario_name else 0.5)

            for phrase in analysis.phrases[:6]:
                if phrase and (phrase in entry.content or phrase in entry.title):
                    score += 2.2
            for keyword in expanded_terms[:16]:
                if keyword and (keyword in entry.content or keyword in entry.title):
                    score += 1.0

            if entry.topic in (context_topics or []):
                score += 2.5
            if entry.scenario_name:
                score += scenario_bonus
            if entry.source_type == "data_file":
                score += 0.6
                if any(len(term) >= 6 for term in matched_terms):
                    score += 2.4

            if detail_flags["timeline"] and entry.timeline:
                score += 2.0
            if detail_flags["fees"] and entry.fees:
                score += 2.0
            if detail_flags["materials"] and entry.materials:
                score += 2.0
            if detail_flags["contact"] and entry.contact:
                score += 1.8
            if detail_flags["eligibility"] and entry.eligibility:
                score += 1.5

            if not entry.scenario_name and any(flag for flag in detail_flags.values()):
                score -= 0.8
            if score <= 0:
                continue
            scored.append(
                {
                    "entry_id": entry.entry_id,
                    "topic": entry.topic,
                    "scenario_name": entry.scenario_name,
                    "title": entry.title,
                    "content": entry.content,
                    "score": round(score, 3),
                    "matched_terms": matched_terms,
                    "overview": entry.overview,
                    "process": entry.process,
                    "timeline": entry.timeline,
                    "fees": entry.fees,
                    "eligibility": entry.eligibility,
                    "materials": entry.materials,
                    "contact": entry.contact,
                    "source_type": entry.source_type,
                    "source_path": entry.source_path,
                }
            )

        scored.sort(
            key=lambda item: (
                item["score"],
                item.get("source_type") == "data_file",
                bool(item["scenario_name"]),
                len(item["matched_terms"]),
                len(item["content"]),
            ),
            reverse=True,
        )
        return scored[:limit]

    def build_context(self, hits: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for index, hit in enumerate(hits, start=1):
            source_tag = "data" if hit.get("source_type") == "data_file" else "policy"
            header = (
                f"[客服参考{index}] 主题：{hit['topic']} | 场景：{hit['scenario_name'] or 'general'} "
                f"| 标题：{hit['title']} | 来源：{source_tag}"
            )
            body_parts = [
                hit.get("overview", ""),
                hit.get("process", ""),
                hit.get("timeline", ""),
                hit.get("fees", ""),
                hit.get("eligibility", ""),
                hit.get("materials", ""),
                hit.get("contact", ""),
            ]
            body = "\n".join(part for part in body_parts if part)
            parts.append(f"{header}\n{body}".strip())
        return "\n\n".join(parts).strip()


def _build_entries() -> list[CustomerServiceKBEntry]:
    entries: list[CustomerServiceKBEntry] = []
    for rule in _TOPIC_RULES:
        entries.append(
            CustomerServiceKBEntry(
                entry_id=f"{rule.topic}::general",
                topic=rule.topic,
                scenario_name="",
                title=_topic_title(rule),
                terms=tuple(rule.terms),
                overview=rule.overview,
                process=rule.process,
                timeline=rule.timeline,
                fees=rule.fees,
                eligibility=rule.eligibility,
                materials=rule.materials,
                contact=rule.contact,
                source_type="policy_projection",
            )
        )
        for scenario in rule.scenarios:
            entries.append(
                CustomerServiceKBEntry(
                    entry_id=f"{rule.topic}::{scenario.name}",
                    topic=rule.topic,
                    scenario_name=scenario.name,
                    title=_scenario_title(rule, scenario.name),
                    terms=tuple(_unique_preserve_order([*rule.terms, *scenario.terms])),
                    overview=scenario.overview or rule.overview,
                    process=scenario.process or rule.process,
                    timeline=scenario.timeline or rule.timeline,
                    fees=scenario.fees or rule.fees,
                    eligibility=scenario.eligibility or rule.eligibility,
                    materials=scenario.materials or rule.materials,
                    contact=scenario.contact or rule.contact,
                    source_type="policy_projection",
                )
            )
    return entries


def _load_entries(data_path: Path) -> list[CustomerServiceKBEntry]:
    data_entries = _load_data_file_entries(data_path)
    policy_entries = _build_entries()
    if not data_entries:
        return policy_entries

    merged: list[CustomerServiceKBEntry] = []
    seen: set[str] = set()
    for entry in data_entries + policy_entries:
        if entry.entry_id in seen:
            continue
        seen.add(entry.entry_id)
        merged.append(entry)
    return merged


def _load_data_file_entries(data_path: Path) -> list[CustomerServiceKBEntry]:
    if not data_path.exists():
        return []

    payload = json.loads(data_path.read_text(encoding="utf-8"))
    raw_entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list):
        return []

    entries: list[CustomerServiceKBEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("entry_id", "")).strip()
        topic = str(item.get("topic", "")).strip()
        title = str(item.get("title", "")).strip()
        if not entry_id or not topic or not title:
            continue
        raw_terms = item.get("terms", [])
        terms = tuple(
            _unique_preserve_order(
                [str(value).strip() for value in raw_terms if str(value).strip()]
            )
        )
        entries.append(
            CustomerServiceKBEntry(
                entry_id=entry_id,
                topic=topic,
                scenario_name=str(item.get("scenario_name", "")).strip(),
                title=title,
                terms=terms,
                overview=str(item.get("overview", "")).strip(),
                process=str(item.get("process", "")).strip(),
                timeline=str(item.get("timeline", "")).strip(),
                fees=str(item.get("fees", "")).strip(),
                eligibility=str(item.get("eligibility", "")).strip(),
                materials=str(item.get("materials", "")).strip(),
                contact=str(item.get("contact", "")).strip(),
                source_type="data_file",
                source_path=str(data_path),
            )
        )
    return entries


def _topic_title(rule: TopicRule) -> str:
    mapping = {
        "refund_exchange": "退换货与退款",
        "invoice": "发票处理",
        "shipping": "物流与配送",
        "complaint": "投诉与升级",
        "after_sales": "售后维修与保修",
        "quality_issue": "破损瑕疵与缺件",
        "order_change": "取消订单与退款到账",
        "manual_request": "说明书获取",
        "platform_service": "平台活动与客服能力",
        "address_change": "修改收货地址",
        "installation_service": "预约安装与上门服务",
        "price_protection": "价保与补差",
        "payment_issue": "支付异常",
        "delivery_delay": "催发货与延迟发货",
        "warranty_period": "保修期确认",
        "accessory_request": "配件补寄",
    }
    return mapping.get(rule.topic, rule.topic)


def _scenario_title(rule: TopicRule, scenario_name: str) -> str:
    mapping = {
        "size_exchange": "换尺寸与差价",
        "refund_arrival": "退款到账",
        "seven_day_no_reason": "7天无理由",
        "quality_reason": "质量问题退换",
        "opened_or_used": "拆封或已使用退换",
        "refund_rejected": "退款驳回",
        "invoice_type": "发票类型",
        "invoice_reissue": "发票重开",
        "invoice_after_issued": "已开票后修改",
        "village_or_overseas": "乡镇或海外配送",
        "tracking_stalled": "物流停滞",
        "package_lost_or_returned": "丢件退回与异常回流",
        "signed_but_missing": "已签收未收到",
        "redirect_or_wrong_address": "改派与地址错误",
        "in_warranty": "保内维修",
        "out_of_warranty": "过保维修",
        "human_damage": "人为损坏",
        "repair_delay_or_repeat_failure": "返修复发与维修延期",
        "rejected_or_disputed": "售后驳回与争议",
        "packaging_damage": "包装破损",
        "missing_items": "少发漏发缺件",
        "trade_in": "以旧换新",
        "coupon": "优惠券使用",
        "trial_sample": "试用装与试用服务",
        "trial_extension_or_fault": "试用延期与试用故障",
        "smart_customer_service": "智能客服与转人工",
        "address_after_shipment": "已发货后改地址",
        "installation_reschedule": "安装改约",
        "repeat_charge": "重复扣款",
        "accessory_rejected": "补寄配件被驳回",
    }
    topic_title = _topic_title(rule)
    return f"{topic_title} - {mapping.get(scenario_name, scenario_name)}"


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _expand_query_terms(query: str, keywords: list[str], phrases: list[str]) -> list[str]:
    expanded = _unique_preserve_order([*phrases, *keywords])
    compact_query = "".join(query.split())
    for triggers, aliases in _QUERY_ALIAS_MAP:
        if any(trigger in compact_query for trigger in triggers):
            expanded.extend(alias for alias in aliases if alias not in expanded)
    return expanded
