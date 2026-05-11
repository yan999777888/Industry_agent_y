#!/usr/bin/env python3
"""Generate a platform submission CSV from public questions via /chat."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_FALLBACK_ANSWER = "根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。"
MAX_SINGLE_ANSWER_CHARS = 520
MAX_MULTI_ANSWER_CHARS = 760
MAX_PIC_MARKERS = 3
SUBMISSION_INCLUDE_PIC = True
_CUSTOMER_SERVICE_KEYWORDS = (
    "退货", "换货", "退款", "运费", "物流", "快递", "发票", "补发", "签收",
    "售后", "维修", "保修", "投诉", "赔偿", "订单", "发货", "包装", "瑕疵",
    "少件", "划痕", "假货", "虚假宣传", "国外", "乡镇",
    "纸质版说明书", "电子版", "说明书", "以旧换新", "优惠券", "试用装",
    "智能客服", "人工客服", "活动", "价格", "优惠", "会员",
)
_IMAGE_ID_RE = re.compile(r"\b(?:Manual\d+_\d+|Manual\s*\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\b")
_RELATED_IMAGE_SECTION_RE = re.compile(r"\n*相关图片：(?:\n[^\n]*)*", flags=re.IGNORECASE)
_LABEL_REPLACEMENTS = (
    ("回答：", ""),
    ("结论：", ""),
    ("操作/说明：", ""),
    ("处理步骤：", ""),
    ("时效/费用：", ""),
    ("补充说明：", ""),
    ("注意事项：", ""),
)
_FALLBACK_SENTENCE_PATTERNS: tuple[str, ...] = (
    r"根据现有资料无法准确回答此问题[。]?",
    r"根据现有资料无法回答此问题[。]?",
    r"根据现有资料无法回答如何[^。！？!?]*[。]?",
    r"根据现有资料，无法(?:准确)?回答[^。！？!?]*[。]?",
    r"根据现有资料，无法提供[^。！？!?]*[。]?",
    r"根据现有资料，无法直接回答[^。！？!?]*[。]?",
    r"请补充更明确的产品名称、型号、故障现象或图片后再试[。]?",
    r"请补充产品名称、型号、故障现象或上传更清晰的图片后再试[。]?",
    r"当前回答仅基于知识库中的说明书资料，请以实际产品和原文为准[。]?",
    r"请以实际产品型号和说明书原文为准[。]?",
    r"建议您检查问题表述是否完整[^。！？!?]*[。]?",
    r"如果您的问题指的是其他类型的[^。！？!?]*[。]?",
    r"建议您查阅您船只的具体操作手册[^。！？!?]*[。]?",
    r"Based on the available references, I cannot provide[^.。！？!?]*[.。]?",
    r"Based on the provided references, there is no specific information given[^.。！？!?]*[.。]?",
    r"The references only mention[^.。！？!?]*[.。]?",
    r"The provided reference materials do not contain[^.。！？!?]*[.。]?",
    r"The references cover topics such as[^.。！？!?]*[.。]?",
    r"They only cover[^.。！？!?]*[.。]?",
    r"According to the existing[^.。！？!?]*[.。]?",
    r"Therefore, according to the available documentation, this question cannot be answered[.。]?",
)
_INTERNAL_SENTENCE_PATTERNS: tuple[str, ...] = (
    r"The answer is extracted from the retrieved manual evidence\.?",
    r"Please follow the original manual for safety-critical operation\.?",
)
_PLACEHOLDER_SENTENCE_PATTERNS: tuple[str, ...] = (
    r"与上一问处理思路一致[^。！？!?]*[。]?",
    r"可按相同材料和流程继续处理[^。！？!?]*[。]?",
    r"模型未返回有效回答[。]?",
    r"当前回答为空[。]?",
    r"未检索到有效内容[。]?",
    r"以上内容仅供参考[。]?",
)
FALLBACK_RE = re.compile(r"无法回答|无法准确|补充.*产品|根据现有资料")
_BAD_TAG_PATTERNS: tuple[str, ...] = (
    r"<IMG\b[^>]*>",
    r"</IMG>",
    r"<text>",
    r"</text>",
    r"<PIC>\s*</PIC>",
    r"<PIC\s*/>",
)
_QUESTION_BLOCK_RE = re.compile(r"(?:^|\n)\s*(问题\s*\d+)\s*[:：]\s*", flags=re.IGNORECASE)
_QUESTION_SPLIT_RE = re.compile(r'"\s*,\s*"|\n+|[？?]\s*["”]?\s*[,，]?\s*["“]?')
_GENERIC_CUSTOMER_SERVICE_PREFIX_RE = re.compile(
    r"^(?:退换货和退款|售后、维修和保修问题|发票问题|物流和配送问题|订单取消和退款到账问题|"
    r"投诉、假货、虚假宣传、二手商品或服务态度问题|破损、瑕疵、少件或保质期异常这类问题|"
    r"配件、附件、包装盒或补寄问题|平台活动、优惠、试用和智能客服能力|"
    r"修改收货地址|预约安装或上门安装服务|支付异常问题|催发货和延迟发货问题)"
)
_GENERIC_CUSTOMER_SERVICE_SUPPORT_RE = re.compile(
    r"^(?:如果你现在方便|如果你这边已经有|如果问题已经影响|如果已经影响|"
    r"建议(?:先)?(?:准备|携带|带上|优先准备|一次性准备)|"
    r"这样(?:通常)?更快|减少重复核对|联系人工客服加急核查)"
)
_CUSTOMER_SERVICE_TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("refund", ("退款", "退货", "换货", "退换货", "7天无理由", "七天无理由", "取消订单")),
    ("invoice", ("发票", "抬头", "税号", "专票", "普票", "电子发票", "发票类型")),
    ("shipping", ("物流", "快递", "发货", "配送", "签收", "改地址", "收货地址", "运费", "乡镇", "国外")),
    ("complaint", ("投诉", "假货", "虚假宣传", "二手", "赔偿", "辱骂")),
    ("after_sales", ("售后", "维修", "保修", "质保", "人为损坏", "进水", "摔坏", "磕碰")),
    ("quality_issue", ("破损", "包装破损", "外包装破损", "瑕疵", "少件", "少发", "漏发", "缺件", "划痕")),
    ("platform_service", ("试用", "试用装", "以旧换新", "优惠券", "会员", "人工客服", "智能客服")),
    ("installation", ("安装服务", "上门安装", "预约安装")),
    ("payment", ("支付失败", "付款失败", "重复扣款", "扣款", "支付异常")),
)
_ENGLISH_QUESTION_STOPWORDS = {
    "what", "when", "where", "which", "while", "with", "without", "how", "why", "who",
    "are", "is", "was", "were", "the", "and", "for", "from", "that", "this", "these",
    "those", "into", "onto", "your", "their", "there", "have", "has", "had", "can",
    "could", "would", "should", "will", "about", "than", "then", "using", "use",
    "used", "best", "method", "methods", "steps", "time",
}
_ENGLISH_INTERNAL_HEADINGS = (
    "direct conclusion",
    "details/description",
    "description",
    "operation/steps",
    "steps",
    "notes",
    "note",
)
_DIRECT_CUSTOMER_SERVICE_TERMS = (
    "24小时", "48小时", "工作日", "原路退回", "原支付账户", "信用卡", "专票", "普票",
    "电子发票", "重开", "更正", "补寄", "缺件", "漏发", "改地址", "改派", "误签",
    "签收", "上门取件", "免费维修", "付费维修", "延长质保", "保修", "价保", "以旧换新",
    "优惠券", "试用期", "重复扣款",
)
_GENERIC_CUSTOMER_SERVICE_REWRITE_TERMS = (
    "这类问题更适合按通用客服流程处理",
    "相关情况需要结合订单信息",
    "通常需要确认",
    "建议一次性准备",
    "减少重复核对",
    "必要时联系人工客服",
    "如果你愿意",
)
_HEAVY_SUBMISSION_NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"问题\s*\d+\s*[:：]", flags=re.IGNORECASE),
    re.compile(r"Direct Conclusion|Details/Description|Operation/Steps|Notes", flags=re.IGNORECASE),
    re.compile(r"与上一问处理思路一致|模型未返回有效回答|当前回答为空|未检索到有效内容"),
    re.compile(r"根据现有资料无法(?:准确)?回答"),
)


def read_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "question"]:
            raise ValueError(f"question file must have columns ['id', 'question'], got {reader.fieldnames}")
        return [{"id": row["id"], "question": row["question"]} for row in reader]


def call_chat(base_url: str, question: str, timeout: int) -> dict:
    payload = {
        "question": question,
        "images": [],
        "session_id": None,
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_submission(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_debug_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def rows_from_debug_records(records: list[dict], fallback_answer: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        raw_response = record.get("response") or {}
        data = raw_response.get("data") or {}
        raw_answer = str(data.get("answer") or "").strip() or fallback_answer
        rows.append(
            {
                "id": str(record.get("id", "")),
                "ret": normalize_submission_answer(
                    raw_answer,
                    question=str(record.get("question", "")),
                    sources=list(data.get("sources", []) or []),
                    image_ids=list(data.get("image_ids", []) or []),
                    references=list(data.get("references", []) or []),
                ),
            }
        )
    return rows


def normalize_submission_answer(answer: str, *, question: str, sources: list[str] | None = None,
                                image_ids: list[str] | None = None,
                                references: list[dict] | None = None) -> str:
    sources = sources or []
    image_ids = image_ids or []
    references = references or []
    is_english_question = bool(re.search(r"[A-Za-z]", question)) and not bool(re.search(r"[\u4e00-\u9fff]", question))
    text = answer.strip()
    if not text:
        return _format_with_images(DEFAULT_FALLBACK_ANSWER, [])

    text = _strip_submission_artifacts(text)
    question_blocks = _split_question_blocks(text)
    if question_blocks:
        return _normalize_multi_question_answer(
            question_blocks,
            question=question,
            sources=sources,
            image_ids=image_ids,
            references=references,
        )

    text = _basic_submission_cleanup(text)

    text_without_fallback = _strip_fallback_sentences(text)
    text_without_fallback = _remove_question_echo(text_without_fallback, question=question)
    text_without_fallback = _remove_question_like_sentences(text_without_fallback, question=question)
    if _looks_like_pure_fallback(text, text_without_fallback):
        reference_answer = _build_reference_based_answer(question=question, references=references)
        if reference_answer:
            return _format_with_images(reference_answer, image_ids)
        fb = _build_submission_fallback(question=question, sources=sources)
        return _format_with_images(fb, [])
    text = text_without_fallback or text

    if _should_prefer_light_submission_cleanup(text, question=question, sources=sources):
        cleaned = _lightweight_submission_finalize(text, question=question)
        if cleaned:
            return _format_with_images(cleaned, image_ids)

    if "customer_service_policy" in sources:
        text = re.sub(r"如果你愿意，我建议[^。]*。?", "", text)
        text = re.sub(r"如果你愿意，我建议下一步优先补充[^。]*。?", "", text)
        text = re.sub(r"这类问题更适合按通用客服流程处理。?", "", text)
        text = _rewrite_customer_service_submission(text, question=question)
        text = _compress_customer_service_answer(text)

    text = re.sub(r"\s{2,}", " ", text).strip(" ，,；;")
    text = _strip_submission_artifacts(text)
    if _is_low_information_submission_text(text):
        reference_answer = _build_reference_based_answer(question=question, references=references)
        text = reference_answer or _build_submission_fallback(question=question, sources=sources)
    text = _remove_question_like_sentences(text, question=question)
    text = _compress_submission_answer(text, question=question)
    text = _polish_submission_text(text, question=question)
    if is_english_question:
        if references and _should_rewrite_english_submission(text):
            reference_answer = _build_reference_based_answer(question=question, references=references)
            text = reference_answer or _build_submission_fallback(question=question, sources=sources)
        elif references and "manual evidence is not sufficient" in text.lower():
            reference_answer = _build_reference_based_answer(question=question, references=references)
            if reference_answer:
                text = reference_answer
    if _is_low_information_submission_text(text):
        text = _build_submission_fallback(question=question, sources=sources)
    if not text.endswith(("。", "！", "？")):
        text += "。"
    return _format_with_images(text, image_ids)


def _normalize_multi_question_answer(
    blocks: list[tuple[str, str]],
    *,
    question: str,
    sources: list[str],
    image_ids: list[str],
    references: list[dict],
) -> str:
    sub_questions = _extract_sub_questions(question)
    normalized_blocks: list[str] = []
    for index, (label, block_text) in enumerate(blocks, start=1):
        sub_question = sub_questions[index - 1] if index - 1 < len(sub_questions) else question
        body = _normalize_single_block_text(
            block_text,
            question=sub_question,
            sources=sources,
            references=references,
        )
        body = _polish_submission_text(body, question=sub_question)
        if not body:
            continue
        normalized_blocks.append(body)

    merged = _merge_submission_segments(normalized_blocks, question=question)
    if not merged:
        reference_answer = _build_reference_based_answer(question=question, references=references)
        merged = reference_answer or _build_submission_fallback(question=question, sources=sources)
    if len(merged) > MAX_MULTI_ANSWER_CHARS:
        merged = merged[:MAX_MULTI_ANSWER_CHARS].rstrip(" ，,；;\n") + "。"
    return _format_with_images(merged, image_ids)


def _normalize_single_block_text(
    text: str,
    *,
    question: str,
    sources: list[str],
    references: list[dict],
) -> str:
    is_english_question = bool(re.search(r"[A-Za-z]", question)) and not bool(re.search(r"[\u4e00-\u9fff]", question))
    cleaned = text
    cleaned = _basic_submission_cleanup(cleaned)

    cleaned_without_fallback = _strip_fallback_sentences(cleaned)
    cleaned_without_fallback = _remove_question_echo(cleaned_without_fallback, question=question)
    cleaned_without_fallback = _remove_question_like_sentences(cleaned_without_fallback, question=question)
    if _looks_like_pure_fallback(cleaned, cleaned_without_fallback):
        reference_answer = _build_reference_based_answer(question=question, references=references)
        if reference_answer:
            return reference_answer
        return _build_submission_fallback(question=question, sources=sources)

    cleaned = cleaned_without_fallback or cleaned
    if _should_prefer_light_submission_cleanup(cleaned, question=question, sources=sources):
        finalized = _lightweight_submission_finalize(cleaned, question=question)
        if finalized:
            return finalized
    if "customer_service_policy" in sources:
        cleaned = re.sub(r"如果你愿意，我建议[^。]*。?", "", cleaned)
        cleaned = re.sub(r"如果你愿意，我建议下一步优先补充[^。]*。?", "", cleaned)
        cleaned = re.sub(r"如果你现在方便，我建议[^。]*。?", "", cleaned)
        cleaned = re.sub(r"这类问题更适合按通用客服流程处理。?", "", cleaned)
        cleaned = _rewrite_customer_service_submission(cleaned, question=question)
        cleaned = _compress_customer_service_answer(cleaned)

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ，,；;")
    cleaned = _strip_submission_artifacts(cleaned)
    if _is_low_information_submission_text(cleaned):
        reference_answer = _build_reference_based_answer(question=question, references=references)
        cleaned = reference_answer or _build_submission_fallback(question=question, sources=sources)
    cleaned = _remove_question_like_sentences(cleaned, question=question)
    cleaned = _compress_submission_answer(cleaned, question=question)
    cleaned = _polish_submission_text(cleaned, question=question)
    if is_english_question:
        if references and _should_rewrite_english_submission(cleaned):
            reference_answer = _build_reference_based_answer(question=question, references=references)
            cleaned = reference_answer or _build_submission_fallback(question=question, sources=sources)
        elif references and "manual evidence is not sufficient" in cleaned.lower():
            reference_answer = _build_reference_based_answer(question=question, references=references)
            if reference_answer:
                cleaned = reference_answer
    if _is_low_information_submission_text(cleaned):
        cleaned = _build_submission_fallback(question=question, sources=sources)
    return cleaned.strip()


def _format_with_images(text: str, image_ids: list[str]) -> str:
    """Normalize final submission text and optionally keep inline picture markers."""
    text = _strip_submission_artifacts(text)
    text = _polish_submission_text(text, question="")
    selected_image_ids = _select_submission_image_ids(image_ids)
    if not SUBMISSION_INCLUDE_PIC or not selected_image_ids:
        text = re.sub(r"\s*<PIC>\s*", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[ \t]{2,}", " ", text).strip(" \n\r\t，,；;")
        if text and not text.endswith(("。", "！", "？", ".", "!", "?")):
            text += "。"
        return text

    existing_pic_count = len(re.findall(r"<PIC>", text, flags=re.IGNORECASE))
    pic_count = min(existing_pic_count if existing_pic_count else len(selected_image_ids), len(selected_image_ids), MAX_PIC_MARKERS)
    cleaned_text = re.sub(r"\s*<PIC>\s*", " ", text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text).strip(" \n\r\t，,；;")
    if pic_count:
        cleaned_text = cleaned_text.rstrip("。！？.!?") + "<PIC>" * pic_count
    elif cleaned_text and not cleaned_text.endswith(("。", "！", "？", ".", "!", "?")):
        cleaned_text += "。"

    ids_json = json.dumps(selected_image_ids[:pic_count], ensure_ascii=False)
    return f'"{cleaned_text}";{ids_json}'


def _basic_submission_cleanup(text: str) -> str:
    cleaned = str(text)
    for old, new in _LABEL_REPLACEMENTS:
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.replace("**", "")
    cleaned = _RELATED_IMAGE_SECTION_RE.sub("", cleaned)
    cleaned = _IMAGE_ID_RE.sub("", cleaned)
    cleaned = cleaned.replace("- 无", "")
    cleaned = cleaned.replace("- ", "")
    cleaned = re.sub(r"参考资料[^\n。]*[。]?", "", cleaned)
    cleaned = re.sub(r"当前资料[^\n。]*[。]?", "", cleaned)
    cleaned = re.sub(r"资料中仅[^\n。]*[。]?", "", cleaned)
    cleaned = re.sub(r"\[参考\s*\d*\]", "", cleaned)
    cleaned = re.sub(r"参考\s*\[\d+\]", "", cleaned)
    cleaned = re.sub(r"（参考\s*[^\）]*）", "", cleaned)
    cleaned = re.sub(r"\(参考\s*[^\)]*\)", "", cleaned)
    cleaned = _strip_internal_sentences(cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n+", " ", cleaned).strip(" |;；，,")
    return cleaned


def _has_direct_customer_service_signal(text: str, *, question: str) -> bool:
    if any(term in text for term in _DIRECT_CUSTOMER_SERVICE_TERMS):
        return True
    question_terms = _extract_question_terms(question)
    overlap = _reference_overlap_score(text, question_terms)
    return overlap >= 2 and len(_split_submission_sentences(text)) <= 4


def _looks_customer_service_generic(text: str) -> bool:
    return _GENERIC_CUSTOMER_SERVICE_PREFIX_RE.match(text) is not None or any(
        term in text for term in _GENERIC_CUSTOMER_SERVICE_REWRITE_TERMS
    )


def _contains_customer_service_material_or_contact_guidance(text: str) -> bool:
    return any(
        term in text
        for term in (
            "订单号", "税号", "抬头", "凭证", "截图", "照片", "视频", "聊天记录",
            "人工客服", "联系客服", "联系承运商", "联系快递", "提交申请", "提交售后",
        )
    )


def _should_prefer_light_submission_cleanup(text: str, *, question: str, sources: list[str]) -> bool:
    stripped = str(text).strip()
    if not stripped:
        return False
    if any(pattern.search(stripped) for pattern in _HEAVY_SUBMISSION_NOISE_PATTERNS):
        return False
    if _is_low_information_submission_text(stripped):
        return False

    limit = MAX_MULTI_ANSWER_CHARS if _is_multi_question(question) else MAX_SINGLE_ANSWER_CHARS
    if len(re.sub(r"<PIC>", "", stripped, flags=re.IGNORECASE)) > limit:
        return False

    is_english_question = bool(re.search(r"[A-Za-z]", question)) and not bool(re.search(r"[\u4e00-\u9fff]", question))
    if is_english_question and _should_rewrite_english_submission(stripped):
        return False

    if "customer_service_policy" in sources:
        if (
            not _question_asks_customer_service_materials_or_contact(question)
            and _contains_customer_service_material_or_contact_guidance(stripped)
            and any(term in stripped for term in ("建议", "联系客服", "人工客服", "提交", "核实"))
        ):
            return False
        if _looks_customer_service_generic(stripped):
            return False
        return _has_direct_customer_service_signal(stripped, question=question)

    return True


def _lightweight_submission_finalize(text: str, *, question: str) -> str:
    cleaned = _polish_submission_text(text, question=question)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" \n\r\t，,；;")
    if not cleaned:
        return ""
    limit = MAX_MULTI_ANSWER_CHARS if _is_multi_question(question) else MAX_SINGLE_ANSWER_CHARS
    if len(cleaned) > limit:
        return ""
    return cleaned


def _build_submission_fallback(*, question: str, sources: list[str]) -> str:
    is_english_question = bool(re.search(r"[A-Za-z]", question)) and not bool(re.search(r"[\u4e00-\u9fff]", question))
    if "customer_service_policy" in sources or any(keyword in question for keyword in _CUSTOMER_SERVICE_KEYWORDS):
        if any(term in question for term in ("纸质版说明书", "电子版", "说明书")):
            return "通常可以优先在商品详情页、订单页或品牌官网查找电子版说明书；纸质说明书是否补寄，需要结合商品型号和包装清单由客服核实。"
        if "以旧换新" in question:
            return "以旧换新通常取决于商品类目、活动规则和旧机状态，建议先查看商品页面是否有以旧换新入口，再按页面要求提交旧机信息和估价。"
        if "优惠券" in question:
            return "优惠券是否可用通常取决于适用商品、有效期、门槛和活动规则，建议在结算页查看是否可勾选使用。"
        if "试用装" in question or "试用" in question:
            return "是否提供试用装或试用服务通常取决于商品活动和库存规则，建议查看商品页面活动说明或联系人工客服确认。"
        if "智能客服" in question or "人工客服" in question:
            return "智能客服通常可以解答订单、物流、退换货、发票和售后等常见问题；如果问题较复杂或需要人工核实，建议转人工客服并提供订单号和相关截图。"
        return "相关情况需要结合订单信息、商品状态和平台规则确认。建议提供订单号、商品名称、问题照片或聊天记录，以便继续判断处理方式。"
    if is_english_question:
        return "The current manual evidence is not sufficient to answer this question. Please provide a clearer product name, model, symptom, or image."
    return "当前还无法准确定位对应的说明书内容，请补充产品名称、型号、故障现象或图片后再试。"


def _strip_submission_artifacts(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r'^\s*"(.+)"\s*;\s*\[(?:"[^"]*"\s*,?\s*)*\]\s*$', r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'";\s*\[(?:"[^"]*"\s*,?\s*)*\]\s*$', "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[(?:\"(?:Manual\d+_\d+|[A-Za-z]+_\d+)\"\s*,?\s*)+\]\s*$", "", cleaned)
    cleaned = re.sub(r"\[(?:\"?\s*(?:Manual\s*\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\s*\"?\s*,?\s*)+\]\s*$", "", cleaned)
    cleaned = re.sub(r'\[(?:\s*"\s*"\s*,?\s*)+\]', " ", cleaned)
    cleaned = re.sub(r'\[(?:\s*"\s+"\s*,?\s*)+\]', " ", cleaned)
    for pattern in _BAD_TAG_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<IMG\b[^。！？!?<]*(?:[。！？!?]|$)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?PIC[^>]*>", "<PIC>", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\s*/?\s*(?!PIC\b)[A-Za-z][^>]{0,80}>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*(?:问题\s*\d+|回答|结论|操作/说明|操作说明|操作|说明|处理步骤|时效/费用|时效费用|补充说明|注意事项|相关图片)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?m)^\s*\d+\s*[.)、。]\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*[一二三四五六七八九十]+\s*[、.．]\s*", "", cleaned)
    cleaned = re.sub(r"(?:^|[\n。；;:：])\s*(?:直接结论|操作步骤|操作要点|使用前与连接|注意事项|补充说明)(?:\s*[:：]|\s+)", "。", cleaned)
    cleaned = re.sub(
        r"(?:^|[\n。；;:：])\s*(?:Direct Conclusion|Conclusion|Details/Description|Description|Operation/Steps|Steps|Notes|Note)(?:\s*[:：]|\s+)",
        "。",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(?:^|[\n。；;:：])\s*(?:操作/说明|注意事项)\s*（[^）]{0,30}）\s*[:：]?", "。", cleaned)
    cleaned = cleaned.replace("###", " ")
    cleaned = cleaned.replace("1.。", "")
    cleaned = cleaned.replace("：。", "：")
    cleaned = cleaned.replace('\\"', '"')
    cleaned = re.sub(r"(?:\b(?:Manual\s*\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\b[、,，\s]*){2,}", " ", cleaned)
    cleaned = cleaned.strip().strip('"').strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(" \n\r\t，,；;")


def _split_question_blocks(text: str) -> list[tuple[str, str]]:
    matches = list(_QUESTION_BLOCK_RE.finditer(text))
    if not matches:
        return []
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = re.sub(r"\s+", "", match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            blocks.append((label, content))
    return blocks


def _extract_sub_questions(question: str) -> list[str]:
    quoted = [item.strip() for item in re.findall(r'"([^\"]+)"', question) if item.strip()]
    if quoted:
        return quoted
    parts = [part.strip(" \"'“”") for part in _QUESTION_SPLIT_RE.split(question) if part.strip(" \"'“”")]
    return parts


def _normalize_multi_block_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", text)


def _is_low_information_submission_text(text: str) -> bool:
    key = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", str(text))
    key = re.sub(r"^(问题\d+|结论|注意事项|操作说明|操作|说明|处理步骤|时效费用|补充说明|相关图片)+", "", key)
    return len(key) < 8


def _select_submission_image_ids(image_ids: list[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for image_id in image_ids:
        normalized = str(image_id).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
        if len(selected) >= MAX_PIC_MARKERS:
            break
    return selected


def _normalize_pic_markers(text: str) -> str:
    cleaned = _strip_submission_artifacts(text)
    pic_count = min(len(re.findall(r"<PIC>", cleaned, flags=re.IGNORECASE)), MAX_PIC_MARKERS)
    cleaned = re.sub(r"\s*<PIC>\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip(" \n\r\t，,；;")
    if pic_count:
        cleaned = cleaned.rstrip("。！？.!?") + "<PIC>" * pic_count
    if cleaned and not cleaned.endswith(("。", "！", "？", ".", "!", "?", ">")):
        cleaned += "。"
    return cleaned


def _strip_internal_sentences(text: str) -> str:
    cleaned = text
    for pattern in _INTERNAL_SENTENCE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([。！？.!?])", r"\1", cleaned)
    return cleaned.strip(" ，,；;。")


def _strip_fallback_sentences(text: str) -> str:
    cleaned = text
    for pattern in _FALLBACK_SENTENCE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"[。]{2,}", "。", cleaned)
    cleaned = re.sub(r"参考\d+\s*产品：[^。！？!?]*[。]?", "", cleaned)
    cleaned = re.sub(r"[、，,\s]{3,}", " ", cleaned)
    return cleaned.strip(" ，,；;。")


def _strip_placeholder_sentences(text: str) -> str:
    cleaned = text
    for pattern in _PLACEHOLDER_SENTENCE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"[。]{2,}", "。", cleaned)
    return cleaned.strip(" ，,；;。")


def _remove_question_echo(text: str, *, question: str) -> str:
    cleaned = text
    candidates = re.findall(r'"([^\"]+)"', question)
    if not candidates:
        candidates = [question]
    for candidate in candidates:
        segment = re.sub(r"\s+", " ", candidate).strip(" ,，;；\"'")
        if len(segment) < 6:
            continue
        cleaned = cleaned.replace(segment, "", 1)
        no_punctuation = segment.strip("。！？!?")
        cleaned = cleaned.replace(no_punctuation + "。", "", 1)
        cleaned = cleaned.replace(no_punctuation, "", 1)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"^[，,；;。:：\s]+", "", cleaned)
    cleaned = re.sub(r"\s+[，,；;。:：]", "", cleaned)
    return cleaned.strip(" ，,；;。")


def _remove_question_like_sentences(text: str, *, question: str) -> str:
    question_terms = set(_extract_question_terms(question))
    kept: list[str] = []
    for index, sentence in enumerate(_split_submission_sentences(text)):
        stripped = sentence.strip(" ，,；;。")
        if not stripped:
            continue
        overlap = _reference_overlap_score(stripped, list(question_terms))
        looks_interrogative = bool(re.search(r"(是什么|有哪些|怎么|如何|多久|吗|能不能|是否|哪里)$", stripped))
        looks_echo_heading = bool(re.search(r"你们的|退款政策|服务范围|发票类型|商品支持|需要自己承担", stripped))
        if index == 0 and (looks_interrogative or looks_echo_heading) and overlap >= 2 and len(stripped) <= 45:
            continue
        kept.append(stripped)
    if not kept:
        return ""
    return "。".join(kept)


def _looks_like_pure_fallback(original: str, stripped: str) -> bool:
    if not original.strip():
        return True
    if original.strip() == DEFAULT_FALLBACK_ANSWER:
        return True
    if not stripped.strip():
        return True
    informative_chars = len(re.sub(r"\s+", "", stripped))
    return informative_chars < 18 and (
        "根据现有资料无法准确回答此问题" in original
        or "根据现有资料无法回答此问题" in original
    )


def _build_reference_based_answer(*, question: str, references: list[dict]) -> str:
    """Build a short extractive answer when the model refused despite evidence."""

    query_terms = _extract_question_terms(question)
    is_english_question = bool(re.search(r"[A-Za-z]", question)) and not bool(re.search(r"[\u4e00-\u9fff]", question))
    candidates: list[tuple[int, int, str]] = []
    order = 0
    for ref in references[:4]:
        title = _clean_reference_text(str(ref.get("title", "")))
        snippet = _clean_reference_text(str(ref.get("text_snippet", "")))
        if title and not _looks_like_reference_noise(title):
            title_overlap = _reference_overlap_score(title, query_terms)
            if title_overlap > 0 and (not is_english_question or title_overlap >= 2):
                candidates.append((title_overlap + 2, order, title))
                order += 1
        for sentence in _split_submission_sentences(snippet):
            sentence = _clean_reference_text(sentence)
            is_numbered_step = bool(re.search(r"(?:^|\s)\d+\s", sentence))
            if is_english_question and re.search(r"[\u4e00-\u9fff]", sentence):
                continue
            if (len(sentence) < 12 and not is_numbered_step) or _looks_like_reference_noise(sentence):
                continue
            bonus = 3 if is_numbered_step else 0
            if re.search(r"取下|装入|插入|连接|按下|选择|确认|确保", sentence):
                bonus += 2
            candidates.append((_reference_overlap_score(sentence, query_terms) + bonus, order, sentence))
            order += 1

    candidates.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
    selected: list[tuple[int, int, str]] = []
    selected_texts: list[str] = []
    for score, candidate_order, candidate in candidates:
        if score <= 0:
            if is_english_question:
                continue
            if selected:
                continue
        if _is_near_duplicate_sentence(candidate, selected_texts):
            continue
        selected.append((score, candidate_order, candidate))
        selected_texts.append(candidate)
        if len(selected) >= 4:
            break

    if not selected:
        return ""
    has_short_quoted_token = bool(re.search(r'"[A-Za-z0-9_-]{1,4}"', question))
    min_english_score = 1 if has_short_quoted_token else 2
    if is_english_question and selected[0][0] < min_english_score:
        return ""
    selected_texts = [text for _, _, text in sorted(selected, key=lambda item: item[1])]
    if is_english_question:
        answer_overlap = _reference_overlap_score(" ".join(selected_texts), query_terms)
        if answer_overlap < min_english_score:
            return ""
        if len(selected_texts) == 1:
            answer = selected_texts[0]
        else:
            answer = selected_texts[0] + " " + " ".join(selected_texts[1:])
        return answer.strip(" ，,；;。") + "."

    lines = [selected_texts[0]]
    if len(selected_texts) > 1:
        lines.append("操作要点：" + "；".join(selected_texts[1:]))
    return "。".join(line.strip("。") for line in lines if line.strip()) + "。"


def _clean_reference_text(text: str) -> str:
    cleaned = str(text)
    cleaned = cleaned.replace("#", " ")
    cleaned = re.sub(r'\[(?:\s*"\s*(?:Manual\s*\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\s*"\s*,?\s*)+\]', " ", cleaned)
    cleaned = _IMAGE_ID_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\[\[PIC[^\]]*\]\]", " ", cleaned)
    cleaned = re.sub(r"<PIC>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\u[0-9a-fA-F]{4}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -|，,；;。")


def _looks_like_reference_noise(text: str) -> bool:
    if re.search(r"Manual\s*\d+", text):
        return True
    if re.search(r'^"?\s*\[', text):
        return True
    if re.search(r"\.{5,}|…{3,}", text):
        return True
    if len(re.findall(r"\b\d+\b", text)) >= 8:
        return True
    return False


def _extract_question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for quoted in re.findall(r'"([^"]+)"', question):
        token = quoted.strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{1,4}", token):
            terms.append(token.lower())
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question):
        lowered = word.lower()
        if len(word) >= 3 and lowered not in _ENGLISH_QUESTION_STOPWORDS:
            terms.append(lowered)
    cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]+", question))
    for size in (4, 3, 2):
        for index in range(0, max(0, len(cjk_text) - size + 1)):
            term = cjk_text[index : index + size]
            if term not in {"如何", "请问", "什么", "哪些", "使用", "时候"}:
                terms.append(term)
    return _unique_texts(terms)


def _reference_overlap_score(text: str, query_terms: list[str]) -> int:
    normalized = re.sub(r"\s+", "", text.lower())
    return sum(1 for term in query_terms if term and term.lower() in normalized)


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _infer_customer_service_topic(question: str) -> str:
    for topic, keywords in _CUSTOMER_SERVICE_TOPIC_RULES:
        if any(keyword in question for keyword in keywords):
            return topic
    return ""


def _question_asks_customer_service_materials_or_contact(question: str) -> bool:
    return any(
        term in question
        for term in (
            "准备什么", "需要什么", "要什么材料", "哪些材料", "什么材料", "哪些凭证",
            "提供什么", "上传什么", "截图", "凭证", "证据", "联系谁", "怎么联系",
            "转人工", "人工客服", "联系客服",
        )
    )


def _build_customer_service_direct_lead(question: str, topic: str) -> str:
    if topic == "refund":
        if "7天无理由" in question or "七天无理由" in question:
            return "未使用且在 7 天无理由时效内的商品通常可以退货；无质量问题时运费一般由买家承担。"
        if "取消订单" in question and any(term in question for term in ("到账", "原路返回", "信用卡")):
            return "已付款订单通常可以先申请取消；退款一般原路退回，到账时间取决于支付渠道处理进度。"
        if any(term in question for term in ("到账", "原路返回", "信用卡")):
            return "退款一般原路退回原支付账户；如果是信用卡支付，到账时间通常要看发卡行处理进度。"
        return ""
    if topic == "invoice":
        if any(term in question for term in ("发票类型", "专票", "普票", "电子发票")):
            return "通常支持按订单开票；具体支持电子发票、普通发票还是专用发票，要以订单开票入口为准。"
        if any(term in question for term in ("重开", "开错", "抬头", "税号")):
            return "发票信息填错后一般可以申请更正或重开；是否能重开，要看发票状态和平台规则。"
        return ""
    if topic == "shipping":
        if any(term in question for term in ("少发", "漏发", "缺件", "补寄")):
            return "少发、漏发或缺件通常可以先按补寄或缺件售后处理；核实后补寄费用一般不由买家承担。"
        if any(term in question for term in ("签收", "未收到", "已签收")):
            return "如果物流显示已签收但实际没收到，通常应先按误签或末端异常回查处理。"
        if any(term in question for term in ("改地址", "收货地址")):
            return "如果订单还没完成配送，通常可以尝试改地址或改派；是否成功要看当前物流节点。"
        return ""
    if topic == "complaint":
        return ""
    if topic == "after_sales":
        if any(term in question for term in ("人为", "进水", "摔坏", "磕碰", "私拆")):
            return "人为损坏通常不能按免费保修处理，但很多情况下仍可申请付费检测或付费维修。"
        if any(term in question for term in ("保修期", "质保期")):
            return "是否还能保修，通常要结合购买时间、故障类型和保修凭证确认。"
        return ""
    if topic == "quality_issue":
        if any(term in question for term in ("包装破损", "外包装破损")):
            return "包装破损时应先确认商品本体和配件是否受损；如影响使用或签收，可申请破损售后。"
        if any(term in question for term in ("少件", "少发", "漏发", "缺件")):
            return "少件、少发或漏发通常可以申请补寄或缺件售后；核实后相关费用一般不由买家承担。"
        return ""
    if topic == "platform_service":
        if "以旧换新" in question:
            return "是否支持以旧换新，要看商品类目、活动规则和旧机状态。"
        if "优惠券" in question:
            return "优惠券是否可用，主要看有效期、门槛、适用品类和是否可叠加。"
        if "试用" in question:
            return "试用、延期试用或试用期故障处理，要结合活动规则、试用协议和故障责任判断。"
        return ""
    if topic == "installation":
        return ""
    if topic == "payment":
        return ""
    return ""


def _tighten_customer_service_sentence(sentence: str) -> str:
    cleaned = sentence.strip(" ，,；;。")
    replacements = (
        ("通常需要确认", "先确认"),
        ("一般建议先", "建议先"),
        ("通常与", "与"),
        ("通常取决于", "要看"),
        ("通常要看", "要看"),
        ("通常还要看", "还要看"),
        ("通常会受", "会受"),
        ("通常受", "受"),
        ("通常更可能由", "通常由"),
        ("更可能由", "通常由"),
        ("通常更可能进入", "通常进入"),
        ("通常更容易", "一般更容易"),
        ("通常可以先", "一般可先"),
        ("通常可以", "一般可以"),
        ("通常应先", "应先"),
        ("通常不是", "一般不是"),
        ("通常不以", "一般不以"),
    )
    for source, target in replacements:
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"^如果你愿意，?", "", cleaned)
    cleaned = re.sub(r"^您好[，,]?\s*", "", cleaned)
    cleaned = _GENERIC_CUSTOMER_SERVICE_PREFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"^\s*[，,；;：:]\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ，,；;。")


def _rewrite_customer_service_submission(text: str, *, question: str) -> str:
    topic = _infer_customer_service_topic(question)
    lead = _build_customer_service_direct_lead(question, topic)
    asks_materials_or_contact = _question_asks_customer_service_materials_or_contact(question)
    sentences = _split_submission_sentences(text)
    sub_questions = _extract_sub_questions(question) or [question]
    selected: list[str] = []
    for index, sentence in enumerate(sentences):
        cleaned = _tighten_customer_service_sentence(sentence)
        if not cleaned or len(cleaned) < 5:
            continue
        if _is_question_echo_sentence(cleaned, question=question):
            continue
        if any(cleaned in sub_question for sub_question in sub_questions) and len(cleaned) <= 14:
            continue
        if re.search(r"(是什么|有哪些|怎么|如何|多久|吗|能不能|是否)$", cleaned):
            continue
        if any(term in cleaned for term in _GENERIC_CUSTOMER_SERVICE_REWRITE_TERMS):
            continue
        if (
            lead
            and not asks_materials_or_contact
            and _GENERIC_CUSTOMER_SERVICE_SUPPORT_RE.match(cleaned)
            and any(term in cleaned for term in ("订单号", "截图", "照片", "视频", "聊天记录", "人工客服"))
        ):
            continue
        score = 0
        if any(term in cleaned for term in ("订单号", "税号", "抬头", "凭证", "截图", "照片", "视频", "故障描述")):
            score += 3
        if any(term in cleaned for term in ("建议", "联系客服", "售后", "申请", "核实", "提交")):
            score += 2
        if any(term in cleaned for term in ("运费", "费用", "到账", "工作日", "时效", "补寄", "维修费")):
            score += 2
        if _GENERIC_CUSTOMER_SERVICE_PREFIX_RE.match(sentence.strip()):
            score -= 2
        if lead and any(term in cleaned for term in ("通常更适合", "一般适合", "先确认订单号", "先确认商品型号")):
            score -= 1
        if "要看" in cleaned and len(cleaned) <= 16:
            score -= 2
        score -= index
        if _is_near_duplicate_sentence(cleaned, [lead, *selected] if lead else selected):
            continue
        if score >= 0:
            selected.append(cleaned)
        if len(selected) >= 3:
            break

    combined: list[str] = []
    if lead:
        combined.append(lead.strip("。"))
    combined.extend(item.strip("。") for item in selected if item.strip("。"))
    if not combined:
        return text
    return "。".join(combined).strip(" ，,；;。") + "。"


def _strip_weak_leads(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"根据现有资料[，,:：]\s*", "", cleaned)
    cleaned = re.sub(r"根据参考资料[，,:：]\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bBased on the references,\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\bAccording to the references,\s*", "", cleaned)
    cleaned = re.sub(r"关于([^。]{1,50})，我知道以下信息[。:：]?", r"\1：", cleaned)
    cleaned = re.sub(r"^正确为(.{1,40})的方法如下", r"\1方法如下", cleaned)
    cleaned = re.sub(r"\b(?:操作说明|具体说明|操作要求)\s*[:：]\s*", "", cleaned)
    cleaned = cleaned.replace("没有专门的", "未单独说明")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ，,；;")


def _compress_customer_service_answer(text: str) -> str:
    sentences = _split_submission_sentences(text)
    if not sentences:
        return text

    selected: list[str] = []
    for sentence in sentences:
        cleaned = sentence.strip(" ，,；;。")
        if len(cleaned) < 4:
            continue
        if cleaned.endswith(("？", "?")):
            continue
        if _is_near_duplicate_sentence(cleaned, selected):
            continue
        selected.append(cleaned)
        if len(selected) >= 5:
            break

    return "。 ".join(selected).strip()


def _compress_submission_answer(text: str, *, question: str) -> str:
    limit = MAX_MULTI_ANSWER_CHARS if _is_multi_question(question) else MAX_SINGLE_ANSWER_CHARS
    cleaned = _normalize_pic_markers(text)
    if len(cleaned) <= limit:
        return cleaned

    sentences = _split_submission_sentences(re.sub(r"<PIC>", " ", cleaned))
    if not sentences:
        return cleaned[:limit].rstrip(" ，,；;") + "。"

    question_terms = _extract_question_terms(question)
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip(" ，,；;。")
        if len(sentence) < 4:
            continue
        score = _reference_overlap_score(sentence, question_terms)
        if re.search(r"步骤|操作|注意|必须|请勿|不要|建议|可以|需要|表示|说明|通常|先|再|最后", sentence):
            score += 2
        if FALLBACK_RE.search(sentence):
            score -= 4
        scored.append((score, -index, sentence))

    scored.sort(reverse=True)
    selected_with_index: list[tuple[int, str]] = []
    selected_texts: list[str] = []
    total = 0
    for _, neg_index, sentence in scored:
        if _is_near_duplicate_sentence(sentence, selected_texts):
            continue
        next_len = len(sentence) + 1
        if selected_texts and total + next_len > limit:
            continue
        selected_with_index.append((-neg_index, sentence))
        selected_texts.append(sentence)
        total += next_len
        if len(selected_texts) >= (7 if _is_multi_question(question) else 5):
            break

    if not selected_texts:
        selected_texts = [sentences[0][:limit].strip(" ，,；;。")]
    else:
        selected_texts = [sentence for _, sentence in sorted(selected_with_index)]
    answer = "。".join(selected_texts).strip(" ，,；;。") + "。"
    if len(answer) > limit:
        answer = answer[:limit].rstrip(" ，,；;。") + "。"
    pic_count = min(len(re.findall(r"<PIC>", cleaned, flags=re.IGNORECASE)), MAX_PIC_MARKERS)
    if pic_count:
        answer = answer.rstrip("。！？.!?") + "<PIC>" * pic_count
    return answer


def _merge_submission_segments(segments: list[str], *, question: str) -> str:
    collected: list[str] = []
    for segment in segments:
        for sentence in _split_submission_sentences(segment):
            cleaned = sentence.strip(" ，,；;。")
            if not cleaned:
                continue
            if _is_near_duplicate_sentence(cleaned, collected):
                continue
            collected.append(cleaned)
    if not collected:
        return ""
    merged = "。".join(collected).strip(" ，,；;。") + "。"
    return _compress_submission_answer(merged, question=question)


def _polish_submission_text(text: str, *, question: str) -> str:
    cleaned = _strip_submission_artifacts(text)
    cleaned = _strip_internal_sentences(cleaned)
    cleaned = _strip_fallback_sentences(cleaned)
    cleaned = _strip_placeholder_sentences(cleaned)
    cleaned = _strip_weak_leads(cleaned)
    cleaned = _remove_question_echo(cleaned, question=question) if question else cleaned
    cleaned = _remove_question_like_sentences(cleaned, question=question) if question else cleaned
    sentences = _split_submission_sentences(cleaned)
    selected: list[str] = []
    for sentence in sentences:
        candidate = sentence.strip(" ，,；;。")
        if not candidate:
            continue
        if candidate in {"无", "暂无", "未知", "无信息"}:
            continue
        if question and _is_question_echo_sentence(candidate, question=question):
            continue
        if _is_near_duplicate_sentence(candidate, selected):
            continue
        selected.append(candidate)
    cleaned = "。".join(selected).strip(" ，,；;。")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" \n\r\t，,；;")
    if cleaned and not cleaned.endswith(("。", "！", "？", ".", "!", "?", ">")):
        cleaned += "。"
    return cleaned


def _strip_english_section_labels(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"\b(?:Direct Conclusion|Conclusion|Details/Description|Description|Operation/Steps|Steps|Notes|Note)\s*:\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _should_rewrite_english_submission(text: str) -> bool:
    lowered = _strip_english_section_labels(text).lower()
    normalized = re.sub(r"(结论|操作/说明|注意事项|相关图片|无)", " ", lowered)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    if cjk_chars >= 6:
        return True
    if any(heading in lowered for heading in _ENGLISH_INTERNAL_HEADINGS):
        return True
    noisy_refusal_patterns = (
        "there is no specific information given",
        "cannot be answered",
        "according to the available documentation",
        "the references cover topics such as",
        "the provided reference materials do not contain",
        "the references only mention",
    )
    return any(pattern in lowered for pattern in noisy_refusal_patterns)


def _is_question_echo_sentence(text: str, *, question: str) -> bool:
    candidate_key = _normalize_sentence_key(text)
    if len(candidate_key) < 10:
        return False
    question_keys = [_normalize_sentence_key(part) for part in (_extract_sub_questions(question) or [question])]
    for question_key in question_keys:
        if len(question_key) < 10:
            continue
        if candidate_key == question_key:
            return True
        if candidate_key in question_key:
            return True
    return False


def _is_multi_question(question: str) -> bool:
    if question.count("?") + question.count("？") >= 2:
        return True
    return len(re.findall(r'"[^"]{4,}"', question)) >= 2


def _split_submission_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])", normalized)
    return [part.strip(" 。！？!?") for part in parts if part.strip(" 。！？!?")]


def _normalize_sentence_key(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "", text.lower())


def _is_near_duplicate_sentence(candidate: str, existing: list[str]) -> bool:
    candidate_key = _normalize_sentence_key(candidate)
    if not candidate_key:
        return True
    for sentence in existing:
        sentence_key = _normalize_sentence_key(sentence)
        if not sentence_key:
            continue
        if candidate_key == sentence_key:
            return True
        shorter = min(len(candidate_key), len(sentence_key))
        if shorter >= 16 and (candidate_key in sentence_key or sentence_key in candidate_key):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission CSV by calling the local /chat API.")
    parser.add_argument("--questions", type=Path, default=Path("submission/question_public.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission/submission_generated.csv"))
    parser.add_argument("--debug-output", type=Path, default=Path("submission/submission_generated_debug.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N questions; 0 means all.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--fallback-answer", default=DEFAULT_FALLBACK_ANSWER)
    parser.add_argument(
        "--from-debug",
        action="store_true",
        help="Regenerate the submission CSV from an existing debug JSONL without calling /chat.",
    )
    args = parser.parse_args()

    if args.from_debug:
        records = read_debug_records(args.debug_output)
        rows = rows_from_debug_records(records, args.fallback_answer)
        write_submission(args.output, rows)
        print(f"Saved submission to {args.output}")
        print(f"Rows: {len(rows)}")
        return

    questions = read_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    rows: list[dict[str, str]] = []
    if args.debug_output.exists():
        args.debug_output.unlink()

    for index, item in enumerate(questions, start=1):
        started = time.time()
        answer = args.fallback_answer
        ok = False
        error = ""
        raw_response: dict | None = None
        try:
            raw_response = call_chat(args.base_url, item["question"], args.timeout)
            data = raw_response.get("data", {})
            raw_answer = str(data.get("answer") or "").strip() or args.fallback_answer
            raw_image_ids = list(data.get("image_ids", []) or [])
            answer = normalize_submission_answer(
                raw_answer,
                question=item["question"],
                sources=list(data.get("sources", []) or []),
                image_ids=raw_image_ids,
                references=list(data.get("references", []) or []),
            )
            ok = raw_response.get("code") == 0
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = str(exc)

        rows.append({"id": item["id"], "ret": answer})
        debug_record = {
            "id": item["id"],
            "question": item["question"],
            "ok": ok,
            "ret": answer,
            "raw_answer": raw_response.get("data", {}).get("answer", "") if raw_response else "",
            "elapsed_sec": round(time.time() - started, 3),
            "error": error,
            "response": raw_response,
        }
        append_jsonl(args.debug_output, debug_record)

        status = "OK" if ok else "FALLBACK"
        print(f"[{index}/{len(questions)}] {status} id={item['id']} elapsed={debug_record['elapsed_sec']}s")
        if args.sleep > 0:
            time.sleep(args.sleep)

    write_submission(args.output, rows)
    print(f"Saved submission to {args.output}")
    print(f"Saved debug log to {args.debug_output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
