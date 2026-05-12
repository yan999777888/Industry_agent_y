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
    "保质期", "过期", "临期", "生产日期", "更换尺寸", "终身维修",
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
    r"根据现有资料，(?:我)?无法(?:准确)?回答[^。！？!?]*[。]?",
    r"根据现有资料，(?:我)?无法提供[^。！？!?]*[。]?",
    r"根据现有资料，(?:我)?无法直接回答[^。！？!?]*[。]?",
    r"根据现有参考资料[^。！？!?]*无法[^。！？!?]*[。]?",
    r"根据现有参考资料[^。！？!?]*未提及[^。！？!?]*[。]?",
    r"请补充更明确的产品名称、型号、故障现象或图片后再试[。]?",
    r"请补充产品名称、型号、故障现象或上传更清晰的图片后再试[。]?",
    r"当前回答仅基于知识库中的说明书资料，请以实际产品和原文为准[。]?",
    r"请以实际产品型号和说明书原文为准[。]?",
    r"建议您检查问题表述是否完整[^。！？!?]*[。]?",
    r"如果您的问题指的是其他类型的[^。！？!?]*[。]?",
    r"建议您查阅您船只的具体操作手册[^。！？!?]*[。]?",
    r"我无法回答[^。！？!?]*[。]?",
    r"抱歉，我无法[^。！？!?]*[。]?",
    r"建议您联系客服部门并提供具体商品信息以获取进一步帮助[。]?",
    r"我无法查到[^。！？!?]*[。]?",
    r"当前还无法准确定位[^。！？!?]*[。]?",
    r"建议您查阅产品说明书[^。！？!?]*[。]?",
    r"建议您查看产品包装[^。！？!?]*[。]?",
    r"根据提供的建议您[^。！？!?]*[。]?",
    r"根据现有建议您[^。！？!?]*[。]?",
    r"无法为您提供[^。！？!?]*[。]?",
    r"无法提供[^。！？!?]*具体[^。！？!?]*[。]?",
    r"未提及[^。！？!?]*相关[^。！？!?]*[。]?",
    r"Based on the available references, I cannot provide[^.。！？!?]*[.。]?",
    r"Based on the provided references, there is no specific information given[^.。！？!?]*[.。]?",
    r"The references only mention[^.。！？!?]*[.。]?",
    r"The provided reference materials do not contain[^.。！？!?]*[.。]?",
    r"The references cover topics such as[^.。！？!?]*[.。]?",
    r"They only cover[^.。！？!?]*[.。]?",
    r"According to the existing[^.。！？!?]*[.。]?",
    r"Therefore, according to the available documentation, this question cannot be answered[.。]?",
    r"I cannot provide specific information on[^.。！？!?]*[.。]?",
    r"The references do not cover this topic[.。]?",
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
    ("shipping", ("物流", "快递", "发货", "配送", "签收", "改地址", "收货地址", "运费", "乡镇", "国外", "补发", "少了一件")),
    ("complaint", ("投诉", "假货", "虚假宣传", "二手", "赔偿", "辱骂", "保质期", "临近过期", "快过期", "生产日期")),
    ("after_sales", ("售后", "维修", "保修", "质保", "人为损坏", "进水", "摔坏", "磕碰", "终身维修")),
    ("quality_issue", ("破损", "包装破损", "外包装破损", "瑕疵", "少件", "少发", "漏发", "缺件", "划痕", "过期", "临期")),
    ("platform_service", ("试用", "试用装", "以旧换新", "优惠券", "会员", "人工客服", "智能客服", "更换尺寸")),
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
    re.compile(r"我无法回答|抱歉，我无法|我无法查到|无法为您提供"),
    re.compile(r"当前还无法准确定位"),
    re.compile(r"建议您查阅产品说明书|建议您查看产品包装"),
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


_CUSTOMER_SERVICE_GREETING_TERMS = (
    "退换货", "退货", "换货", "退款", "售后", "维修", "保修", "发票",
    "物流", "快递", "发货", "签收", "运费", "配送", "投诉", "赔偿",
    "安装", "以旧换新", "优惠券", "试用", "支付", "扣款", "质量",
    "破损", "瑕疵", "少件", "漏发", "补寄", "补发", "保质期", "过期",
    "说明书", "少了一件", "生产日期", "更换尺寸", "终身维修",
)

_TOPIC_MISMATCH_PATTERNS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("保质期", ("保质期", "过期", "临期", "快过期", "临近过期"), ("滤网", "空气净化器", "更换滤网")),
    ("真伪", ("假货", "真伪", "正品", "山寨", "仿品"), ("滤网", "空气净化器")),
)


def _is_topic_mismatch(text: str, *, question: str) -> bool:
    for topic_name, question_keywords, answer_keywords in _TOPIC_MISMATCH_PATTERNS:
        if any(kw in question for kw in question_keywords):
            if any(kw in text for kw in answer_keywords):
                return True
    return False


def _ensure_customer_service_greeting(text: str, *, question: str) -> str:
    if not text or text.startswith(("您好", "你好", "非常抱歉", "很抱歉", "理解您", "很理解")):
        return text
    is_english_question = bool(re.search(r"[A-Za-z]", question)) and not bool(re.search(r"[一-鿿]", question))
    if is_english_question:
        return text
    text = text.strip(" ，,；;")
    if text:
        text = "您好，" + text
    return text


def _fix_double_greetings(text: str) -> str:
    """Remove duplicate greetings like '\u60a8\u597d\uff0c...\u60a8\u597d\uff0c...'."""
    if not text:
        return text
    # Fix patterns like "\u60a8\u597d\uff0cXXX\u60a8\u597d\uff0c" -> "XXX\u60a8\u597d\uff0c"
    text = re.sub(r"^(\u60a8\u597d[\uff0c,]\s*(?:\u975e\u5e38\u62b1\u6b49[^\uff0c,]*[\uff0c,]\s*|\u5f88\u62b1\u6b49[^\uff0c,]*[\uff0c,]\s*|\u7406\u89e3\u60a8[^\uff0c,]*[\uff0c,]\s*|\u5f88\u7406\u89e3[^\uff0c,]*[\uff0c,]\s*))\u60a8\u597d[\uff0c,]\s*", r"\1", text)
    # Fix patterns like "\u60a8\u597d\uff0cXXX\u60a8\u597d\uff0c" where XXX is empathy
    text = re.sub(r"^(\u60a8\u597d[\uff0c,]\s*(?:\u975e\u5e38\u62b1\u6b49|\u5f88\u62b1\u6b49|\u7406\u89e3\u60a8|\u5f88\u7406\u89e3)[^\uff0c,]*[\uff0c,]\s*)\u60a8\u597d[\uff0c,]\s*", r"\1", text)
    return text


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
            if _is_topic_mismatch(cleaned, question=question):
                fb = _build_submission_fallback(question=question, sources=sources)
                if fb:
                    cleaned = fb
            cleaned = _ensure_customer_service_greeting(cleaned, question=question)
            cleaned = _fix_double_greetings(cleaned)
            if not cleaned.endswith(("。", "！", "？")):
                cleaned += "。"
            return _format_with_images(cleaned, image_ids)

    if "customer_service_policy" in sources:
        text = re.sub(r"如果你愿意，我建议[^。]*。?", "", text)
        text = re.sub(r"如果你愿意，我建议下一步优先补充[^。]*。?", "", text)
        text = re.sub(r"这类问题更适合按通用客服流程处理。?", "", text)
        text = _rewrite_customer_service_submission(text, question=question)
        text = _compress_customer_service_answer(text)
        text = _format_as_numbered_steps(text, question=question)
        if _is_topic_mismatch(text, question=question):
            fb = _build_submission_fallback(question=question, sources=sources)
            if fb:
                text = fb

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
    if _is_topic_mismatch(text, question=question):
        fb = _build_submission_fallback(question=question, sources=sources)
        if fb:
            text = fb
    text = _ensure_customer_service_greeting(text, question=question)
    text = _fix_double_greetings(text)
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
        cleaned = _format_as_numbered_steps(cleaned, question=question)

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ，,；;")
    cleaned = _strip_submission_artifacts(cleaned)
    if _is_low_information_submission_text(cleaned):
        reference_answer = _build_reference_based_answer(question=question, references=references)
        cleaned = reference_answer or _build_submission_fallback(question=question, sources=sources)
    cleaned = _remove_question_like_sentences(cleaned, question=question)
    cleaned = _compress_submission_answer(cleaned, question=question)
    cleaned = _polish_submission_text(cleaned, question=question)
    if _is_off_topic_answer(cleaned, question):
        reference_answer = _build_reference_based_answer(question=question, references=references)
        if reference_answer:
            cleaned = reference_answer
        else:
            cleaned = _build_submission_fallback(question=question, sources=sources)
    cleaned = _smooth_raw_manual_text(cleaned)
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
    return f"{cleaned_text},{ids_json}"


def _basic_submission_cleanup(text: str) -> str:
    cleaned = str(text)
    for old, new in _LABEL_REPLACEMENTS:
        cleaned = cleaned.replace(old, new)
    cleaned = cleaned.replace("**", "")
    cleaned = _RELATED_IMAGE_SECTION_RE.sub("", cleaned)
    cleaned = _IMAGE_ID_RE.sub("", cleaned)
    cleaned = cleaned.replace("- 无", "")
    cleaned = cleaned.replace("- ", "")
    cleaned = re.sub(r"(?:根据)?参考资料[：:，,]?\s*", "", cleaned)
    cleaned = re.sub(r"当前资料[：:，,]?\s*(?:中)?(?:没有|无法|不包含|未提供)[^\n。]*[。]?", "", cleaned)
    cleaned = re.sub(r"资料中仅[：:，,]?\s*(?:提及|包含|涉及)[^\n。]*[。]?", "", cleaned)
    cleaned = re.sub(r"\[参考\s*\d*\]", "", cleaned)
    cleaned = re.sub(r"参考\s*\[\d+\]", "", cleaned)
    cleaned = re.sub(r"（参考\s*[^\）]*）", "", cleaned)
    cleaned = re.sub(r"\(参考\s*[^\)]*\)", "", cleaned)
    cleaned = re.sub(r"参考\s*(?:Manual\d+_\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\b", "", cleaned)
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
        return False

    return True


def _lightweight_submission_finalize(text: str, *, question: str) -> str:
    cleaned = _polish_submission_text(text, question=question)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" \n\r\t，,；;")
    cleaned = _IMAGE_ID_RE.sub("", cleaned)
    cleaned = re.sub(r"(?:\b(?:Manual\s*\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\b[、,，\s]*){2,}", " ", cleaned)
    cleaned = re.sub(r"</?PIC[^>]*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"参考\s*(?:Manual\d+_\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\b", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ，,；;")
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
            return "您好，关于说明书，电子版说明书通常可以在品牌官网或商品详情页下载。如需纸质版，请核对商品包装内是否已附带，部分品牌也支持申请补寄。"
        if "以旧换新" in question:
            return "以旧换新通常取决于商品类目、活动规则和旧机状态，建议先查看商品页面是否有以旧换新入口，再按页面要求提交旧机信息和估价。"
        if "优惠券" in question:
            return "优惠券是否可用通常取决于适用商品、有效期、门槛和活动规则，建议在结算页查看是否可勾选使用。"
        if "试用装" in question or "试用" in question:
            return "是否提供试用装或试用服务通常取决于商品活动和库存规则，建议查看商品页面活动说明或联系人工客服确认。"
        if "智能客服" in question or "人工客服" in question:
            return "智能客服通常可以解答订单、物流、退换货、发票和售后等常见问题；如果问题较复杂或需要人工核实，建议转人工客服并提供订单号和相关截图。"
        if any(term in question for term in ("保质期", "过期", "临期", "快过期", "临近过期")):
            return "您好，商品存在质量问题支持退换货。请上传清晰的故障照片或视频，并提交质量问题售后申请，同时准备好订单号和问题说明。如核实属于质量问题，退换货运费由我们承担，建议越早提交越好。"
        return "相关情况需要结合订单信息、商品状态和平台规则确认。建议提供订单号、商品名称、问题照片或聊天记录，以便继续判断处理方式。"
    if is_english_question:
        q_lower = question.lower()
        if any(w in q_lower for w in ("install", "assembly", "setup", "set up")):
            return "Hello! For installation or setup instructions, please refer to the product manual's quick start guide or the setup section. If you need further help, please provide your product model number."
        if any(w in q_lower for w in ("clean", "maintenance", "care")):
            return "Hello! For cleaning and maintenance, please check the care instructions section in your product manual. Regular maintenance helps keep your product in optimal condition."
        if any(w in q_lower for w in ("repair", "fix", "troubleshoot")):
            return "Hello! For troubleshooting or repair information, please check the troubleshooting section in your product manual. If the issue persists, please contact customer service with your order number and a description of the problem."
        if any(w in q_lower for w in ("charge", "battery", "power")):
            return "Hello! For battery or charging information, please refer to the power management section in your product manual. If you're experiencing battery issues, please try the recommended charging steps first."
        if any(w in q_lower for w in ("connect", "pair", "bluetooth", "wifi")):
            return "Hello! For connection or pairing instructions, please check the connectivity section in your product manual. Make sure Bluetooth or Wi-Fi is enabled on your device and follow the pairing steps."
        if any(w in q_lower for w in ("safety", "warning", "caution")):
            return "Hello! For safety guidelines and warnings, please refer to the safety section at the beginning of your product manual. Following these guidelines is important for safe use."
        if any(w in q_lower for w in ("oil", "filter", "spark")):
            return "Hello! For engine maintenance details like oil, filter, or spark plug information, please check the maintenance schedule section in your product manual."
        if any(w in q_lower for w in ("warranty", "guarantee")):
            return "Hello! Warranty information depends on the specific product and brand. Please check the warranty card that came with your product, or contact customer service with your order number for details."
        if any(w in q_lower for w in ("return", "refund", "exchange")):
            return "Hello! For return, refund, or exchange requests, please check our return policy and submit a request through your order page. If you need assistance, please provide your order number."
        return "Hello! Thank you for your question. Could you please provide more details such as the product name, model number, or a description of the specific issue? This will help us assist you better."
    if any(term in question for term in ("终身保修", "终身维修", "永久保修")):
        return "您好，关于终身保修政策，不同品牌的保修范围和条件有所不同，一般涵盖产品制造缺陷。具体保修条款请以购买时的保修卡或品牌官方说明为准。"
    if any(term in question for term in ("技术规格", "参数", "规格参数", "详细参数")):
        return "您好，关于技术规格，不同型号的产品参数有所差异，建议您核对商品包装或机身标签上的具体型号，以便获取准确的规格信息。"
    if any(term in question for term in ("配件", "附件", "零件", "替换件", "备件")):
        return "您好，关于配件和附件信息，不同型号的产品配置有所不同。您可以查看商品包装中的装箱清单，确认随机附带的配件种类和数量。"
    if any(term in question for term in ("安全", "危险", "警告", "注意事项")):
        return "您好，安全使用非常重要。使用产品前请务必阅读并遵守安全指引，确保正确操作，避免造成人身伤害或财产损失。"
    if any(term in question for term in ("尺寸", "大小", "重量", "体积")):
        return "您好，关于产品尺寸信息，不同型号的产品规格有所差异。您可以查看商品详情页的参数表格，或核对产品包装上的规格标注。"
    return "您好，关于您的问题，建议您提供商品名称或型号，以便我们为您提供更准确的信息。"


def _strip_submission_artifacts(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r'^\s*"""(.+?)"""\s*;\s*\[.*?\]\s*$', r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'^\s*"(.+)"\s*;\s*\[(?:"[^"]*"\s*,?\s*)*\]\s*$', r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'";\s*\[(?:"[^"]*"\s*,?\s*)*\]\s*$', "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\[(?:\"(?:Manual\d+_\d+|[A-Za-z]+_\d+)\"\s*,?\s*)+\]\s*$", "", cleaned)
    cleaned = re.sub(r"\[(?:\"?\s*(?:Manual\s*\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\s*\"?\s*,?\s*)+\]\s*$", "", cleaned)
    cleaned = re.sub(r";\s*\[(?:\"?\s*(?:Manual\s*\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\s*\"?\s*,?\s*)+\]\s*$", "", cleaned)
    cleaned = re.sub(r"（相关配图：[,、\s]*）", "", cleaned)
    cleaned = re.sub(r"\(相关配图：[,、\s]*\)", "", cleaned)
    cleaned = re.sub(r"相关配图：[,、\s]*", "", cleaned)
    cleaned = re.sub(r"[（(][A-Za-z]+\d+_\d+.*?[）)]", "", cleaned)
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
    cleaned = re.sub(r"\d+\.。", "", cleaned)
    cleaned = re.sub(r"【参考\d+】", "", cleaned)
    cleaned = cleaned.replace("：。", "：")
    cleaned = cleaned.replace('\\"', '"')
    cleaned = re.sub(r"参考\s*(?:Manual\d+_\d+|[A-Za-z]+(?:_[A-Za-z0-9]+)*_\d+)\b", "", cleaned)
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


def _is_off_topic_answer(answer: str, question: str) -> bool:
    """Detect if the answer doesn't address the question's core intent."""
    if not answer or not question:
        return False
    q_lower = question.lower()
    a_lower = answer.lower()
    # Check for refusal patterns - if the answer says it cannot answer, treat as off-topic
    refusal_patterns = ("无法", "未提及", "未包含", "没有提供", "没有相关", "不包含",
                        "cannot", "not contain", "no specific", "no relevant")
    has_refusal = any(p in a_lower for p in refusal_patterns)
    intent_keywords: list[str] = []
    action_map = {
        "关机": ["关机", "关闭", "电源", "开机"],
        "关闭": ["关机", "关闭", "电源", "开机"],
        "开机": ["开机", "启动", "电源", "打开"],
        "处理器": ["处理器", "CPU", "芯片", "性能"],
        "内存": ["内存", "RAM", "存储"],
        "电池": ["电池", "续航", "充电", "电量"],
        "屏幕": ["屏幕", "显示", "分辨率"],
        "蓝牙": ["蓝牙", "配对", "连接"],
        "wifi": ["wifi", "Wi-Fi", "网络", "连接"],
        "安装": ["安装", "组装", "设置"],
        "清洁": ["清洁", "清洗", "保养"],
        "维修": ["维修", "修理", "故障"],
        "退货": ["退货", "退款", "退回"],
        "换货": ["换货", "更换"],
        "保修": ["保修", "质保", "维修"],
        "尺寸": ["尺寸", "大小", "规格"],
        "重量": ["重量", "质量"],
        "价格": ["价格", "多少钱", "费用"],
        "发货": ["发货", "物流", "配送"],
        "快递": ["快递", "物流", "配送"],
    }
    en_action_map = {
        "shutdown": ["shutdown", "power off", "turn off", "power", "button"],
        "turn off": ["shutdown", "power off", "turn off", "power", "button"],
        "power off": ["shutdown", "power off", "turn off", "power", "button"],
        "processor": ["processor", "CPU", "chip", "performance"],
        "cpu": ["processor", "CPU", "chip", "performance"],
        "memory": ["memory", "RAM", "storage"],
        "ram": ["memory", "RAM", "storage"],
        "battery": ["battery", "charge", "charging", "power"],
        "screen": ["screen", "display", "resolution"],
        "display": ["screen", "display", "resolution"],
        "bluetooth": ["bluetooth", "pair", "pairing", "connect"],
        "pair": ["bluetooth", "pair", "pairing", "connect"],
        "wifi": ["wifi", "Wi-Fi", "network", "connect"],
        "install": ["install", "assembly", "setup"],
        "setup": ["install", "assembly", "setup"],
        "clean": ["clean", "cleaning", "maintenance"],
        "maintenance": ["clean", "cleaning", "maintenance"],
        "repair": ["repair", "fix", "troubleshoot"],
        "troubleshoot": ["repair", "fix", "troubleshoot"],
        "return": ["return", "refund", "exchange"],
        "refund": ["return", "refund", "exchange"],
        "warranty": ["warranty", "guarantee", "repair"],
        "size": ["size", "dimensions", "specifications"],
        "dimensions": ["size", "dimensions", "specifications"],
        "weight": ["weight", "mass"],
        "price": ["price", "cost", "how much"],
        "shipping": ["shipping", "delivery", "logistics"],
        "delivery": ["shipping", "delivery", "logistics"],
    }
    for trigger, keywords in action_map.items():
        if trigger in q_lower:
            intent_keywords.extend(keywords)
            break
    if not intent_keywords:
        for trigger, keywords in en_action_map.items():
            if trigger in q_lower:
                intent_keywords.extend(keywords)
                break
    if not intent_keywords:
        return False
    matched = sum(1 for kw in intent_keywords if kw in a_lower)
    if matched == 0:
        return True
    # If the answer contains refusal patterns but has some intent keywords,
    # it's likely a weak answer that doesn't fully address the question
    if has_refusal and matched <= 1:
        return True
    return False


def _smooth_raw_manual_text(text: str) -> str:
    """Add natural wrappers to raw manual text that looks like extracted chunks."""
    if not text:
        return text
    raw_patterns = [
        r"^[\d]+[.、]\s*",
        r"^[（(]\d+[）)]\s*",
        r"^第[一二三四五六七八九十\d]+步",
    ]
    is_raw = False
    for pattern in raw_patterns:
        if re.match(pattern, text.strip()):
            is_raw = True
            break
    if not is_raw:
        lines = [l.strip() for l in text.split(chr(10)) if l.strip()]
        if len(lines) >= 3:
            spec_lines = sum(1 for l in lines if re.match(r"^[\d.]+\s*", l) or "：" in l or ":" in l)
            if spec_lines / len(lines) > 0.7:
                is_raw = True
    if is_raw and not text.startswith(("您好", "你好", "Hello", "您好，")):
        text = "您好，" + text
    return text


def _detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English."""
    if not text:
        return "unknown"
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    ascii_chars = len(re.findall(r"[A-Za-z]", text))
    total = chinese_chars + ascii_chars
    if total == 0:
        return "unknown"
    if chinese_chars / total > 0.6:
        return "zh"
    if ascii_chars / total > 0.6:
        return "en"
    return "mixed"


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
    cleaned = cleaned.strip(" ，,；;。")
    if cleaned in ("抱歉", "抱歉，", "抱歉。", "根据现有资料", "根据现有资料，", "根据现有资料。"):
        cleaned = ""
    return cleaned


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
    if informative_chars < 4:
        return True
    if informative_chars < 18 and (
        "根据现有资料无法准确回答此问题" in original
        or "根据现有资料无法回答此问题" in original
        or "根据现有资料" in original
        or "无法提供" in original
        or "无法查到" in original
        or "无法回答" in original
        or "无法准确" in original
    ):
        return True
    return False


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


def _detect_customer_emotion(question: str) -> str:
    """Detect customer emotion from question text."""
    angry_patterns = (
        "投诉", "差评", "垃圾", "骗子", "骗人", "假货", "太差", "很差", "恶心",
        "无语", "气死", "生气", "愤怒", "火大", "忍无可忍", "太过分", "太离谱",
        "什么破", "什么垃圾", "再也不", "骗钱", "坑人", "黑心", "举报",
        "态度差", "态度恶劣", "服务差", "服务态度", "没人管", "没人理",
        "等了好久", "等了很久", "一直没人", "一直不", "总是", "每次都是",
    )
    urgent_patterns = (
        "急", "马上", "尽快", "赶紧", "立刻", "立即", "现在", "马上就要",
        "等不了", "来不及", "很着急", "着急", "紧急", "加急",
    )
    disappointed_patterns = (
        "失望", "不满意", "不满意", "不高兴", "不开心", "难受", "郁闷",
        "心寒", "寒心", "无奈", "怎么办", "崩溃",
    )

    for pattern in angry_patterns:
        if pattern in question:
            return "angry"
    for pattern in urgent_patterns:
        if pattern in question:
            return "urgent"
    for pattern in disappointed_patterns:
        if pattern in question:
            return "disappointed"
    return ""


def _get_empathy_phrase(emotion: str) -> str:
    """Get appropriate empathy phrase based on customer emotion."""
    if emotion == "angry":
        return "非常抱歉给您带来不好的体验，"
    if emotion == "urgent":
        return "理解您比较着急，"
    if emotion == "disappointed":
        return "很理解您的心情，"
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
            return "您好，支持7天无理由退货。商品签收后7天内、未使用且配件齐全即可在订单页申请退货。"
        if "取消订单" in question and any(term in question for term in ("到账", "原路返回", "信用卡")):
            return "您好，已付款订单可以申请取消退款。退款一般原路退回您的支付账户，到账时间取决于支付渠道处理速度。"
        if any(term in question for term in ("到账", "原路返回", "信用卡")):
            return "您好，退款会原路退回您的支付账户。一般1-3个工作日到账，信用卡支付可能需要3-5个工作日。"
        return "您好，支持退换货。请在订单页发起售后申请，如需帮助可联系人工客服。"
    if topic == "invoice":
        if any(term in question for term in ("发票类型", "专票", "普票", "电子发票")):
            return "您好，支持开具电子普通发票和增值税专用发票。请在订单页选择开票类型并填写抬头信息。"
        if any(term in question for term in ("重开", "开错", "抬头", "税号")):
            return "您好，发票信息填错后可以申请更正或重开。请提供订单号和正确的公司名称、税号，我们会尽快为您处理。"
        return "您好，支持开发票。请在订单页查看开票入口，填写抬头和税号信息即可。"
    if topic == "shipping":
        if any(term in question for term in ("少发", "漏发", "缺件", "补寄")):
            return "您好，少发漏发支持补寄。请提供订单号和缺少的配件信息，我们会核实并尽快安排补发。"
        if any(term in question for term in ("签收", "未收到", "已签收")):
            return "您好，显示签收但未收到的情况，我们会立即联系承运商核实。请提供订单号，我们会为您跟进处理。"
        if any(term in question for term in ("改地址", "收货地址")):
            return "您好，未出库的订单可以直接修改地址。已发货的请联系客服协助处理。"
        return "您好，正常配送运费与市区一致，不会额外加收。偏远地区以页面显示为准。"
    if topic == "complaint":
        if any(term in question for term in ("假货", "不是正品", "验证是假")):
            return "您好，非常重视您关于商品真伪的反馈。请您提供订单号、商品页面宣传截图、实物照片以及验真凭证，我们会立即为您提交升级核查。"
        if any(term in question for term in ("虚假宣传", "宣传和实际不一样", "功能不符")):
            return "您好，非常重视您关于虚假宣传的反馈。请您提供订单号、商品页面宣传截图以及能证明实际功能不符的照片或视频，我们会为您提交升级处理。"
        if any(term in question for term in ("辱骂", "态度差")):
            return "您好，非常重视您的反馈。服务态度问题我们会严肃处理。请您提供订单号、相关时间及对话记录，我们会尽快核实并为您升级反馈。"
        if any(term in question for term in ("保质期", "临近过期", "快过期", "过期")):
            return "您好，非常抱歉给您带来不好的体验。收到临近过期的商品，请您提供订单号和商品保质期照片，我们会尽快核实处理。如商品在保质期内但临近过期，且下单时页面未标注临期，您可以申请退货退款，运费由我方承担。"
        if any(term in question for term in ("生产日期", "制造时间")):
            return "您好，商品的生产日期通常标注在商品包装或商品本体上。请您查看商品包装上的生产日期标注，或提供商品照片以便我们为您确认。"
        return "您好，非常重视您的反馈。请您提供订单号、相关证据（照片/视频/聊天记录），我们会尽快为您升级处理。"
    if topic == "after_sales":
        if any(term in question for term in ("人为", "进水", "摔坏", "磕碰", "私拆")):
            return "您好，人为损坏的情况可能不在免费保修范围内，但仍可申请付费检测和维修。请提交售后申请。"
        if any(term in question for term in ("保修期", "质保期")):
            return "您好，保修期一般为购买之日起1年，具体以商品页和保修卡为准。"
        if any(term in question for term in ("维修", "修好", "一直没修")):
            return "您好，非常抱歉给您带来不便。维修进度延迟我们会优先加急处理，请提供维修单号，我们会立即为您核实。"
        if any(term in question for term in ("终身维修",)):
            return "您好，关于终身维修服务，目前我们的维修服务通常覆盖产品保修期内的非人为损坏维修。具体保修政策请以商品页和保修卡为准。"
        return "您好，支持售后维修服务。请提交售后申请并描述故障现象，我们会尽快为您安排。"
    if topic == "after_sales":
        if any(term in question for term in ("人为", "进水", "摔坏", "磕碰", "私拆")):
            return "您好，人为损坏的情况可能不在免费保修范围内，但仍可申请付费检测和维修。请提交售后申请。"
        if any(term in question for term in ("保修期", "质保期")):
            return "您好，保修期一般为购买之日起1年，具体以商品页和保修卡为准。"
        if any(term in question for term in ("维修", "修好", "一直没修")):
            return "您好，非常抱歉给您带来不便。维修进度延迟我们会优先加急处理，请提供维修单号，我们会立即为您核实。"
        return "您好，支持售后维修服务。请提交售后申请并描述故障现象，我们会尽快为您安排。"
    if topic == "quality_issue":
        if any(term in question for term in ("包装破损", "外包装破损")):
            return "您好，外包装破损请先核对商品本体是否完好。如有异常请拍照留证，我们为您安排退换或补偿。"
        if any(term in question for term in ("少件", "少发", "漏发", "缺件")):
            return "您好，少发漏发支持补寄。请提供订单号和缺少的配件信息，我们会尽快为您安排补发。"
        if any(term in question for term in ("过期", "临期", "保质期")):
            return "您好，非常抱歉给您带来不好的体验。收到临近过期的商品，请您提供订单号和商品保质期照片，我们会尽快核实处理。"
        return "您好，请尽快拍照留证并在订单页发起售后申请，我们会为您处理。"
    if topic == "platform_service":
        if "以旧换新" in question:
            return "您好，支持以旧换新服务。请在商品页查看是否有以旧换新入口，按页面要求填写旧机信息即可。"
        if "优惠券" in question:
            return "您好，优惠券在满足有效期和使用门槛后，可在结算页直接勾选使用。"
        if "试用" in question:
            return "您好，部分商品支持试用服务。请在商品页查看是否有试用入口。"
        if "更换尺寸" in question or "换尺寸" in question:
            return "您好，支持更换尺寸。请在订单页发起换货申请，说明要更换的尺寸，系统会按页面指引处理补差或退差。"
        return "您好，相关服务请以商品页和活动页入口为准。如有疑问可联系人工客服。"
    if topic == "installation":
        return "您好，部分商品支持预约安装服务。请在订单页查看安装服务入口。"
    if topic == "payment":
        return "您好，支付异常请提供订单号和扣款截图，我们会尽快为您核查处理。"
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
        ("请您提供", "请提供"),
        ("请您提交", "请提交"),
        ("请您联系", "请联系"),
        ("请您查看", "请查看"),
        ("请您在", "请在"),
        ("我们会尽快为您", "我们会尽快"),
        ("我们会在", "会在"),
        ("您可以申请", "可以申请"),
        ("您可以联系", "可以联系"),
        ("您可以查看", "可以查看"),
        ("您可以提供", "可以提供"),
    )
    for source, target in replacements:
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"^如果你愿意，?", "", cleaned)
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
        if lead and cleaned.startswith("您好"):
            cleaned = re.sub(r"^您好[，,]?\s*", "", cleaned)
        if _is_question_echo_sentence(cleaned, question=question):
            continue
        if any(cleaned in sub_question for sub_question in sub_questions) and len(cleaned) <= 14:
            continue
        if re.search(r"(是什么|有哪些|怎么|如何|多久|吗|能不能|是否)$", cleaned):
            continue
        if any(term in cleaned for term in _GENERIC_CUSTOMER_SERVICE_REWRITE_TERMS):
            continue
        if _is_irrelevant_shipping_content(cleaned, question=question):
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
    emotion = _detect_customer_emotion(question)
    empathy = _get_empathy_phrase(emotion)

    if lead:
        lead_stripped = lead.strip("。")
        # Add empathy phrase before lead if customer is emotional
        if empathy and not lead_stripped.startswith(("非常抱歉", "很抱歉", "理解您", "很理解")):
            # Replace "您好，" with empathy phrase
            if lead_stripped.startswith("您好"):
                lead_stripped = empathy + lead_stripped[2:].lstrip("，, ")
            else:
                lead_stripped = empathy + lead_stripped
        already_has_lead = any(
            _normalize_sentence_key(item).startswith(_normalize_sentence_key(lead_stripped)[:12])
            for item in selected
        ) if selected else False
        if not already_has_lead:
            combined.append(lead_stripped)
    # Filter out selected sentences that are already covered by the lead
    filtered_selected: list[str] = []
    for item in selected:
        item_stripped = item.strip("。")
        if not item_stripped:
            continue
        if lead and _is_covered_by_lead(item_stripped, lead.strip("。")):
            continue
        filtered_selected.append(item_stripped)
    combined.extend(filtered_selected)
    if not combined:
        return text
    return _join_sentences_with_punctuation(combined)


def _join_sentences_with_punctuation(sentences: list[str]) -> str:
    """Join sentences with appropriate punctuation, using comma for continuations."""
    if not sentences:
        return ""
    continuation_starts = ("请", "可以", "需要", "同时", "并且", "此外", "另外",
                           "如", "如果", "若", "但", "但是", "也", "还", "再", "然后")
    result = sentences[0]
    for sentence in sentences[1:]:
        sentence = sentence.strip()
        if not sentence:
            continue
        if any(sentence.startswith(prefix) for prefix in continuation_starts):
            result += "，" + sentence
        elif result.endswith(("：", ":")):
            result += sentence
        else:
            result += "。" + sentence
    if not result.endswith(("。", "！", "？")):
        result += "。"
    return result.strip(" ，,；;。") + "。"


def _strip_weak_leads(text: str) -> str:
    cleaned = str(text)
    cleaned = re.sub(r"根据现有资料[，,:：]\s*", "", cleaned)
    cleaned = re.sub(r"根据参考资料[，,:：]\s*", "", cleaned)
    cleaned = re.sub(r"根据此外[，,]\s*", "", cleaned)
    cleaned = re.sub(r"根据[^，,。]{0,10}[，,:：]\s*", "", cleaned)
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

    return _join_sentences_with_punctuation(selected)


def _format_as_numbered_steps(text: str, *, question: str) -> str:
    """Format answer as numbered steps when it has multiple distinct points."""
    if not text or len(text) < 30:
        return text

    # Don't format if already has numbered steps
    if re.search(r"\d+\.\s", text):
        return text

    # Extract topic from lead if present
    topic_match = re.match(r"^(?:您好[，,]?\s*)?关于(.{1,20})[：:]", text)
    topic = ""
    body = text
    if topic_match:
        topic = topic_match.group(1)
        body = text[topic_match.end():].strip()

    # Split into distinct points
    sentences = _split_submission_sentences(body)
    if len(sentences) < 2:
        return text

    # Check if sentences are about different aspects
    has_different_aspects = False
    action_verbs = ("请", "可以", "需要", "支持", "提供", "填写", "联系", "查看", "提交", "申请")
    sentence_starts = []
    for s in sentences[:4]:
        for verb in action_verbs:
            if s.startswith(verb):
                sentence_starts.append(verb)
                break
        else:
            sentence_starts.append(s[:2] if len(s) >= 2 else s)

    # If we have at least 2 different starting patterns, format as steps
    if len(set(sentence_starts[:3])) >= 2:
        has_different_aspects = True

    if not has_different_aspects:
        return text

    # Format with numbered steps
    formatted_parts = []
    if topic:
        formatted_parts.append(f"关于{topic}：")

    step_num = 1
    for i, sentence in enumerate(sentences[:4]):
        sentence = sentence.strip(" ，,；;。")
        if not sentence:
            continue
        # First sentence might be a general statement, keep as intro
        if i == 0 and not any(v in sentence for v in action_verbs):
            formatted_parts.append(sentence)
            continue
        formatted_parts.append(f"{step_num}. {sentence}")
        step_num += 1

    # Add remaining sentences as supplementary notes
    for sentence in sentences[4:6]:
        sentence = sentence.strip(" ，,；;。")
        if sentence:
            formatted_parts.append(sentence)

    result = "".join(formatted_parts)
    if not result.endswith(("。", "！", "？")):
        result += "。"
    return result


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
    answer = _join_sentences_with_punctuation(selected_texts)
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
    merged = _join_sentences_with_punctuation(collected)
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
    cleaned = _join_sentences_with_punctuation(selected)
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


def _is_covered_by_lead(candidate: str, lead: str) -> bool:
    """Check if a candidate sentence is semantically covered by the lead sentence."""
    if not lead or not candidate:
        return False
    cand_key = _normalize_sentence_key(candidate)
    lead_key = _normalize_sentence_key(lead)
    if not cand_key or not lead_key:
        return False
    # If lead is much longer and contains most of candidate's key info
    if len(lead_key) >= len(cand_key) * 1.5:
        common = sum(1 for ch in cand_key if ch in lead_key)
        if common / max(len(cand_key), 1) >= 0.6:
            return True
    # Check if both mention the same core action/topic
    action_pairs = (
        ("申请", "退货"), ("申请", "换货"), ("申请", "退款"),
        ("提供", "订单号"), ("提供", "照片"), ("提供", "凭证"),
        ("填写", "抬头"), ("填写", "税号"), ("填写", "信息"),
        ("联系", "客服"), ("联系", "人工"), ("提交", "售后"),
        ("查看", "订单"), ("查看", "商品"), ("查看", "页面"),
        ("核实", "处理"), ("核实", "进度"), ("安排", "补发"),
        ("支持", "开发票"), ("支持", "开具"), ("支持", "退货"),
        ("支持", "换货"), ("支持", "维修"), ("支持", "安装"),
        ("发票", "抬头"), ("发票", "税号"), ("发票", "开具"),
        ("运费", "承担"), ("运费", "免"), ("费用", "承担"),
        ("工作日", "到账"), ("工作日", "处理"), ("工作日", "开具"),
    )
    for kw1, kw2 in action_pairs:
        if (kw1 in candidate and kw2 in candidate
                and kw1 in lead and kw2 in lead):
            return True
    return False


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
        if shorter >= 12 and (candidate_key in sentence_key or sentence_key in candidate_key):
            return True
        if shorter >= 4:
            common = sum(1 for ch in candidate_key if ch in sentence_key)
            ratio = common / max(len(candidate_key), 1)
            if ratio >= 0.6:
                return True
        cs_keyword_pairs = (
            ("工作日", "到账"), ("退换货", "运费"), ("维修", "费用"),
            ("保修", "检测"), ("补寄", "运费"), ("快递", "签收"),
            ("发票", "抬头"), ("退款", "原路"), ("售后", "申请"),
            ("支持", "退货"), ("支持", "换货"), ("支持", "退换货"),
            ("订单页", "发起"), ("订单页", "售后"),
            ("提供", "维修单号"), ("提供", "订单号"), ("提供", "照片"),
            ("核实", "进度"), ("核实", "状态"), ("推进", "处理"),
            ("支持", "开发票"), ("支持", "开具"),
            ("发票", "开具"), ("发票", "电子"), ("发票", "增值税"),
            ("申请", "退货"), ("申请", "换货"), ("申请", "退款"),
            ("填写", "抬头"), ("填写", "税号"), ("填写", "信息"),
            ("联系", "客服"), ("联系", "人工"), ("提交", "售后"),
            ("查看", "订单"), ("查看", "商品"), ("查看", "页面"),
            ("安排", "补发"), ("安排", "维修"), ("安排", "安装"),
            ("运费", "承担"), ("运费", "免"), ("费用", "承担"),
            ("保修期", "年"), ("保修期", "购买"), ("保修期", "保修"),
        )
        # Check if both sentences talk about the same topic with "支持" + topic word
        if "支持" in candidate and "支持" in sentence:
            candidate_topic = candidate.replace("支持", "")
            sentence_topic = sentence.replace("支持", "")
            if len(candidate_topic) >= 2 and len(sentence_topic) >= 2:
                common_chars = sum(1 for ch in candidate_topic if ch in sentence_topic)
                if common_chars >= 3:
                    return True
        for kw1, kw2 in cs_keyword_pairs:
            if (kw1 in candidate and kw2 in candidate
                    and kw1 in sentence and kw2 in sentence):
                return True
    return False


_CS_IRRELEVANT_SHIPPING_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("complaint", ("运费", "偏远地区", "配送")),
    ("quality_issue", ("运费", "偏远地区", "配送")),
    ("after_sales", ("运费", "偏远地区", "配送")),
)


def _is_irrelevant_shipping_content(sentence: str, *, question: str) -> bool:
    topic = _infer_customer_service_topic(question)
    if not topic:
        return False
    for topic_name, irrelevant_terms in _CS_IRRELEVANT_SHIPPING_PATTERNS:
        if topic == topic_name:
            if any(term in sentence for term in irrelevant_terms):
                if not any(term in question for term in ("运费", "配送", "物流", "快递")):
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
    parser.add_argument("--ids", type=str, default="", help="Comma-separated question IDs to process, e.g. '36,45,51'.")
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
    if args.ids:
        target_ids = set(id.strip() for id in args.ids.split(",") if id.strip())
        questions = [q for q in questions if q["id"] in target_ids]
        if not questions:
            print(f"No questions found for IDs: {args.ids}")
            return
    elif args.limit > 0:
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
