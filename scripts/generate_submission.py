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
_CUSTOMER_SERVICE_KEYWORDS = (
    "退货", "换货", "退款", "运费", "物流", "快递", "发票", "补发", "签收",
    "售后", "维修", "保修", "投诉", "赔偿", "订单", "发货", "包装", "瑕疵",
    "少件", "划痕", "假货", "虚假宣传", "国外", "乡镇",
)
_IMAGE_ID_RE = re.compile(r"\b(?:Manual\d+_\d+|drill\d*_?\d+|pump_\d+|generator_\d+)\b")
_RELATED_IMAGE_SECTION_RE = re.compile(r"\n*相关图片：(?:\n[^\n]*)*", flags=re.IGNORECASE)
_LABEL_REPLACEMENTS = (
    ("问题1：", ""),
    ("问题 1：", ""),
    ("问题2：", ""),
    ("问题 2：", ""),
    ("问题3：", ""),
    ("问题 3：", ""),
    ("回答：", ""),
    ("结论：", ""),
    ("操作/说明：", ""),
    ("注意事项：", ""),
)
_FALLBACK_SENTENCE_PATTERNS: tuple[str, ...] = (
    r"根据现有资料无法准确回答此问题[。]?",
    r"根据现有资料无法回答此问题[。]?",
    r"请补充更明确的产品名称、型号、故障现象或图片后再试[。]?",
    r"请补充产品名称、型号、故障现象或上传更清晰的图片后再试[。]?",
    r"当前回答仅基于知识库中的说明书资料，请以实际产品和原文为准[。]?",
)
_INTERNAL_SENTENCE_PATTERNS: tuple[str, ...] = (
    r"The answer is extracted from the retrieved manual evidence\.?",
    r"Please follow the original manual for safety-critical operation\.?",
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
    text = answer.strip()
    if not text:
        return _format_with_images(DEFAULT_FALLBACK_ANSWER, [])

    for old, new in _LABEL_REPLACEMENTS:
        text = text.replace(old, new)
    text = text.replace("**", "")
    text = _RELATED_IMAGE_SECTION_RE.sub("", text)
    text = _IMAGE_ID_RE.sub("", text)
    text = text.replace("- 无", "")
    text = text.replace("- ", "")
    text = re.sub(r"参考资料[^\n。]*[。]?", "", text)
    text = re.sub(r"当前资料[^\n。]*[。]?", "", text)
    text = re.sub(r"资料中仅[^\n。]*[。]?", "", text)
    text = re.sub(r"\[参考\s*\d*\]", "", text)
    text = re.sub(r"参考\s*\[\d+\]", "", text)
    text = re.sub(r"（参考\s*[^\）]*）", "", text)
    text = re.sub(r"\(参考\s*[^\)]*\)", "", text)
    text = _strip_internal_sentences(text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n+", " ", text).strip(" |;；，,")

    text_without_fallback = _strip_fallback_sentences(text)
    text_without_fallback = _remove_question_echo(text_without_fallback, question=question)
    if _looks_like_pure_fallback(text, text_without_fallback):
        reference_answer = _build_reference_based_answer(question=question, references=references)
        if reference_answer:
            return _format_with_images(reference_answer, image_ids)
        fb = _build_submission_fallback(question=question, sources=sources)
        return _format_with_images(fb, [])
    text = text_without_fallback or text

    if "customer_service_policy" in sources:
        text = re.sub(r"如果你愿意，我建议[^。]*。?", "", text)
        text = re.sub(r"如果你愿意，我建议下一步优先补充[^。]*。?", "", text)
        text = re.sub(r"这类问题更适合按通用客服流程处理。?", "", text)
        text = _compress_customer_service_answer(text)

    text = re.sub(r"\s{2,}", " ", text).strip(" ，,；;")
    if not text.endswith(("。", "！", "？")):
        text += "。"
    return _format_with_images(text, image_ids)


def _format_with_images(text: str, image_ids: list[str]) -> str:
    """Append image IDs in the competition submission format: "answer";["id1","id2"]."""
    if not image_ids:
        return text
    # Add <PIC> markers for image-text complementarity scoring
    if "<PIC>" not in text:
        pic_markers = "<PIC>" * len(image_ids)
        text = text.rstrip("。！？.!?") + pic_markers
        if not text.endswith(("。", "！", "？", ".", "!", "?")):
            text += "。"
    ids_json = json.dumps(image_ids, ensure_ascii=False)
    return f'"{text}";{ids_json}'


def _build_submission_fallback(*, question: str, sources: list[str]) -> str:
    if "customer_service_policy" in sources or any(keyword in question for keyword in _CUSTOMER_SERVICE_KEYWORDS):
        return "您好，相关情况需要结合订单信息、商品情况和售后规则进一步核实。请您补充订单号、商品名称、问题照片或聊天记录，我们会继续为您处理。"
    return "您好，当前还无法准确定位对应的说明书内容。请补充产品名称、型号、故障现象或图片，我再继续帮您查询。"


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
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"^[，,；;。:：\s]+", "", cleaned)
    cleaned = re.sub(r"\s+[，,；;。:：]", "", cleaned)
    return cleaned.strip(" ，,；;。")


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
    candidates: list[tuple[int, str]] = []
    for ref in references[:4]:
        title = _clean_reference_text(str(ref.get("title", "")))
        snippet = _clean_reference_text(str(ref.get("text_snippet", "")))
        if title and not _looks_like_reference_noise(title):
            candidates.append((_reference_overlap_score(title, query_terms) + 2, title))
        for sentence in _split_submission_sentences(snippet):
            sentence = _clean_reference_text(sentence)
            if len(sentence) < 12 or _looks_like_reference_noise(sentence):
                continue
            candidates.append((_reference_overlap_score(sentence, query_terms), sentence))

    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    selected: list[str] = []
    for score, candidate in candidates:
        if score <= 0 and selected:
            continue
        if _is_near_duplicate_sentence(candidate, selected):
            continue
        selected.append(candidate)
        if len(selected) >= 3:
            break

    if not selected:
        return ""

    is_english_question = bool(re.search(r"[A-Za-z]", question)) and not bool(re.search(r"[\u4e00-\u9fff]", question))
    if is_english_question:
        if len(selected) == 1:
            answer = selected[0]
        else:
            answer = selected[0] + " " + " ".join(selected[1:])
        return answer.strip(" ，,；;。") + "."

    lines = [selected[0]]
    if len(selected) > 1:
        lines.append("操作要点：" + "；".join(selected[1:]))
    return "。".join(line.strip("。") for line in lines if line.strip()) + "。"


def _clean_reference_text(text: str) -> str:
    cleaned = str(text)
    cleaned = cleaned.replace("#", " ")
    cleaned = re.sub(r"\[\[PIC[^\]]*\]\]", " ", cleaned)
    cleaned = re.sub(r"<PIC>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\u[0-9a-fA-F]{4}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -|，,；;。")


def _looks_like_reference_noise(text: str) -> bool:
    if re.search(r"\.{5,}|…{3,}", text):
        return True
    if len(re.findall(r"\b\d+\b", text)) >= 8:
        return True
    return False


def _extract_question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", question):
        if len(word) >= 3:
            terms.append(word.lower())
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


def _compress_customer_service_answer(text: str) -> str:
    sentences = _split_submission_sentences(text)
    if not sentences:
        return text

    selected: list[str] = []
    for sentence in sentences:
        cleaned = sentence.strip(" ，,；;。")
        if len(cleaned) < 8:
            continue
        if cleaned.endswith(("？", "?")):
            continue
        if _is_near_duplicate_sentence(cleaned, selected):
            continue
        selected.append(cleaned)
        if len(selected) >= 5:
            break

    return "。 ".join(selected).strip()


def _split_submission_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[。！？!?])\s+|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])", normalized)
    return [part.strip() for part in parts if part.strip()]


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
