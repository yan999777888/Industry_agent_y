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


def normalize_submission_answer(answer: str, *, question: str, sources: list[str] | None = None) -> str:
    sources = sources or []
    text = answer.strip()
    if not text:
        return DEFAULT_FALLBACK_ANSWER

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
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n+", " ", text).strip(" |;；，,")

    text_without_fallback = _strip_fallback_sentences(text)
    text_without_fallback = _remove_question_echo(text_without_fallback, question=question)
    if _looks_like_pure_fallback(text, text_without_fallback):
        return _build_submission_fallback(question=question, sources=sources)
    text = text_without_fallback or text

    if "customer_service_policy" in sources:
        text = re.sub(r"如果你愿意，我建议[^。]*。?", "", text)
        text = re.sub(r"如果你愿意，我建议下一步优先补充[^。]*。?", "", text)
        text = re.sub(r"这类问题更适合按通用客服流程处理。?", "", text)
        text = _compress_customer_service_answer(text)

    text = re.sub(r"\s{2,}", " ", text).strip(" ，,；;")
    if not text.endswith(("。", "！", "？")):
        text += "。"
    return text


def _build_submission_fallback(*, question: str, sources: list[str]) -> str:
    if "customer_service_policy" in sources or any(keyword in question for keyword in _CUSTOMER_SERVICE_KEYWORDS):
        return "您好，相关情况需要结合订单信息、商品情况和售后规则进一步核实。请您补充订单号、商品名称、问题照片或聊天记录，我们会继续为您处理。"
    return "您好，当前还无法准确定位对应的说明书内容。请补充产品名称、型号、故障现象或图片，我再继续帮您查询。"


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
    args = parser.parse_args()

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
            answer = normalize_submission_answer(
                raw_answer,
                question=item["question"],
                sources=list(data.get("sources", []) or []),
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
