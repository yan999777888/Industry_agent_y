"""Normalize answer style for manual QA and customer-service replies."""

from __future__ import annotations

import re


def format_manual_answer(answer: str, *, image_ids: list[str]) -> str:
    text = _strip_markdown(answer)
    if "根据现有资料无法回答此问题" in text:
        return _format_manual_fallback()

    if not any(label in text for label in ("结论：", "操作/说明：", "注意事项：")):
        body = text.strip() or "请参考说明书相关章节。"
        image_line = "、".join(image_ids[:4]) if image_ids else "无"
        return (
            "结论：\n"
            f"- {body}\n\n"
            "操作/说明：\n"
            f"- {body}\n\n"
            "注意事项：\n"
            "- 请以实际产品型号和说明书原文为准。\n\n"
            "相关图片：\n"
            f"- {image_line}"
        )

    text = _replace_related_images_section(text, image_ids=image_ids)
    if "相关图片：" not in text:
        image_line = "、".join(image_ids[:4]) if image_ids else "无"
        text = f"{text.rstrip()}\n\n相关图片：\n- {image_line}"
    return text.strip()


def format_customer_service_answer(answer: str) -> str:
    text = _strip_markdown(answer)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith(("。", "！", "？")):
        text += "。"
    return text


def _format_manual_fallback() -> str:
    return (
        "结论：\n"
        "- 根据现有资料无法准确回答此问题。\n\n"
        "操作/说明：\n"
        "- 请补充产品名称、型号、故障现象或上传更清晰的图片后再试。\n\n"
        "注意事项：\n"
        "- 回答仅基于当前知识库中的说明书资料。\n\n"
        "相关图片：\n"
        "- 无"
    )


def _replace_related_images_section(text: str, *, image_ids: list[str]) -> str:
    image_line = "、".join(image_ids[:4]) if image_ids else "无"
    pattern = re.compile(r"相关图片：[\s\S]*$", flags=re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(f"相关图片：\n- {image_line}", text)
    return text


def _strip_markdown(text: str) -> str:
    cleaned = text.replace("**", "").strip()
    return re.sub(r"[ \t]+\n", "\n", cleaned)
