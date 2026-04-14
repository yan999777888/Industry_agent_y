"""Agent orchestration: retrieve context -> build prompt -> call LLM."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from industry_agent.config import settings
from industry_agent.rag.retriever import SQLiteRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRIEVAL_LIMIT = 8         # chunks to retrieve before evidence filtering
FINAL_CONTEXT_CHUNKS = 4    # chunks passed into the LLM
MAX_CONTEXT_CHARS = 4000    # truncate context to fit model window
MAX_HISTORY_TURNS = 5       # keep last N turns per session
MIN_TOP_SCORE = 10.0        # below this, do not ask LLM to hallucinate
MIN_KEEP_SCORE = 8.0        # chunks below this score are discarded

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")

SYSTEM_TEMPLATE = """\
你是一个专业的工业产品客服智能体。请严格遵守以下规则：

1. **只基于下方【参考资料】回答**，不得编造任何信息。
2. 如果参考资料不足以回答问题，请明确说明"根据现有资料无法回答此问题"。
3. 如果用户一次问了多个问题，请拆成多个小标题逐一回答。
4. 回答格式固定为：
   - 结论：
   - 操作/说明：
   - 注意事项：
   - 相关图片：
5. 相关图片只能使用参考资料中出现的图片 ID，不要编造图片 ID。
6. 直接给出回答，不要输出任何思考过程。

【参考资料】
{context}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChatRequest:
    question: str
    images: list[str] | None = None
    session_id: str | None = None


@dataclass
class ChatResponse:
    answer: str
    image_ids: list[str]
    images: list[dict[str, str | bool]]
    sources: list[str]
    references: list[dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    retrieval_debug: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_SESSION_STORE: dict[str, list[dict[str, str]]] = {}


# ---------------------------------------------------------------------------
# Helper functions (defined before the class that uses them)
# ---------------------------------------------------------------------------

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ANSWER_START_RE = re.compile(
    r"^([\u4e00-\u9fff]|#{1,3}\s|根据|您好|以下|关于|\d+[\.、])"
)


def _strip_thinking(text: str) -> str:
    """Remove thinking/reasoning blocks that some models (qwen3) emit.

    Handles both:
      - <think>...</think> XML blocks
      - "Thinking Process:\n..." free-form prefix (stops at first Chinese or
        markdown answer section)
    """
    # 1. Strip <think>...</think> blocks
    text = _THINK_TAG_RE.sub("", text).strip()

    # 2. Strip "Thinking Process:" free-form prefix
    if text.startswith("Thinking"):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and _ANSWER_START_RE.match(stripped):
                preceding = "\n".join(lines[:i])
                if len(preceding) > 100:
                    text = "\n".join(lines[i:]).strip()
                    break

    return text


def _filter_evidence(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep high-confidence, same-product evidence for cleaner prompts."""
    if not chunks:
        return []

    top = chunks[0]
    top_score = float(top.get("_score", 0.0))
    if top_score < MIN_TOP_SCORE:
        return []

    top_product = top.get("product_name", "")
    filtered: list[dict[str, Any]] = []
    for chunk in chunks:
        score = float(chunk.get("_score", 0.0))
        if score < MIN_KEEP_SCORE:
            continue
        if top_product and chunk.get("product_name") != top_product:
            continue
        filtered.append(chunk)
        if len(filtered) >= FINAL_CONTEXT_CHUNKS:
            break
    return filtered


def _confidence_from_chunks(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.15
    top_score = float(chunks[0].get("_score", 0.0))
    second_score = float(chunks[1].get("_score", 0.0)) if len(chunks) > 1 else 0.0
    confidence = 0.35 + min(top_score / 50.0, 0.45) + min(max(top_score - second_score, 0.0) / 40.0, 0.15)
    return round(min(confidence, 0.95), 2)


def _assemble_context(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[str], list[str], list[dict[str, str]]]:
    """Build context string, collect image IDs, sources, and references."""
    parts: list[str] = []
    all_image_ids: list[str] = []
    seen_images: set[str] = set()
    sources: list[str] = []
    seen_sources: set[str] = set()
    references: list[dict[str, str]] = []
    total_chars = 0

    for idx, chunk in enumerate(chunks, start=1):
        product = chunk.get("product_name", "")
        title = chunk.get("title", "")
        text = chunk.get("text", "")

        # Parse image_ids (stored as JSON string in SQLite)
        raw_img = chunk.get("image_ids", "[]")
        img_ids = _parse_json_list(raw_img)

        # Collect unique image IDs
        for img_id in img_ids:
            if img_id and img_id not in seen_images:
                seen_images.add(img_id)
                all_image_ids.append(img_id)

        # Collect unique sources
        if product and product not in seen_sources:
            seen_sources.add(product)
            sources.append(product)

        # Build context part
        score = chunk.get("_score", "")
        header = f"[参考{idx}] 产品：{product} | 章节：{title} | 检索分：{score}"
        body = text.strip()
        if img_ids:
            body += f"\n（相关配图：{', '.join(img_ids)}）"
        part = f"{header}\n{body}"

        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                parts.append(part[:remaining] + "……")
            break
        parts.append(part)
        total_chars += len(part)

        # Reference snippet for response metadata
        references.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "title": title,
            "text_snippet": text[:100],
            "product_name": product,
            "score": str(chunk.get("_score", "")),
        })

    context = "\n\n".join(parts)
    return context, all_image_ids, sources, references


def _parse_json_list(value: Any) -> list[str]:
    """Safely parse a JSON-encoded list or return as-is if already a list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _load_image_index() -> dict[str, dict[str, str | bool]]:
    """Load image metadata generated by the knowledge-base builder."""
    image_index_path = settings.processed_dir / "images.jsonl"
    records: dict[str, dict[str, str | bool]] = {}
    if not image_index_path.exists():
        return records
    with image_index_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            image_id = str(item.get("image_id", ""))
            if not image_id:
                continue
            records[image_id] = {
                "image_id": image_id,
                "file_name": item.get("file_name") or "",
                "path": item.get("path") or "",
                "exists": bool(item.get("exists", False)),
            }
    return records


def _image_details(image_ids: list[str], image_index: dict[str, dict[str, str | bool]]) -> list[dict[str, str | bool]]:
    details: list[dict[str, str | bool]] = []
    for image_id in image_ids:
        if image_id in image_index:
            details.append(image_index[image_id])
            continue
        fallback_path = settings.image_dir / f"{image_id}.jpg"
        details.append({
            "image_id": image_id,
            "file_name": fallback_path.name,
            "path": str(fallback_path.relative_to(settings.project_root)) if fallback_path.exists() else "",
            "exists": fallback_path.exists(),
        })
    return details


# ---------------------------------------------------------------------------
# Agent service
# ---------------------------------------------------------------------------

class AgentService:
    """Retrieve -> assemble context -> call Ollama LLM -> return answer."""

    def __init__(
        self,
        retriever: SQLiteRetriever | None = None,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
    ) -> None:
        self.retriever = retriever or SQLiteRetriever()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.http_client = httpx.Client(proxy=None, timeout=120.0)
        self.image_index = _load_image_index()

    def generate_response(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        image_input: str | None = None,
    ) -> dict[str, Any]:
        # 1. Retrieve
        chunks = self.retriever.search(query, limit=RETRIEVAL_LIMIT)
        evidence_chunks = _filter_evidence(chunks)

        if not evidence_chunks:
            return {
                "answer": "根据现有资料无法回答此问题。请补充更明确的产品名称、型号、故障现象或图片后再试。",
                "image_ids": [],
                "images": [],
                "sources": [],
                "references": [],
                "confidence": 0.15,
                "retrieval_debug": {
                    "retrieved_count": len(chunks),
                    "top_score": chunks[0].get("_score", 0) if chunks else 0,
                    "top_title": chunks[0].get("title", "") if chunks else "",
                    "reason": "low_confidence_or_no_evidence",
                },
            }

        # 2. Assemble context / collect metadata
        context, image_ids, sources, references = _assemble_context(evidence_chunks)
        images = _image_details(image_ids, self.image_index)
        confidence = _confidence_from_chunks(evidence_chunks)

        # 3. Build messages
        system_msg = SYSTEM_TEMPLATE.format(context=context if context else "（未找到相关资料）")
        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]

        # Append conversation history (if any)
        if history:
            messages.extend(history[-MAX_HISTORY_TURNS * 2 :])

        messages.append({"role": "user", "content": query})

        # 4. Call LLM
        answer = self._call_llm(messages)

        return {
            "answer": answer,
            "image_ids": image_ids,
            "images": images,
            "sources": sources,
            "references": references,
            "confidence": confidence,
            "retrieval_debug": {
                "retrieved_count": len(chunks),
                "evidence_count": len(evidence_chunks),
                "top_score": evidence_chunks[0].get("_score", 0) if evidence_chunks else 0,
                "top_title": evidence_chunks[0].get("title", "") if evidence_chunks else "",
                "top_product": evidence_chunks[0].get("product_name", "") if evidence_chunks else "",
            },
        }

    def chat(self, request: ChatRequest) -> ChatResponse:
        """High-level chat with session memory."""
        history = _SESSION_STORE.get(request.session_id or "", [])

        result = self.generate_response(
            query=request.question,
            history=history,
            image_input=request.images[0] if request.images else None,
        )

        # Update session
        if request.session_id:
            hist = _SESSION_STORE.setdefault(request.session_id, [])
            hist.append({"role": "user", "content": request.question})
            hist.append({"role": "assistant", "content": result["answer"]})
            if len(hist) > MAX_HISTORY_TURNS * 2:
                _SESSION_STORE[request.session_id] = hist[-MAX_HISTORY_TURNS * 2 :]

        return ChatResponse(
            answer=result["answer"],
            image_ids=result["image_ids"],
            images=result["images"],
            sources=result["sources"],
            references=result["references"],
            confidence=result["confidence"],
            retrieval_debug=result["retrieval_debug"],
        )

    # ------------------------------------------------------------------
    # LLM call — Ollama native /api/chat (supports think=false)
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        try:
            resp = self.http_client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "think": False,           # disable qwen3 thinking mode
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1024,  # max output tokens
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            content = _strip_thinking(content)
            return content.strip() if content.strip() else "模型未返回有效回答。"
        except Exception as exc:
            return f"LLM 调用失败: {exc}"
