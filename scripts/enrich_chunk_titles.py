"""Generate concise chunk titles via Mimo multimodal LLM.

Reads chunks.jsonl, for each chunk sends text (+ associated images) to Mimo
to generate a short descriptive title (≤20 chars).  Writes updated titles back
to chunks.jsonl so that rebuild_index_from_jsonl.py can pick them up.

Usage:
    python scripts/enrich_chunk_titles.py

Resume after interruption:
    python scripts/enrich_chunk_titles.py          # auto-skips done chunks

After completion:
    python scripts/rebuild_index_from_jsonl.py     # sync SQLite + rebuild indexes
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from industry_agent.config import settings

# ── Mimo API — MUST use Mimo directly, NOT DashScope (even if dashscope_enabled=1) ──
# LLMClient auto-routes to DashScope when dashscope_enabled=True, so we bypass it.
MIMO_API_KEY = settings.llm_api_key
MIMO_BASE_URL = settings.llm_base_url
MIMO_MODEL = "mimo-v2.5"

# ── Paths ──────────────────────────────────────────────────────────────────
CHUNKS_PATH = settings.processed_dir / "chunks.jsonl"
IMAGES_JSONL_PATH = settings.processed_dir / "images.jsonl"
IMAGE_DIR = settings.image_dir
PROGRESS_PATH = settings.processed_dir / "chunks_title_progress.json"

MAX_WORKERS = int(os.getenv("TITLE_WORKERS", "3"))
MAX_RETRIES = 5
SAVE_INTERVAL = 50  # save progress every N chunks

# ── Prompts ────────────────────────────────────────────────────────────────

TEXT_ONLY_SYSTEM_PROMPT = (
    "你是一个产品说明书标题优化专家。根据产品手册段落内容生成简洁准确的中文段落标题。"
)

TEXT_ONLY_PROMPT = """根据以下产品手册段落内容，生成一个简洁准确的中文段落标题。

要求：
- 标题简短，包含具体关键词（型号、部件、操作等）
- 不要笼统概括，不要用"注意事项"、"说明"、"警告"这类通用词
- 突出段落核心主题
- 格式：[TITLE]标题[/TITLE]（替换"标题"为实际内容，不要保留"标题"二字）

段落内容：
{chunk_text}"""

EN_TEXT_ONLY_SYSTEM_PROMPT = (
    "You are a product manual title optimization expert. Generate concise English section titles based on the manual paragraph content."
)

EN_TEXT_ONLY_PROMPT = """Generate a concise English section title with specific keywords from the following product manual paragraph.

Requirements:
- Short and specific title
- Include exact model numbers, component names, or operations from the text
- NOT generic categories like "Compliance", "Warning", "Setup"
- Output [TITLE]title[/TITLE]

Paragraph content:
{chunk_text}"""

VISION_SYSTEM_PROMPT = (
    "你是一个产品说明书标题优化专家。根据产品手册段落文本和配图生成简洁准确的中文段落标题。"
)

VISION_PROMPT = """根据以下产品手册段落文本和配图，生成一个简洁准确的中文段落标题。

要求：
- 标题简短，包含具体关键词（型号、部件、操作等）
- 不要笼统概括，不要用"注意事项"、"说明"、"警告"这类通用词
- 突出段落核心主题
- 配图中的信息可用于辅助理解段落核心内容
- 格式：[TITLE]标题[/TITLE]（替换"标题"为实际内容，不要保留"标题"二字）

段落内容：
{chunk_text}"""

EN_VISION_SYSTEM_PROMPT = (
    "You are a product manual title optimization expert. Generate concise English section titles based on the manual paragraph text and accompanying images."
)

EN_VISION_PROMPT = """Generate a concise English section title with specific keywords from the following product manual paragraph text and accompanying images.

Requirements:
- Short and specific title
- Include exact model numbers, component names, or operations from the text
- NOT generic categories like "Compliance", "Warning", "Setup"
- Use images to help understand the content
- Output [TITLE]title[/TITLE]

Paragraph content:
{chunk_text}"""

# ── Helpers ────────────────────────────────────────────────────────────────


def load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_image_index(path: Path) -> dict[str, dict]:
    """Load images.jsonl -> {image_id: metadata}."""
    index: dict[str, dict] = {}
    if not path.exists():
        return index
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            iid = str(rec.get("image_id", ""))
            if iid:
                index[iid] = rec
    return index


def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("done_ids", []))


def save_checkpoint(path: Path, done_ids: set[str]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done_ids": sorted(done_ids)}, f, ensure_ascii=False)
    os.replace(str(tmp), str(path))


def resolve_image_path(image_id: str, image_index: dict) -> str | None:
    """Find the actual file path for an image_id.

    Checks images.jsonl path first, falls back to IMAGE_DIR/<id>.png/jpg.
    """
    rec = image_index.get(image_id)
    if rec:
        raw_path = rec.get("path") or ""
        if raw_path:
            p = Path(raw_path)
            if p.is_absolute():
                if p.exists():
                    return str(p)
            else:
                full = PROJECT_ROOT / raw_path
                if full.exists():
                    return str(full)
                full = IMAGE_DIR / raw_path
                if full.exists():
                    return str(full)
        # Try file_name in IMAGE_DIR
        fname = rec.get("file_name") or ""
        if fname:
            full = IMAGE_DIR / fname
            if full.exists():
                return str(full)

    # Last resort: try IMAGE_DIR / <id>.png and .jpg
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        full = IMAGE_DIR / f"{image_id}{ext}"
        if full.exists():
            return str(full)
    return None


def encode_image(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return base64.b64encode(raw).decode("ascii")
    except Exception:
        return None


def _is_english_text(text: str) -> bool:
    """Check if text is primarily English (vs Chinese)."""
    if not text or not text.strip():
        return False
    import re
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    english_words = len(re.findall(r'[a-zA-Z]{2,}', text))
    if chinese_chars == 0 and len(re.findall(r'[a-zA-Z]', text)) > 0:
        return True
    return english_words >= 2 and chinese_chars < english_words


def generate_title_text(
    chunk: dict,
    image_index: dict,
    openai_client: OpenAI,
) -> str | None:
    """Call Mimo (text or vision) to generate a title for this chunk.

    Returns the new title string, or None on failure.
    """
    product = chunk.get("product_name", "")
    # Skip generic manual labels that aren't real product names
    if product in ("汇总英文", "汇总英文手册", "英文汇总", "All Chinese Manuals"):
        product = ""
    chunk_text = chunk.get("text", "")
    # Truncate long text — title only needs the beginning to understand topic
    if len(chunk_text) > 600:
        chunk_text = chunk_text[:600] + "..."

    image_ids = chunk.get("image_ids", [])
    if isinstance(image_ids, str):
        try:
            image_ids = json.loads(image_ids)
        except (json.JSONDecodeError, TypeError):
            image_ids = []

    is_en = _is_english_text(chunk_text)
    has_images = bool(image_ids)

    if has_images:
        title = _call_vision_api(product, chunk_text, image_ids, image_index, openai_client, is_en=is_en)
        if title is None:
            return _call_text_api(product, chunk_text, openai_client, is_en=is_en)
        return title
    else:
        return _call_text_api(product, chunk_text, openai_client, is_en=is_en)


def _extract_title(raw: str) -> str:
    """Extract title from model output using [TITLE] markers, take LAST match, fallback to last line."""
    import re
    matches = list(re.finditer(r'\[TITLE\](.*?)\[/TITLE\]', raw, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    # fallback: last non-empty line
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def _get_content(message) -> str:
    """Extract title from response message, with reasoning_content fallback."""
    raw = message.content or ""
    if not raw and message.model_extra:
        raw = message.model_extra.get("reasoning_content", "") or ""
    return _extract_title(raw)


def _safe_text(text: str) -> str:
    """Escape curly braces in user-provided text so str.format doesn't choke."""
    return text.replace("{", "{{").replace("}", "}}")


def _call_text_api(product: str, chunk_text: str, openai_client: OpenAI, *, is_en: bool = False) -> str | None:
    prompt = EN_TEXT_ONLY_PROMPT if is_en else TEXT_ONLY_PROMPT
    sys_prompt = EN_TEXT_ONLY_SYSTEM_PROMPT if is_en else TEXT_ONLY_SYSTEM_PROMPT
    user_msg = prompt.format(
        product=_safe_text(product),
        chunk_text=_safe_text(chunk_text),
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = openai_client.chat.completions.create(
                model=MIMO_MODEL,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=2048,
                top_p=0.9,
            )
            raw = _get_content(resp.choices[0].message)
            title = raw.strip()
            if title:
                return _clean_title(title)
            print(f"  [TEXT_EMPTY] attempt={attempt} prod={product[:20]!r} model={MIMO_MODEL}", flush=True)
        except Exception as exc:
            last_exc = exc
            is_429 = "429" in str(exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 if is_429 else 1.5**attempt)
    if last_exc is not None:
        print(f"  [TEXT_API_ERR] {last_exc}", flush=True)
    return None


def _build_vision_messages(
    product: str,
    chunk_text: str,
    image_ids: list,
    image_index: dict,
    *,
    is_en: bool = False,
) -> list[dict] | None:
    """Build multimodal messages with text + up to 2 images.

    Returns None if no images could be loaded (caller should fall back to text).
    """
    sys_prompt = EN_VISION_SYSTEM_PROMPT if is_en else VISION_SYSTEM_PROMPT
    vision_prompt = EN_VISION_PROMPT if is_en else VISION_PROMPT
    user_content: list[dict] = []
    user_content.append({
        "type": "text",
        "text": vision_prompt.format(
            product=_safe_text(product),
            chunk_text=_safe_text(chunk_text),
        ),
    })

    loaded = 0
    for img_id in image_ids[:2]:  # max 2 images per chunk
        img_path = resolve_image_path(img_id, image_index)
        if not img_path:
            continue
        b64 = encode_image(img_path)
        if not b64:
            continue
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        loaded += 1

    if loaded == 0:
        return None  # no usable images, fall back to text

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]


def _call_vision_api(
    product: str,
    chunk_text: str,
    image_ids: list,
    image_index: dict,
    openai_client: OpenAI,
    *,
    is_en: bool = False,
) -> str | None:
    messages = _build_vision_messages(product, chunk_text, image_ids, image_index, is_en=is_en)
    if messages is None:
        return _call_text_api(product, chunk_text, openai_client, is_en=is_en)

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = openai_client.chat.completions.create(
                model=MIMO_MODEL,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=2048,
                top_p=0.9,
            )
            title = _get_content(resp.choices[0].message).strip()
            if title:
                return _clean_title(title)
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            # Don't retry 4xx errors (model doesn't support vision)
            if "404" in exc_str or "400" in exc_str:
                break
            is_429 = "429" in exc_str
            if attempt < MAX_RETRIES - 1:
                time.sleep(5 if is_429 else 1.5**attempt)
    # Vision failed — silently fall back to text-only
    return None


def _clean_title(title: str) -> str:
    """Strip quotes, whitespace, and trailing punctuation."""
    title = title.strip().strip('"').strip("'").strip()
    return title


def save_chunks(path: Path, chunks: list[dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(path))


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    # 1. Load data
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  Loaded {len(chunks)} chunks")

    print(f"Loading image index from {IMAGES_JSONL_PATH}...")
    image_index = load_image_index(IMAGES_JSONL_PATH)
    print(f"  Loaded {len(image_index)} image records")

    done_ids = load_checkpoint(PROGRESS_PATH)
    print(f"  Already processed: {len(done_ids)} chunks")

    # 2. Collect todo list
    todo = [(i, c) for i, c in enumerate(chunks) if c.get("chunk_id") not in done_ids]
    print(f"  Remaining: {len(todo)} chunks")

    if not todo:
        print("All chunks already processed! Run rebuild_index_from_jsonl.py to sync.")
        return

    # 3. Initialize OpenAI client — must point at Mimo, NOT DashScope
    # Mimo uses a custom `api-key` header (NOT `Authorization: Bearer`),
    # so we pass a dummy api_key and inject the header via a custom httpx client.
    from openai import OpenAI
    import httpx
    openai_client = OpenAI(
        api_key="",
        base_url=MIMO_BASE_URL,
        http_client=httpx.Client(
            headers={"api-key": MIMO_API_KEY},
            timeout=httpx.Timeout(120.0, connect=10.0),
        ),
    )

    # 3a. Quick test to verify Mimo API works
    try:
        test_resp = openai_client.chat.completions.create(
            model=MIMO_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=10,
        )
        test_text = _get_content(test_resp.choices[0].message).strip()
        print(f"  Mimo API test: '{test_text}' (model={MIMO_MODEL}, base={MIMO_BASE_URL})")
        if test_resp.choices[0].message.model_extra:
            print(f"  Extra fields: {test_resp.choices[0].message.model_extra}")
        if hasattr(test_resp, 'model_extra') and test_resp.model_extra:
            print(f"  Response extra: {test_resp.model_extra}")
    except Exception as e:
        print(f"  Mimo API test FAILED: {e}")
        print(f"  Model: {MIMO_MODEL}, Base URL: {MIMO_BASE_URL}")
        return

    # 4. Process with thread pool
    success = 0
    failed = 0
    start_time = time.time()
    new_done_ids = set(done_ids)
    completed_in_session = 0
    total_todo = len(todo)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {}
        for idx, chunk in todo:
            future = pool.submit(generate_title_text, chunk, image_index, openai_client)
            future_map[future] = (idx, chunk)

        for future in as_completed(future_map):
            idx, chunk = future_map[future]
            completed_in_session += 1

            try:
                new_title = future.result()
            except Exception as exc:
                print(f"  [FAIL] {chunk.get('chunk_id', '?')}: {exc}")
                new_title = None

            if new_title:
                old_title = chunk.get("title", "")
                chunk["title"] = new_title
                chunks[idx] = chunk
                success += 1
                if completed_in_session <= 5:
                    cid_short = chunk.get("chunk_id", "?")[-12:]
                    print(f"  [OK] {cid_short}: \"{old_title[:40]}...\" → \"{new_title}\"")
            else:
                failed += 1
                if completed_in_session <= 5:
                    cid_short = chunk.get("chunk_id", "?")[-12:]
                    print(f"  [FAIL] {cid_short}: generate_title_text returned None")

            cid = chunk.get("chunk_id", "")
            if cid:
                new_done_ids.add(cid)

            # Incremental save
            if completed_in_session % SAVE_INTERVAL == 0 or completed_in_session == total_todo:
                save_checkpoint(PROGRESS_PATH, new_done_ids)
                save_chunks(CHUNKS_PATH, chunks)
                elapsed = time.time() - start_time
                rate = completed_in_session / elapsed if elapsed > 0 else 0
                print(f"  --- Progress: {completed_in_session}/{total_todo} | "
                      f"OK={success} FAIL={failed} | {rate:.1f} chunk/s | "
                      f"elapsed={elapsed:.0f}s ---")

    # 5. Final save
    save_checkpoint(PROGRESS_PATH, new_done_ids)
    save_chunks(CHUNKS_PATH, chunks)
    elapsed = time.time() - start_time
    print(f"\nDone! Total: {len(chunks)} | Updated: {success} | Failed: {failed} | "
          f"Time: {elapsed:.0f}s")
    print(f"\nNext step: python scripts/rebuild_index_from_jsonl.py")
    print("Then restart the API service.")


if __name__ == "__main__":
    main()
