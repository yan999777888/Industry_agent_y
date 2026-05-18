"""Generate visual descriptions for all manual images via DashScope qwen3-vl-plus.

Loads images.jsonl, reads each PNG from disk, calls the vision API for
a description, stores results incrementally, and writes back images.jsonl.

Usage:
    DASHSCOPE_API_KEY=sk-xxx python scripts/describe_all_images.py
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

try:
    from openai import OpenAI
except ImportError:
    print("openai package required. Run: pip install openai")
    sys.exit(1)

KB_DIR = PROJECT_ROOT / "data" / "processed" / "kb"
IMAGES_PATH = KB_DIR / "images.jsonl"
IMAGE_DIR = PROJECT_ROOT / "Knowledge_base" / "插图"

# DashScope config
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VISION_MODEL = os.getenv("DASHSCOPE_VISION_MODEL", "qwen3-vl-plus")
FALLBACK_VISION_MODEL = os.getenv("FALLBACK_VISION_MODEL", "mimo-v2.5-pro")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "tp-cx5mi0rw3ehfudkdh44xxhpceik5nt1fctxi1phsn8x07jgy")
MAX_WORKERS = int(os.getenv("DESCRIBE_WORKERS", "10"))
MAX_RETRIES = 3

DESCRIBE_SYSTEM_PROMPT = (
    "你是一个产品说明书图片分析专家。准确描述图片内容，禁止编造和猜测。"
)

DESCRIBE_PROMPT = (
    "描述这张产品手册图片的内容，控制在2-3句中文。要求：\n"
    "1. 只描述你确定看到的内容，不知道的绝对不要编造\n"
    "2. 如果不确定产品类型，只说'产品部件/组件'，不要猜测具体类别\n"
    "3. 禁止凭空添加'汽车''车辆'等未明确显示的产品类别\n"
    "4. 描述图中可见的部件形状、操作动作、文字标签、编号或安全符号\n"
    "5. 如果是表格或图示，说明其内容用途\n"
    "6. 图片不清晰时明确说'图片不清晰'"
)


def _image_base64(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("ascii")


def describe_image(client: OpenAI, image_id: str, b64: str) -> str:
    """Call vision API to describe a single image. Falls back to FALLBACK_VISION_MODEL on quota errors."""
    models_to_try = [VISION_MODEL]
    if FALLBACK_VISION_MODEL:
        models_to_try.append(FALLBACK_VISION_MODEL)

    for model in models_to_try:
        for attempt in range(MAX_RETRIES):
            try:
                kwargs = dict(
                    model=model,
                    messages=[
                        {"role": "system", "content": DESCRIBE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": DESCRIBE_PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            ],
                        },
                    ],
                    temperature=0.1,
                    max_completion_tokens=256,
                    top_p=0.9,
                )
                # Switch API endpoint for fallback model if it's mimo
                if model == FALLBACK_VISION_MODEL and FALLBACK_BASE_URL and FALLBACK_API_KEY:
                    fallback_client = OpenAI(api_key=FALLBACK_API_KEY, base_url=FALLBACK_BASE_URL)
                    resp = fallback_client.chat.completions.create(**kwargs)
                else:
                    resp = client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if text.strip():
                    return text.strip()
            except Exception as exc:
                is_quota = "AllocationQuota" in str(exc) or "FreeTierOnly" in str(exc)
                if is_quota and model != models_to_try[-1]:
                    print(f"  [QUOTA] {image_id}: switching model from {model}")
                    break  # try next model
                wait = 2 ** attempt
                print(f"  [RETRY {attempt + 1}/{MAX_RETRIES}] {image_id} ({model}): {exc}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
    return ""


def main():
    if not DASHSCOPE_API_KEY:
        print("ERROR: DASHSCOPE_API_KEY environment variable is required.")
        sys.exit(1)

    # Load image metadata
    images: list[dict] = []
    with open(IMAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            images.append(rec)
    print(f"Loaded {len(images)} images from images.jsonl")

    # Count already-described
    already_done = sum(1 for img in images if img.get("description", "").strip())
    print(f"Already have descriptions: {already_done}")

    # Find images needing description
    todo = []
    for img in images:
        if img.get("description", "").strip():
            continue
        # Resolve path
        file_name = img.get("file_name", "")
        path = img.get("path", "")
        if path and os.path.exists(path):
            todo.append((img, path))
        elif file_name:
            fallback = os.path.join(str(IMAGE_DIR), file_name)
            if os.path.exists(fallback):
                img["path"] = fallback  # fix missing path
                todo.append((img, fallback))
            # Try referenced_by paths
            else:
                for ref in img.get("referenced_by", []):
                    ref_path = os.path.join(str(KB_DIR.parent.parent / "Knowledge_base" / ref), file_name)
                    if os.path.exists(ref_path):
                        img["path"] = ref_path
                        todo.append((img, ref_path))
                        break
                else:
                    # Last resort: search by file name in image_dir
                    for root, _dirs, files in os.walk(str(IMAGE_DIR)):
                        if file_name in files:
                            full = os.path.join(root, file_name)
                            img["path"] = full
                            todo.append((img, full))
                            break

    print(f"Need descriptions: {len(todo)}")
    if not todo:
        print("All images already described!")
        sys.exit(0)

    # Check image dir exists
    if not os.path.isdir(str(IMAGE_DIR)):
        print(f"WARNING: Image directory not found: {IMAGE_DIR}")
        print(f"Will try paths in images.jsonl directly")

    # Initialize OpenAI client
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    # Track stats
    success = 0
    failed = 0
    start_time = time.time()
    done_ids: set[str] = set()

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for img, path in todo:
            try:
                b64 = _image_base64(path)
                future = pool.submit(describe_image, client, img["image_id"], b64)
                futures[future] = img
            except Exception as exc:
                print(f"  [FAIL] {img['image_id']}: read error {exc}")
                failed += 1

        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            img = futures[future]
            completed += 1
            try:
                desc = future.result()
                if desc:
                    img["description"] = desc
                    success += 1
                    done_ids.add(img["image_id"])
                else:
                    img["description"] = ""  # mark as failed
                    failed += 1
            except Exception as exc:
                print(f"  [FAIL] {img['image_id']}: {exc}")
                failed += 1

            # Incremental save every 20 images or at the end
            if completed % 20 == 0 or completed == total:
                _save_images(images, IMAGES_PATH)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed}/{total} | OK={success} FAIL={failed} | "
                      f"{rate:.1f} img/s | elapsed={elapsed:.0f}s")

    # Final save
    _save_images(images, IMAGES_PATH)
    elapsed = time.time() - start_time
    print(f"\nDone! Total: {len(images)} | Described: {success} | Failed: {failed} | Time: {elapsed:.0f}s")

    # Print some samples
    described = [img for img in images if img.get("description", "").strip()]
    print(f"\n--- Sample descriptions ({min(5, len(described))} shown) ---")
    for img in described[:5]:
        print(f"  {img['image_id']}: {img['description'][:100]}...")


def _save_images(images: list[dict], path: Path) -> None:
    """Atomic write back to images.jsonl."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for img in images:
            f.write(json.dumps(img, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(path))


if __name__ == "__main__":
    main()
