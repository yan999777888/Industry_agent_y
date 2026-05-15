"""Test Florence-2 image captioning on 8 sample manual images.

Florence-2 is a lightweight (0.77B) vision model by Microsoft with a dedicated
<DETAILED_CAPTION> mode that produces much richer descriptions than BLIP.
"""
import sys, os, time
from pathlib import Path

SAMPLES = [
    ("Manual40_13", "Manual40_13.jpg", "摩托艇手册"),
    ("Manual04_34", "Manual04_34.jpg", "吹风机手册"),
    ("Manual01_23", "Manual01_23.jpg", "空调手册"),
    ("oven_15", "oven_15.jpg", "烤箱手册"),
    ("Manual17_13", "Manual17_13.jpg", "冰箱手册"),
    ("Manual16_32", "Manual16_32.jpg", "健身追踪器手册"),
    ("Manual16_12", "Manual16_12.jpg", "健身追踪器手册"),
    ("Manual05_14", "Manual05_14.jpg", "蒸汽清洁机手册"),
]

IMAGE_DIR = Path(__file__).resolve().parent.parent / "Knowledge_base" / "插图"


def main():
    # Florence-2 checks flash_attn at import time — force-disable it
    import transformers.utils.import_utils as iu
    iu.is_flash_attn_2_available = lambda: False

    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    model_id = "microsoft/Florence-2-large"
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    for image_id, filename, manual in SAMPLES:
        path = IMAGE_DIR / filename
        if not path.exists():
            print(f"[SKIP] {image_id}: file not found")
            continue

        img = Image.open(path).convert("RGB")
        size_kb = path.stat().st_size / 1024

        t0 = time.perf_counter()

        # <MORE_DETAILED_CAPTION> gives richer output than <DETAILED_CAPTION>
        task = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task, images=img, return_tensors="pt")
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3,
            do_sample=False,
        )
        desc = processor.decode(generated_ids[0], skip_special_tokens=True)

        elapsed = time.perf_counter() - t0

        print(f"--- {image_id} ({manual}, {size_kb:.0f}KB, {elapsed:.1f}s) ---")
        print(f"EN: {desc.strip()}")
        print()


if __name__ == "__main__":
    main()
