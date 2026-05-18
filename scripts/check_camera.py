import json
with open("data/processed/kb/images.jsonl") as f:
    records = [json.loads(l) for l in f]
camera = [r for r in records if r["image_id"].startswith("Camera")]
has = sum(1 for r in camera if r.get("chunk_ids"))
print(f"Camera images: {len(camera)}, with chunks: {has}, orphans: {len(camera)-has}")
for r in camera[:5]:
    cids = r.get("chunk_ids")
    ref = r.get("referenced_by")
    print(f"  {r['image_id']} chunks={cids} ref={ref}")
# Check if any non-Camera images have chunk_ids that reference the camera manual
all_with_chunks = [r for r in records if r.get("chunk_ids")]
camera_chunked = [r for r in all_with_chunks if "相机" in str(r.get("referenced_by"))]
print(f"Images with chunks referencing camera manual: {len(camera_chunked)}")
for r in camera_chunked[:5]:
    print(f"  {r['image_id']} ref={r['referenced_by']}")
