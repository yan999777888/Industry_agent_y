"""Parse and display a saved /chat response."""
import sys, json

with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_resp.json") as f:
    raw = json.load(f)
r = raw.get("data", raw)  # handle nested {"code":0, "data":{...}} format

print("=== ANSWER ===")
ans = r.get("answer", "NO ANSWER")
print(ans[:800])
print(f"\n(总长度: {len(ans)} 字符)")

print("\n=== IMAGES ===")
imgs = r.get("images", [])
print(f"Count: {len(imgs)}")
for img in imgs[:8]:
    print(f"  {img.get('image_id', '?')} - {img.get('file_name', '?')}")

print("\n=== RETRIEVAL DEBUG ===")
dbg = r.get("retrieval_debug", {})
for k, v in dbg.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} items - {v[:3]}")
    else:
        print(f"  {k}: {v}")

print("\n=== SOURCES ===")
for s in r.get("sources", [])[:5]:
    print(f"  {s}")
