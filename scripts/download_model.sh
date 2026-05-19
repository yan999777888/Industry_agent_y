#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${INDUSTRY_AGENT_MODELS_DIR:-$PROJECT_ROOT/models}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_ENDPOINT_DEFAULT="https://hf-mirror.com"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-$HF_ENDPOINT_DEFAULT}"

MODELS=(
  "BAAI/bge-m3"
  "BAAI/bge-reranker-v2-m3"
)

mkdir -p "$MODELS_DIR"

echo "[download_model] project_root=$PROJECT_ROOT"
echo "[download_model] models_dir=$MODELS_DIR"
echo "[download_model] HF_ENDPOINT=$HF_ENDPOINT_VALUE"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[download_model] python not found: $PYTHON_BIN" >&2
  exit 1
fi

export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
export MODELS_DIR

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

models = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
]

models_dir = Path(os.environ["MODELS_DIR"]).expanduser().resolve()
models_dir.mkdir(parents=True, exist_ok=True)

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install it with: pip install huggingface_hub"
    ) from exc

def slugify(model_name: str) -> str:
    return model_name.strip("/").replace("/", "--")

allow_patterns = [
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
    "*.md",
    "*.safetensors",
    "*.bin",
    "*.ot",
    "tokenizer*",
    "sentence_*",
    "special_tokens_map*",
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/*",
    "2_Dense/*",
]

for model_name in models:
    local_dir = models_dir / slugify(model_name)
    print(f"[download_model] downloading {model_name} -> {local_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=allow_patterns,
    )
    print(f"[download_model] finished {model_name}")
PY
