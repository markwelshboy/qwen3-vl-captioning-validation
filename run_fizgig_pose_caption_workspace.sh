#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGES_DIR --trigger TOKEN [fizgig-pose-caption options...]" >&2
  exit 2
fi

IMAGES_DIR="$1"
shift

PY="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "Could not find qwen workspace Python." >&2
  exit 2
fi

# Some pod/runtime environments export HF_HUB_ENABLE_HF_TRANSFER=1 globally,
# while this dedicated vLLM environment intentionally does not install the
# optional hf_transfer package. Hugging Face then fails before model loading.
# Keep fast transfer enabled when the package is actually available; otherwise
# disable only this obsolete optional path and let huggingface_hub use its
# normal downloader/cache.
if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
  if ! "$PY" -c 'import hf_transfer' >/dev/null 2>&1; then
    echo "INFO: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is unavailable; disabling hf_transfer for this run."
    export HF_HUB_ENABLE_HF_TRANSFER=0
  fi
fi

exec "$PY" -m qwen_caption_validate.fizgig_pose_caption "$IMAGES_DIR" "$@"
