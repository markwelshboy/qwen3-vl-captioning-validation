#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR --captions-dir DIR [caption-refiner options...]" >&2
  exit 2
fi

RUN_DIR="$1"
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
# while the dedicated qwen vLLM environment may not include the optional
# hf_transfer package. Hugging Face then fails before model loading. Keep it
# enabled when importable; otherwise fall back to huggingface_hub's normal
# downloader/cache for this invocation only.
if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
  if ! "$PY" -c 'import hf_transfer' >/dev/null 2>&1; then
    echo "INFO: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is unavailable; disabling hf_transfer for this run."
    export HF_HUB_ENABLE_HF_TRANSFER=0
  fi
fi

exec "$PY" -m qwen_caption_validate.caption_refiner_pose_vlm "$RUN_DIR" "$@"
