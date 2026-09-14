#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [broad-pose retry options...]" >&2
  exit 2
fi

PY="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "Could not find qwen text-only workspace Python." >&2
  exit 2
fi

export QWEN_VLLM_TEXT_ONLY_PROFILE="${QWEN_VLLM_TEXT_ONLY_PROFILE:-1}"

echo "Broad-pose retry Python: $PY" >&2
echo "vLLM text-only profile: $QWEN_VLLM_TEXT_ONLY_PROFILE" >&2
exec "$PY" -m qwen_caption_validate.fact_sheet_text_composer_broad_pose_retry_v01 "$@"
