#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [Phase-5 text-composer options...]" >&2
  exit 2
fi

# Phase 5 defaults to the vLLM backend / FP8 Qwen checkpoint, so prefer the
# dedicated vLLM workspace.  Fall back to the base qwen3 workspace for
# Transformers-only runs or installations without the vLLM environment.
PY="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "Could not find qwen vLLM or base workspace Python." >&2
  exit 2
fi

if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" == "1" ]]; then
  if ! "$PY" -c 'import hf_transfer' >/dev/null 2>&1; then
    echo "INFO: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is unavailable; disabling hf_transfer for this run." >&2
    export HF_HUB_ENABLE_HF_TRANSFER=0
  fi
fi

# The composer is deliberately text-only.  Tell the shared vLLM loader to
# disable image/video inputs, skip multimodal profiling, and use eager mode so
# this job does not pay the VRAM startup cost of the vision path.
export QWEN_VLLM_TEXT_ONLY_PROFILE="${QWEN_VLLM_TEXT_ONLY_PROFILE:-1}"

echo "Phase-5 Python: $PY" >&2
echo "Phase-5 vLLM text-only profile: $QWEN_VLLM_TEXT_ONLY_PROFILE" >&2
exec "$PY" -m qwen_caption_validate.fact_sheet_text_composer_v01 "$@"
