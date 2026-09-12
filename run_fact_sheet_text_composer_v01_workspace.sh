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

echo "Phase-5 Python: $PY" >&2
exec "$PY" -m qwen_caption_validate.fact_sheet_text_composer_v01 "$@"
