#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [Phase-4B.1 specialist-normalizer options...]" >&2
  exit 2
fi

PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "Could not find qwen workspace Python." >&2
  exit 2
fi

exec "$PY" -m qwen_caption_validate.fact_sheet_specialist_normalizer_v02 "$@"
