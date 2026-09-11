#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 RUN_DIR [caption-perception-policy options...]" >&2
  exit 2
fi

RUN_DIR="$1"
shift

PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="${QWEN_VLLM_WORKSPACE_ROOT:-/workspace/qwen3-vllm}/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" ]]; then
  echo "Could not find Python." >&2
  exit 2
fi

exec "$PY" -m qwen_caption_validate.caption_perception_policy "$RUN_DIR" "$@"
