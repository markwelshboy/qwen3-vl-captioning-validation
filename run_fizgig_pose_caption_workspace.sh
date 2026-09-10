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

exec "$PY" -m qwen_caption_validate.fizgig_pose_caption "$IMAGES_DIR" "$@"
