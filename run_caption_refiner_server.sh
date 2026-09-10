#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 BUNDLE_DIR [--host 0.0.0.0] [--port 8765]" >&2
  exit 2
fi

PY="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

exec "$PY" -m qwen_caption_validate.caption_refiner_server "$@"
