#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGE_OR_DIR --output-dir DIR [--only KEY ...] [--provider cpu|auto] [--dwpose-dir DIR]" >&2
  exit 2
fi

ROOT="${UNIFACE_WORKSPACE_ROOT:-/workspace/uniface-probe}"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "UniFace probe environment not found at $ROOT/.venv" >&2
  echo "Run ./bootstrap_face_authority_uniface_workspace.sh first." >&2
  exit 2
fi

exec "$PY" -m qwen_caption_validate.face_authority_uniface_v2 "$@"
