#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGE_OR_DIR --output-dir DIR [--only KEY ...] [--device cpu|cuda] [--dwpose-dir DIR]" >&2
  exit 2
fi

ROOT="${GAZE_WORKSPACE_ROOT:-/workspace/gaze-ptgaze}"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Gaze probe environment not found at $ROOT/.venv" >&2
  echo "Run ./bootstrap_gaze_probe_workspace.sh first." >&2
  exit 2
fi

exec "$PY" -m qwen_caption_validate.gaze_probe_ptgaze_compat "$@"
