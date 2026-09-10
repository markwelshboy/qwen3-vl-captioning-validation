#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGE_OR_DIR --output-dir DIR [--only KEY ...] [--device cpu|cuda] [--dwpose-dir DIR]" >&2
  exit 2
fi

ROOT="${GAZE_L2CS_WORKSPACE_ROOT:-/workspace/gaze-l2cs}"
PY="$ROOT/.venv/bin/python"
WEIGHTS="${L2CS_WEIGHTS:-$ROOT/models/L2CSNet_gaze360.pkl}"

if [[ ! -x "$PY" ]]; then
  echo "L2CS gaze environment not found at $ROOT/.venv" >&2
  echo "Run ./bootstrap_gaze_l2cs_workspace.sh first." >&2
  exit 2
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "L2CS weights not found at $WEIGHTS" >&2
  echo "Run ./bootstrap_gaze_l2cs_workspace.sh first." >&2
  exit 2
fi

exec "$PY" -m qwen_caption_validate.gaze_probe_l2cs "$@" --weights "$WEIGHTS"
