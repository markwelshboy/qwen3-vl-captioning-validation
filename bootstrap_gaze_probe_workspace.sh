#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAZE_WORKSPACE_ROOT:-/workspace/gaze-ptgaze}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"

mkdir -p "$ROOT"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install 'ptgaze==0.3.0'

cat <<EOF
Gaze probe environment ready:
  $VENV

If MediaPipe fails to import on Ubuntu/Debian because GLES libraries are missing, run:
  apt-get update && apt-get install -y libgles2
EOF
