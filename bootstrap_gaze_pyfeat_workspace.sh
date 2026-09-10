#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAZE_PYFEAT_WORKSPACE_ROOT:-/workspace/gaze-pyfeat}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"
PYFEAT_REF="${PYFEAT_REF:-30693abe3d6bf0f94d3d8e745a7eac21d3da98ea}"

mkdir -p "$ROOT"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
# Pin the current py-feat 2.1.3 / Detectorv2 v2.8 source exactly so this
# experiment remains reproducible as upstream main evolves.
"$VENV/bin/python" -m pip install \
  "git+https://github.com/cosanlab/py-feat.git@${PYFEAT_REF}"
# py-feat's image path uses cv2 but upstream does not currently declare
# OpenCV in requirements.txt. Keep it explicit in this isolated probe venv.
"$VENV/bin/python" -m pip install 'opencv-python-headless>=4.8'

cat <<EOF
py-feat gaze/head probe environment ready:
  $VENV
  upstream ref: $PYFEAT_REF

The first Detectorv2 run will download the current v2.8 multitask weights
(face_multitask_v28.safetensors) from py-feat/face_multitask_v2.
EOF
