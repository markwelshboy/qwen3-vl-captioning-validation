#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNIFACE_WORKSPACE_ROOT:-/workspace/uniface-probe}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"
SRC="$ROOT/uniface"
UNIFACE_REF="${UNIFACE_REF:-d51618dabe336c4a781ca7de4c4a2e9d80deec43}"

mkdir -p "$ROOT"

if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/yakhyo/uniface.git "$SRC"
fi

git -C "$SRC" fetch origin "$UNIFACE_REF" --depth 1
git -C "$SRC" checkout --detach "$UNIFACE_REF"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
# CPU ONNX is deliberate for this small diagnostic pass: it avoids coupling
# the experiment to the pod's CUDA/ONNX Runtime combination while the models
# are still being evaluated.
"$VENV/bin/python" -m pip install -e "${SRC}[cpu]"

cat <<EOF
UniFace authority probe environment ready:
  $VENV
  upstream ref: $UNIFACE_REF

The first probe run will download the selected UniFace ONNX weights and verify
them using UniFace's model store.
EOF
