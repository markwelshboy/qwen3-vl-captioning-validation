#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAZE_L2CS_WORKSPACE_ROOT:-/workspace/gaze-l2cs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"
MODELS="$ROOT/models"
WEIGHTS="$MODELS/L2CSNet_gaze360.pkl"
L2CS_COMMIT="4a0f978d5b4c426a7d37022d8c927d6ea031dcb6"
EXPECTED_SHA256="8a7f3480d868dd48261e1d59f915b0ef0bb33ea12ea00938fb2168f212080665"

mkdir -p "$ROOT" "$MODELS"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install \
  "git+https://github.com/edavalosanaya/L2CS-Net.git@${L2CS_COMMIT}" \
  "huggingface_hub>=0.30"

if [[ ! -f "$WEIGHTS" ]]; then
  WEIGHTS="$WEIGHTS" "$VENV/bin/python" - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil
import os

src = Path(hf_hub_download(repo_id="tianfxc/l2cs", filename="L2CSNet_gaze360.pkl"))
dst = Path(os.environ["WEIGHTS"])
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print(f"Downloaded L2CS Gaze360 weights -> {dst}")
PY
fi

actual_sha256="$(sha256sum "$WEIGHTS" | awk '{print $1}')"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "ERROR: L2CS weight SHA256 mismatch" >&2
  echo "  expected: $EXPECTED_SHA256" >&2
  echo "  actual:   $actual_sha256" >&2
  exit 1
fi

cat <<EOF
L2CS gaze probe environment ready:
  venv:    $VENV
  weights: $WEIGHTS
  sha256:  $actual_sha256
EOF
