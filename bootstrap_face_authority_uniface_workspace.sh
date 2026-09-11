#!/usr/bin/env bash
set -euo pipefail

ROOT="${UNIFACE_WORKSPACE_ROOT:-/workspace/uniface-probe}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="$ROOT/.venv"
SRC="$ROOT/uniface"
UNIFACE_REF="${UNIFACE_REF:-d51618dabe336c4a781ca7de4c4a2e9d80deec43}"
UNIFACE_RUNTIME="${UNIFACE_RUNTIME:-cuda}"
# PyPI onnxruntime-gpu 1.27+ switched to CUDA 13.  The captioning pod's
# established stack is CUDA 12.8, so keep UniFace on the CUDA-12 generation.
UNIFACE_ORT_GPU_SPEC="${UNIFACE_ORT_GPU_SPEC:-onnxruntime-gpu[cuda,cudnn]>=1.23,<1.27}"

case "$UNIFACE_RUNTIME" in
  cuda|cpu) ;;
  *)
    echo "ERROR: UNIFACE_RUNTIME must be 'cuda' or 'cpu' (got: $UNIFACE_RUNTIME)" >&2
    exit 2
    ;;
esac

mkdir -p "$ROOT"

if [[ ! -d "$SRC/.git" ]]; then
  git clone https://github.com/yakhyo/uniface.git "$SRC"
fi

git -C "$SRC" fetch origin "$UNIFACE_REF" --depth 1
git -C "$SRC" checkout --detach "$UNIFACE_REF"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

# UniFace deliberately keeps CPU and GPU ONNX Runtime as optional extras.
# Install the base package first, remove either ORT variant left by an older
# workspace, then install exactly one runtime.  Never leave onnxruntime and
# onnxruntime-gpu installed together in the same venv.
"$PY" -m pip install -e "$SRC"
"$PY" -m pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true

if [[ "$UNIFACE_RUNTIME" == "cuda" ]]; then
  echo "Installing UniFace CUDA ONNX Runtime: $UNIFACE_ORT_GPU_SPEC"
  "$PY" -m pip install "$UNIFACE_ORT_GPU_SPEC"
else
  echo "Installing UniFace CPU ONNX Runtime"
  "$PY" -m pip install 'onnxruntime>=1.16'
fi

echo
echo "=== UniFace ONNX Runtime preflight ==="
UNIFACE_RUNTIME="$UNIFACE_RUNTIME" "$PY" - <<'PY'
import os
import onnxruntime as ort

runtime = os.environ["UNIFACE_RUNTIME"]

# Recent GPU ORT wheels can preload CUDA/cuDNN from NVIDIA site-packages.
# This is especially useful in an isolated venv that does not install torch.
if runtime == "cuda" and hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls(directory="")
    except TypeError:
        # Older CUDA-12 ORT versions expose preload_dlls without directory.
        ort.preload_dlls()

providers = ort.get_available_providers()
print("onnxruntime:", ort.__version__)
print("ORT device:", ort.get_device())
print("ORT providers:", providers)

if runtime == "cuda" and "CUDAExecutionProvider" not in providers:
    raise SystemExit(
        "ERROR: CUDAExecutionProvider is unavailable in the UniFace venv; "
        "refusing to accept a GPU build that would silently run on CPU."
    )

if runtime == "cpu" and "CPUExecutionProvider" not in providers:
    raise SystemExit("ERROR: CPUExecutionProvider is unavailable in the UniFace venv.")
PY

cat <<EOF

UniFace authority probe environment ready:
  venv:         $VENV
  upstream ref: $UNIFACE_REF
  runtime:      $UNIFACE_RUNTIME

The first probe run will download the selected UniFace ONNX weights and verify
them using UniFace's model store.

GPU is now the default.  To deliberately build a CPU-only environment:
  UNIFACE_RUNTIME=cpu bash ./bootstrap_face_authority_uniface_workspace.sh
EOF
