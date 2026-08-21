#!/usr/bin/env bash
set -euo pipefail

# Build a separate SAM 3D Body environment under /workspace so the experimental
# 3-D probe cannot destabilize the Qwen/DWPose validation environments.
#
# The first probe deliberately avoids Detectron2/SAM/MoGe: target boxes come
# from our existing DWPose cache and SAM 3D Body uses its default camera model.
# This keeps the experiment focused on whether the recovered 3-D body geometry
# adds useful evidence for torso depth rotation/recline.

WORK_ROOT="${SAM3D_WORKSPACE_ROOT:-/workspace/sam3d-body}"
PYTHON_VERSION="${SAM3D_PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${SAM3D_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
UPSTREAM_REPO="${SAM3D_UPSTREAM_REPO:-https://github.com/facebookresearch/sam-3d-body.git}"
UPSTREAM_REF="${SAM3D_UPSTREAM_REF:-b5c765a0d89d789985e186d396315e7590887b94}"
HF_REPO="${SAM3D_HF_REPO:-facebook/sam-3d-body-dinov3}"
CLEAN=0
DOWNLOAD=0

usage() {
    cat <<EOF
Usage: bash ./build_sam3d_workspace.sh [--clean] [--download]

Options:
  --clean       Recreate the SAM3D venv and upstream checkout.
  --download    Pre-download the full gated SAM 3D Body model after access check.
  -h,--help     Show this help.

Environment overrides:
  SAM3D_WORKSPACE_ROOT   Workspace root (default: /workspace/sam3d-body)
  SAM3D_PYTHON_VERSION   Python version (default: 3.11; matches upstream docs)
  SAM3D_TORCH_INDEX_URL  PyTorch wheel index (default: CUDA 12.8)
  SAM3D_UPSTREAM_REPO    Upstream git URL
  SAM3D_UPSTREAM_REF     Upstream git ref/SHA (default pinned validation commit)
  SAM3D_HF_REPO          Gated checkpoint repo (default: facebook/sam-3d-body-dinov3)

Authentication:
  Existing HF_TOKEN is honored. If no token is available, authenticate locally
  after the venv is built with:

    HF_HOME="$WORK_ROOT/huggingface" "$WORK_ROOT/.venv/bin/hf" auth login

Never paste a Hugging Face token into chat or commit it to this repository.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --download)
            DOWNLOAD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$WORK_ROOT" != /workspace && "$WORK_ROOT" != /workspace/* ]]; then
    echo "ERROR: SAM3D_WORKSPACE_ROOT must be /workspace or beneath it." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$WORK_ROOT/.venv"
SRC_ROOT="$WORK_ROOT/src"
UPSTREAM_DIR="$SRC_ROOT/sam-3d-body"
BIN_DIR="$WORK_ROOT/bin"

if (( CLEAN )); then
    rm -rf "$VENV" "$UPSTREAM_DIR"
fi

mkdir -p \
    "$WORK_ROOT/tmp" \
    "$WORK_ROOT/uv-cache" \
    "$WORK_ROOT/pip-cache" \
    "$WORK_ROOT/huggingface/hub" \
    "$WORK_ROOT/huggingface/xet" \
    "$WORK_ROOT/.cache/nv" \
    "$WORK_ROOT/torch-cache" \
    "$SRC_ROOT" \
    "$BIN_DIR"

export TMPDIR="$WORK_ROOT/tmp"
export UV_CACHE_DIR="$WORK_ROOT/uv-cache"
export PIP_CACHE_DIR="$WORK_ROOT/pip-cache"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export TORCH_HOME="$WORK_ROOT/torch-cache"
export CUDA_CACHE_PATH="$WORK_ROOT/.cache/nv"
export HF_HOME="$WORK_ROOT/huggingface"
export HF_HUB_CACHE="$WORK_ROOT/huggingface/hub"
export HF_XET_CACHE="$WORK_ROOT/huggingface/xet"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset HF_HUB_ENABLE_HF_TRANSFER || true

find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    if [[ -x "$BIN_DIR/uv" ]]; then
        echo "$BIN_DIR/uv"
        return 0
    fi
    return 1
}

if ! UV_BIN="$(find_uv)"; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: uv is not installed and curl is unavailable." >&2
        exit 1
    fi
    echo "Installing uv under $BIN_DIR ..."
    export UV_INSTALL_DIR="$BIN_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="$BIN_DIR/uv"
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required." >&2
    exit 1
fi

echo "=== SAM 3D Body workspace build ==="
echo "Validator repo:    $REPO_ROOT"
echo "Workspace root:    $WORK_ROOT"
echo "Python:            $PYTHON_VERSION"
echo "Torch index:       $TORCH_INDEX_URL"
echo "Upstream repo:     $UPSTREAM_REPO"
echo "Pinned upstream:   $UPSTREAM_REF"
echo "Checkpoint repo:   $HF_REPO"
echo "OpenGL platform:   $PYOPENGL_PLATFORM"
echo

df -h "$WORK_ROOT" / 2>/dev/null || true

# PyOpenGL's EGL backend loads the GLVND dispatcher (libEGL.so.1), not the
# NVIDIA vendor library directly. GPU container runtimes commonly inject
# libEGL_nvidia.so.0 while omitting the generic dispatcher package, which makes
# pyrender fail with "EGL: cannot open shared object file" even though the
# NVIDIA EGL vendor library is visible in ldconfig.
echo
echo "Checking system EGL runtime ..."
if ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; then
    echo "System EGL dispatcher: OK"
else
    echo "libEGL.so.1 is missing. Installing the GLVND EGL dispatcher (libegl1) ..."
    if [[ "$(id -u)" -ne 0 ]]; then
        cat >&2 <<'EOF'
ERROR: libEGL.so.1 is required by PyOpenGL/pyrender and is not installed.
Run as root:
  apt-get update && apt-get install -y --no-install-recommends libegl1
Then rerun this build script.
EOF
        exit 1
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: apt-get is unavailable; install the system package providing libEGL.so.1." >&2
        exit 1
    fi
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libegl1
    ldconfig
    if ! ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; then
        echo "ERROR: libegl1 installed but libEGL.so.1 is still unavailable." >&2
        exit 1
    fi
    echo "System EGL dispatcher: installed"
fi

if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
    echo
echo "Cloning SAM 3D Body ..."
    git clone "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi

if [[ -n "$(git -C "$UPSTREAM_DIR" status --porcelain)" ]]; then
    echo "ERROR: upstream checkout has local modifications: $UPSTREAM_DIR" >&2
    echo "Use --clean or preserve your changes manually before continuing." >&2
    exit 1
fi

echo
echo "Pinning upstream checkout ..."
git -C "$UPSTREAM_DIR" fetch origin "$UPSTREAM_REF" --depth=1 || git -C "$UPSTREAM_DIR" fetch origin
git -C "$UPSTREAM_DIR" checkout --detach "$UPSTREAM_REF"
echo "SAM 3D Body commit: $(git -C "$UPSTREAM_DIR" rev-parse HEAD)"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo
echo "Creating Python $PYTHON_VERSION venv ..."
    "$UV_BIN" venv "$VENV" --python "$PYTHON_VERSION" --seed
else
    echo
echo "Reusing existing venv: $VENV"
fi

PY="$VENV/bin/python"

echo
echo "Installing CUDA PyTorch ..."
"$UV_BIN" pip install \
    --python "$PY" \
    --index-url "$TORCH_INDEX_URL" \
    torch torchvision

# These are the upstream INSTALL.md dependencies needed by the core model and
# its utility modules. Detectron2, SAM3 and MoGe are intentionally omitted from
# this first probe because we supply DWPose target bboxes and use the model's
# default camera. If the 3-D evidence proves useful we can add those components
# in a later matched experiment.
echo
echo "Installing SAM 3D Body core dependencies ..."
"$UV_BIN" pip install --python "$PY" \
    pytorch-lightning \
    pyrender \
    opencv-python \
    yacs \
    scikit-image \
    einops \
    timm \
    dill \
    pandas \
    rich \
    hydra-core \
    hydra-submitit-launcher \
    hydra-colorlog \
    pyrootutils \
    webdataset \
    chump \
    'networkx==3.2.1' \
    roma \
    joblib \
    seaborn \
    wandb \
    appdirs \
    ffmpeg \
    cython \
    jsonlines \
    pytest \
    xtcocotools \
    loguru \
    optree \
    fvcore \
    black \
    pycocotools \
    tensorboard \
    huggingface_hub

export PYTHONPATH="$UPSTREAM_DIR:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo
echo "=== Runtime preflight ==="
"$PY" - <<PY
import os
import sys
import torch
import sam_3d_body

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("sam_3d_body:", getattr(sam_3d_body, "__version__", "unknown"))
print("upstream PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("PYOPENGL_PLATFORM:", os.environ.get("PYOPENGL_PLATFORM"))
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to PyTorch.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
PY

echo
echo "=== Headless EGL renderer preflight ==="
"$PY" - <<'PY'
import ctypes
import os

print("PYOPENGL_PLATFORM:", os.environ.get("PYOPENGL_PLATFORM"))
ctypes.CDLL("libEGL.so.1")
print("libEGL.so.1 load: OK")

import pyrender
from sam_3d_body.visualization.renderer import Renderer

renderer = pyrender.OffscreenRenderer(viewport_width=16, viewport_height=16)
renderer.delete()
print("pyrender EGL context: OK")
print("SAM3D Renderer import: OK")
PY

echo
echo "=== Gated checkpoint access check ==="
set +e
"$PY" - <<PY
from huggingface_hub import hf_hub_download
repo = ${HF_REPO@Q}
path = hf_hub_download(repo_id=repo, filename="model_config.yaml")
print("Access OK:", path)
PY
ACCESS_RC=$?
set -e

if (( ACCESS_RC != 0 )); then
    cat >&2 <<EOF

SAM 3D Body code is installed, but the Hugging Face gated checkpoint access
check failed for:
  $HF_REPO

If access has already been approved, authenticate locally in this workspace:

  HF_HOME="$WORK_ROOT/huggingface" \\
  HF_HUB_CACHE="$WORK_ROOT/huggingface/hub" \\
  "$VENV/bin/hf" auth login

Then rerun this build script. Do not paste your token into chat or the repo.
EOF
    exit "$ACCESS_RC"
fi

if (( DOWNLOAD )); then
    echo
echo "Pre-downloading $HF_REPO ..."
    "$PY" - <<PY
from huggingface_hub import snapshot_download
repo = ${HF_REPO@Q}
path = snapshot_download(repo_id=repo)
print("Downloaded:", path)
PY
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
fi

echo
echo "=== SAM 3D Body build complete ==="
echo "Venv:       $VENV"
echo "Upstream:   $UPSTREAM_DIR"
echo "Runner:     $REPO_ROOT/run_sam3d_probe_workspace.sh"
echo
cat <<EOF
Four-image probe example:
  bash "$REPO_ROOT/run_sam3d_probe_workspace.sh" /data/jQTv \\
    --dwpose-dir "$REPO_ROOT/runs/blind-validation-01/dwpose" \\
    --output "$REPO_ROOT/runs/blind-validation-01-v2-1/sam3d-probe" \\
    --include jQTv_720x1280_00008.png \\
    --include jQTv_720x1280_00015.png \\
    --include jQTv_720x1280_00011.png \\
    --include jQTv_720x1280_00013.png
EOF
