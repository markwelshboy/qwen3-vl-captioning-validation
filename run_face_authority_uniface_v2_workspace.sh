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

# The Python probe historically defaulted to CPU.  For the rebuilt captioning
# workstation, prefer ORT's registered provider order (CUDA first when the GPU
# bootstrap has been used) unless the caller explicitly chooses a provider.
args=("$@")
has_provider=0
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--provider" || "${args[$i]}" == --provider=* ]]; then
    has_provider=1
    break
  fi
done

if (( ! has_provider )); then
  args+=(--provider auto)
fi

echo "=== UniFace runtime ==="
"$PY" - <<'PY'
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls(directory="")
    except TypeError:
        ort.preload_dlls()
    except Exception as exc:
        print("ORT preload warning:", exc)

providers = ort.get_available_providers()
print("onnxruntime:", ort.__version__)
print("device:", ort.get_device())
print("registered providers:", providers)
PY

echo
exec "$PY" -m qwen_caption_validate.face_authority_uniface_v2 "${args[@]}"
