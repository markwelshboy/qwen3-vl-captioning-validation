#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 IMAGE_OR_DIR --output-dir DIR [--only KEY ...] [--provider cuda|cpu|auto] [--dwpose-dir DIR]" >&2
  exit 2
fi

ROOT="${UNIFACE_WORKSPACE_ROOT:-/workspace/uniface-probe}"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "UniFace probe environment not found at $ROOT/.venv" >&2
  echo "Run ./bootstrap_face_authority_uniface_workspace.sh first." >&2
  exit 2
fi

# Prefer an explicit CUDA -> CPU fallback provider list so TensorRT does not
# become the implicit first choice merely because ORT registered it.  Callers
# can still request --provider cpu or --provider auto deliberately.
args=("$@")
has_provider=0
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--provider" || "${args[$i]}" == --provider=* ]]; then
    has_provider=1
    break
  fi
done

if (( ! has_provider )); then
  args+=(--provider cuda)
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
print("default UniFace mode: cuda -> [CUDAExecutionProvider, CPUExecutionProvider]")
PY

echo
exec "$PY" -m qwen_caption_validate.face_authority_uniface_v2 "${args[@]}"
