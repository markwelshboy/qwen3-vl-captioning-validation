#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 RUN_DIR [--overwrite] [--save-mesh] [--bbox-padding N] [--sam3d-output PATH] [--profile-output PATH] [--census-output PATH] [--tar]" >&2
  exit 2
fi
shift

RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
IMAGES_DIR="${RUN_DIR}/images"
DWPOSE_DIR="${RUN_DIR}/dwpose"
QWEN_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}"
QWEN_PY="${QWEN_ROOT}/.venv/bin/python"

[[ -d "${IMAGES_DIR}" ]] || { echo "Images directory not found: ${IMAGES_DIR}" >&2; exit 2; }
[[ -d "${DWPOSE_DIR}" ]] || { echo "DWPose directory not found: ${DWPOSE_DIR}" >&2; exit 2; }
[[ -x "${QWEN_PY}" ]] || { echo "Python environment not found: ${QWEN_PY}" >&2; exit 2; }

SAM3D_OUTPUT="${RUN_DIR}/sam3d-pose-discovery-01"
PROFILE_OUTPUT=""
CENSUS_OUTPUT=""
BBOX_PADDING="0.20"
OVERWRITE=0
SAVE_MESH=0
MAKE_TAR=0

while (($#)); do
  case "$1" in
    --overwrite) OVERWRITE=1; shift ;;
    --save-mesh) SAVE_MESH=1; shift ;;
    --bbox-padding) [[ $# -ge 2 ]] || { echo "--bbox-padding requires a value" >&2; exit 2; }; BBOX_PADDING="$2"; shift 2 ;;
    --bbox-padding=*) BBOX_PADDING="${1#--bbox-padding=}"; shift ;;
    --sam3d-output) [[ $# -ge 2 ]] || { echo "--sam3d-output requires a path" >&2; exit 2; }; SAM3D_OUTPUT="$2"; shift 2 ;;
    --sam3d-output=*) SAM3D_OUTPUT="${1#--sam3d-output=}"; shift ;;
    --profile-output) [[ $# -ge 2 ]] || { echo "--profile-output requires a path" >&2; exit 2; }; PROFILE_OUTPUT="$2"; shift 2 ;;
    --profile-output=*) PROFILE_OUTPUT="${1#--profile-output=}"; shift ;;
    --census-output) [[ $# -ge 2 ]] || { echo "--census-output requires a path" >&2; exit 2; }; CENSUS_OUTPUT="$2"; shift 2 ;;
    --census-output=*) CENSUS_OUTPUT="${1#--census-output=}"; shift ;;
    --tar) MAKE_TAR=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage:
  run_pose_discovery_workspace.sh RUN_DIR [options]

Options:
  --overwrite            Re-run SAM3D even when cached JSON exists.
  --save-mesh            Save OBJ meshes for every eligible image (default: no meshes).
  --bbox-padding N       DWPose bbox padding fraction (default: 0.20).
  --sam3d-output PATH    SAM3D cache directory.
  --profile-output PATH  Relational profile directory.
  --census-output PATH   Census/report directory.
  --tar                  Tar the relational profile directory (includes the default census).

Images without a usable DWPose target bbox are skipped from pose discovery and
recorded in pose_discovery_skipped.json. They are not sent through full-image
SAM3D because a crop with no observed body geometry should not gain a
reconstruction-driven pose claim merely to make the batch complete.
EOF
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

SAM3D_OUTPUT="$("${QWEN_PY}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${SAM3D_OUTPUT}")"
if [[ -z "${PROFILE_OUTPUT}" ]]; then PROFILE_OUTPUT="${SAM3D_OUTPUT}/relational-pose-profile-v0.6"; fi
PROFILE_OUTPUT="$("${QWEN_PY}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${PROFILE_OUTPUT}")"
if [[ -z "${CENSUS_OUTPUT}" ]]; then CENSUS_OUTPUT="${PROFILE_OUTPUT}/pose-library-census"; fi
CENSUS_OUTPUT="$("${QWEN_PY}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${CENSUS_OUTPUT}")"

mkdir -p "${SAM3D_OUTPUT}"
ELIGIBLE_FILE="${SAM3D_OUTPUT}/pose_discovery_eligible.txt"
SKIPPED_FILE="${SAM3D_OUTPUT}/pose_discovery_skipped.json"

"${QWEN_PY}" - "${IMAGES_DIR}" "${DWPOSE_DIR}" "${ELIGIBLE_FILE}" "${SKIPPED_FILE}" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
images_dir, dwpose_dir, eligible_path, skipped_path = map(lambda x: Path(x).resolve(), sys.argv[1:5])
eligible = []
skipped = []
for image in sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
    rel = image.relative_to(images_dir)
    key = str(rel.with_suffix("")).replace("/", "__").replace("\\", "__")
    record_path = dwpose_dir / f"{key}.dwpose.json"
    if not record_path.is_file():
        skipped.append({"image": rel.as_posix(), "image_key": key, "reason": "missing_dwpose_record"})
        continue
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as exc:
        skipped.append({"image": rel.as_posix(), "image_key": key, "reason": f"invalid_dwpose_json:{type(exc).__name__}"})
        continue
    bbox = (((record.get("derived") or {}).get("target") or {}).get("keypoint_bbox") or {})
    if not all(name in bbox for name in ("x0", "y0", "x1", "y1")):
        skipped.append({"image": rel.as_posix(), "image_key": key, "reason": "missing_dwpose_target_bbox"})
        continue
    eligible.append(rel.as_posix())
eligible_path.write_text("".join(f"{item}\n" for item in eligible), encoding="utf-8")
skipped_path.write_text(json.dumps({"schema_version": "pose-discovery-preflight-0.1", "eligible_count": len(eligible), "skipped_count": len(skipped), "skipped": skipped}, indent=2) + "\n", encoding="utf-8")
print(f"Pose-discovery preflight: eligible={len(eligible)} skipped={len(skipped)}")
if skipped: print(f"Skipped detail: {skipped_path}")
PY

mapfile -t ELIGIBLE_IMAGES < "${ELIGIBLE_FILE}"
[[ "${#ELIGIBLE_IMAGES[@]}" -gt 0 ]] || { echo "No images have a usable DWPose target bbox; nothing to run." >&2; exit 2; }

echo "Pose discovery"
echo "  run:       ${RUN_DIR}"
echo "  images:    ${IMAGES_DIR}"
echo "  dwpose:    ${DWPOSE_DIR}"
echo "  sam3d:     ${SAM3D_OUTPUT}"
echo "  profiles:  ${PROFILE_OUTPUT}"
echo "  census:    ${CENSUS_OUTPUT}"
echo "  eligible:  ${#ELIGIBLE_IMAGES[@]}"
echo

SAM_ARGS=("${IMAGES_DIR}" --dwpose-dir "${DWPOSE_DIR}" --output "${SAM3D_OUTPUT}" --bbox-source dwpose --bbox-padding "${BBOX_PADDING}")
for image in "${ELIGIBLE_IMAGES[@]}"; do SAM_ARGS+=(--include "${image}"); done
if [[ "${SAVE_MESH}" -eq 1 ]]; then SAM_ARGS+=(--save-mesh); else SAM_ARGS+=(--no-save-mesh); fi
if [[ "${OVERWRITE}" -eq 1 ]]; then SAM_ARGS+=(--overwrite); fi

echo "=== 1/3 SAM3D cache ==="
SAM3D_WORKSPACE_ROOT="${SAM3D_WORKSPACE_ROOT:-/workspace/sam3d-body}" bash "${REPO_ROOT}/run_sam3d_probe_workspace.sh" "${SAM_ARGS[@]}"

echo
echo "=== 2/3 Relational pose profile v0.6 ==="
QWEN_WORKSPACE_ROOT="${QWEN_ROOT}" bash "${REPO_ROOT}/run_sam3d_relational_pose_profile_06_workspace.sh" \
  "${SAM3D_OUTPUT}" --dwpose-dir "${DWPOSE_DIR}" --images-dir "${IMAGES_DIR}" --output "${PROFILE_OUTPUT}"

echo
echo "=== 3/3 Pose-library census v0.2 ==="
QWEN_WORKSPACE_ROOT="${QWEN_ROOT}" bash "${REPO_ROOT}/run_pose_library_census_02_workspace.sh" \
  "${PROFILE_OUTPUT}" --output "${CENSUS_OUTPUT}"

if [[ "${MAKE_TAR}" -eq 1 ]]; then
  TAR_PATH="${PROFILE_OUTPUT}.tar"
  tar -C "$(dirname "${PROFILE_OUTPUT}")" -cf "${TAR_PATH}" "$(basename "${PROFILE_OUTPUT}")"
  echo
  echo "Tar: ${TAR_PATH}"
fi

echo
echo "Pose discovery complete."
echo "Census: ${CENSUS_OUTPUT}/pose_library_census.md"
echo "Review keys: ${CENSUS_OUTPUT}/review_keys.txt"
