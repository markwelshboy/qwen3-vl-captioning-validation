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

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "Images directory not found: ${IMAGES_DIR}" >&2
  exit 2
fi
if [[ ! -d "${DWPOSE_DIR}" ]]; then
  echo "DWPose directory not found: ${DWPOSE_DIR}" >&2
  exit 2
fi

SAM3D_OUTPUT="${RUN_DIR}/sam3d-pose-discovery-01"
PROFILE_OUTPUT=""
CENSUS_OUTPUT=""
BBOX_PADDING="0.20"
OVERWRITE=0
SAVE_MESH=0
MAKE_TAR=0

while (($#)); do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --save-mesh)
      SAVE_MESH=1
      shift
      ;;
    --bbox-padding)
      [[ $# -ge 2 ]] || { echo "--bbox-padding requires a value" >&2; exit 2; }
      BBOX_PADDING="$2"
      shift 2
      ;;
    --bbox-padding=*)
      BBOX_PADDING="${1#--bbox-padding=}"
      shift
      ;;
    --sam3d-output)
      [[ $# -ge 2 ]] || { echo "--sam3d-output requires a path" >&2; exit 2; }
      SAM3D_OUTPUT="$2"
      shift 2
      ;;
    --sam3d-output=*)
      SAM3D_OUTPUT="${1#--sam3d-output=}"
      shift
      ;;
    --profile-output)
      [[ $# -ge 2 ]] || { echo "--profile-output requires a path" >&2; exit 2; }
      PROFILE_OUTPUT="$2"
      shift 2
      ;;
    --profile-output=*)
      PROFILE_OUTPUT="${1#--profile-output=}"
      shift
      ;;
    --census-output)
      [[ $# -ge 2 ]] || { echo "--census-output requires a path" >&2; exit 2; }
      CENSUS_OUTPUT="$2"
      shift 2
      ;;
    --census-output=*)
      CENSUS_OUTPUT="${1#--census-output=}"
      shift
      ;;
    --tar)
      MAKE_TAR=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  run_pose_discovery_workspace.sh RUN_DIR [options]

Options:
  --overwrite            Re-run SAM3D even when cached JSON exists.
  --save-mesh            Save OBJ meshes for every image (default: no meshes).
  --bbox-padding N       DWPose bbox padding fraction (default: 0.20).
  --sam3d-output PATH    SAM3D cache directory.
  --profile-output PATH  Relational profile directory.
  --census-output PATH   Census/report directory.
  --tar                  Tar the relational profile directory (includes the default census).
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

SAM3D_OUTPUT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${SAM3D_OUTPUT}")"
if [[ -z "${PROFILE_OUTPUT}" ]]; then
  PROFILE_OUTPUT="${SAM3D_OUTPUT}/relational-pose-profile-v0.3"
fi
PROFILE_OUTPUT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${PROFILE_OUTPUT}")"
if [[ -z "${CENSUS_OUTPUT}" ]]; then
  CENSUS_OUTPUT="${PROFILE_OUTPUT}/pose-library-census"
fi
CENSUS_OUTPUT="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${CENSUS_OUTPUT}")"

echo "Pose discovery"
echo "  run:       ${RUN_DIR}"
echo "  images:    ${IMAGES_DIR}"
echo "  dwpose:    ${DWPOSE_DIR}"
echo "  sam3d:     ${SAM3D_OUTPUT}"
echo "  profiles:  ${PROFILE_OUTPUT}"
echo "  census:    ${CENSUS_OUTPUT}"
echo

SAM_ARGS=(
  "${IMAGES_DIR}"
  --dwpose-dir "${DWPOSE_DIR}"
  --output "${SAM3D_OUTPUT}"
  --bbox-source dwpose
  --bbox-padding "${BBOX_PADDING}"
)
if [[ "${SAVE_MESH}" -eq 1 ]]; then
  SAM_ARGS+=(--save-mesh)
else
  SAM_ARGS+=(--no-save-mesh)
fi
if [[ "${OVERWRITE}" -eq 1 ]]; then
  SAM_ARGS+=(--overwrite)
fi

echo "=== 1/3 SAM3D cache ==="
SAM3D_WORKSPACE_ROOT="${SAM3D_WORKSPACE_ROOT:-/workspace/sam3d-body}" \
bash "${REPO_ROOT}/run_sam3d_probe_workspace.sh" "${SAM_ARGS[@]}"

echo
echo "=== 2/3 Relational pose profile v0.3 ==="
QWEN_WORKSPACE_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}" \
bash "${REPO_ROOT}/run_sam3d_relational_pose_profile_03_workspace.sh" \
  "${SAM3D_OUTPUT}" \
  --dwpose-dir "${DWPOSE_DIR}" \
  --images-dir "${IMAGES_DIR}" \
  --output "${PROFILE_OUTPUT}"

echo
echo "=== 3/3 Pose-library census ==="
QWEN_WORKSPACE_ROOT="${QWEN_WORKSPACE_ROOT:-/workspace/qwen3}" \
bash "${REPO_ROOT}/run_pose_library_census_workspace.sh" \
  "${PROFILE_OUTPUT}" \
  --output "${CENSUS_OUTPUT}"

if [[ "${MAKE_TAR}" -eq 1 ]]; then
  TAR_PATH="${PROFILE_OUTPUT}.tar"
  tar -C "$(dirname "${PROFILE_OUTPUT}")" \
    -cf "${TAR_PATH}" \
    "$(basename "${PROFILE_OUTPUT}")"
  echo
  echo "Tar: ${TAR_PATH}"
fi

echo
echo "Pose discovery complete."
echo "Census: ${CENSUS_OUTPUT}/pose_library_census.md"
echo "Review keys: ${CENSUS_OUTPUT}/review_keys.txt"
