from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from . import fact_sheet_specialist_normalizer_v02 as base
from .sam3d_caption_orientation_v01 import build_caption_orientation

SCHEMA_VERSION = "caption-fact-sheet-0.2.2"


def _read_optional_dwpose(path_text: Any) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(str(path_text)).expanduser()
    if not path.is_file():
        return None
    try:
        return base.base._read_json(path)
    except Exception:
        return None


def _enrich_torso_orientation(sheet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(sheet)
    warnings: list[str] = []

    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    torso = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    if not torso or not torso.get("available"):
        return out, warnings

    source_text = torso.get("source")
    if not source_text:
        warnings.append("caption_torso_orientation_missing_sam3d_source")
        return out, warnings
    source = Path(str(source_text)).expanduser()
    if not source.is_file():
        warnings.append("caption_torso_orientation_sam3d_source_not_found")
        return out, warnings

    try:
        with np.load(source, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        dwpose = _read_optional_dwpose(torso.get("dwpose_source"))
        orientation = build_caption_orientation(arrays, dwpose)
    except Exception as exc:
        warnings.append(f"caption_torso_orientation_failed:{type(exc).__name__}")
        torso["caption_orientation_error"] = f"{type(exc).__name__}: {exc}"
        return out, warnings

    # Preserve the old root/global optical-axis quantity explicitly before
    # replacing the ambiguous generic fields with the caption-facing upper
    # torso orientation relative to the physical camera center.
    torso["legacy_root_optical_axis"] = {
        "torso_camera_orientation": torso.get("torso_camera_orientation"),
        "torso_yaw_magnitude_deg": torso.get("torso_yaw_magnitude_deg"),
        "reference": "body_root_forward_vs_camera_optical_axis",
    }
    torso["body_root_orientation"] = orientation["body_root_orientation"]
    torso["upper_torso_orientation"] = orientation["upper_torso_orientation"]
    torso["caption_orientation"] = orientation["caption_orientation"]
    torso["orientation_policy"] = orientation["policy"]

    upper = orientation["upper_torso_orientation"]
    if upper.get("orientation_band") is not None:
        torso["torso_camera_orientation"] = upper.get("orientation_band")
    if upper.get("yaw_magnitude_deg") is not None:
        torso["torso_yaw_magnitude_deg"] = upper.get("yaw_magnitude_deg")
    torso["orientation_reference"] = "upper_torso_to_physical_camera_center"
    torso["note_phase4b2"] = (
        "Caption-facing torso orientation uses the reconstructed shoulder/hip plane relative to the physical "
        "camera center. Body/root orientation is retained separately so meaningful torso twist can survive."
    )
    return out, warnings


def _apply_phase4b2(sheet: dict[str, Any]) -> dict[str, Any]:
    out, warnings = _enrich_torso_orientation(sheet)
    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    existing = [str(x) for x in (audit.get("warnings") or []) if x]
    audit["warnings"] = sorted(set(existing + warnings))
    audit["phase"] = "4B.2"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        root_orientation_and_upper_torso_orientation_are_separate=True,
        torso_caption_reference_is_physical_camera_center=True,
        optical_axis_root_yaw_is_diagnostic_not_caption_default=True,
        orientation_bands_are_not_globally_rethresholded=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


_BASE_BUILD_FACT_SHEET = base._build_fact_sheet


def _build_fact_sheet(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _apply_phase4b2(_BASE_BUILD_FACT_SHEET(*args, **kwargs))


def main() -> int:
    # Reuse the stable 4B.1 CLI/discovery/output flow and add only the
    # camera-center / articulated-torso refinement.
    base._build_fact_sheet = _build_fact_sheet
    base.SCHEMA_VERSION = SCHEMA_VERSION
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
