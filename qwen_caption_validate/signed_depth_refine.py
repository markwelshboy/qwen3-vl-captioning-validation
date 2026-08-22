from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

from .laterality_geometry import _load_sam2d, _mirror_sensitive, _read, _sam_vote, _target_points, _write
from .runner import model_slug, resolve_model_id


MIN_SIGNED_DEPTH_DEG = 15.0
MIN_SIGNED_DEPTH_FRACTION = 0.20


def _nearer_side(value: Any) -> str | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction >= MIN_SIGNED_DEPTH_FRACTION:
        return "left"
    if fraction <= -MIN_SIGNED_DEPTH_FRACTION:
        return "right"
    return None


def _component(
    fusion: dict[str, Any],
    dw: dict[str, Any],
    sam: dict[str, Any],
    sam_path: Path,
    *,
    component: str,
) -> dict[str, Any]:
    sam_audit = fusion.get("sam3d_geometry_audit") or {}
    source_name = "shoulder_depth_rotation" if component == "shoulder" else "hip_depth_rotation"
    source = sam_audit.get(source_name) or {}
    provenance = sam_audit.get("target_provenance") or {}
    diagnostics = sam_audit.get("signed_depth_diagnostics") or {}
    diag_name = "shoulder_left_to_right" if component == "shoulder" else "hip_left_to_right"
    left_name = f"left_{component}"
    right_name = f"right_{component}"

    record: dict[str, Any] = {
        "component": component,
        "magnitude_deg": source.get("magnitude_deg"),
        "signed_depth_fraction_left_to_right": diagnostics.get(diag_name),
        "action": "withheld",
    }

    if provenance.get("context_risk") == "requires_review":
        record["reason"] = "target_provenance_requires_review"
        return record
    if source.get("authority") != "qualified_component_geometry":
        record["reason"] = str(source.get("authority") or "component_geometry_not_qualified")
        return record
    try:
        magnitude = float(source.get("magnitude_deg"))
    except (TypeError, ValueError):
        record["reason"] = "magnitude_unavailable"
        return record
    if magnitude < MIN_SIGNED_DEPTH_DEG:
        record["reason"] = "depth_rotation_below_signed_caption_threshold"
        return record

    points = _target_points(dw)
    if left_name not in points or right_name not in points:
        record["reason"] = "bilateral_dwpose_component_not_observed"
        return record

    sam2d = _load_sam2d(sam_path, sam)
    if sam2d is None:
        record["reason"] = "sam3d_2d_projection_unavailable"
        return record

    votes = {
        "left": _sam_vote(left_name, dw, points, sam2d),
        "right": _sam_vote(right_name, dw, points, sam2d),
    }
    record["sam3d_label_votes"] = votes
    if any(vote.get("status") != "agrees" for vote in votes.values()):
        record["reason"] = "sam3d_anatomical_labels_not_correlated_with_observed_dwpose_pair"
        return record

    nearer = _nearer_side(diagnostics.get(diag_name))
    if nearer is None:
        record["reason"] = "signed_depth_fraction_too_small_or_unavailable"
        return record

    record.update(
        action="qualified",
        nearer_anatomical_side=nearer,
        farther_anatomical_side="right" if nearer == "left" else "left",
        authority="dwpose_observed_pair_plus_sam3d_correlated_signed_depth",
        reason=(
            "both anatomical joints are observed by DWPose, SAM3D same-side labels correlate in image space, "
            "and the visibility-qualified SAM3D depth magnitude exceeds the signed-caption threshold"
        ),
    )
    return record


def _signed_shoulder_part(shoulder: dict[str, Any], torso_confirmed: bool) -> dict[str, Any]:
    side = str(shoulder["nearer_anatomical_side"])
    magnitude = float(shoulder.get("magnitude_deg") or 0.0)
    if magnitude >= 50.0:
        depth_phrase = "very strong shoulder depth staggering"
    elif magnitude >= 30.0:
        depth_phrase = "clear shoulder depth staggering"
    else:
        depth_phrase = "moderate shoulder depth staggering"

    geometry = f"closer to camera than the opposite shoulder; {depth_phrase}"
    if torso_confirmed:
        geometry += "; shoulder and hip depth agree that the torso is angled in depth rather than square-on to the camera"

    return {
        "part": f"{side}_shoulder",
        "anatomical_side": side,
        "ownership": "target",
        "visibility": "visible",
        "visible_subparts": ["shoulder"],
        "connectivity_to_target_chain": "connected_visible",
        "geometry": geometry,
        "contact": None,
        "support": None,
        "foreshortening": "depth_staggered",
        "confidence": None,
        "derived_signed_depth": True,
        "fusion_v2": {
            "qualified_ownership": "target",
            "qualified_anatomical_side": side,
            "selection_usable": True,
            "laterality_selection_usable": True,
            "laterality_authority": "dwpose_observed_pair_plus_sam3d_correlated_signed_depth",
            "reasons": [
                "synthetic caption-facing shoulder relation derived only from image-supported deterministic geometry"
            ],
            "laterality_reasons": [
                f"Fusion-v2.3.3 qualifies {side} shoulder as nearer to camera from observed DWPose/SAM3D-correlated signed depth"
            ],
        },
    }


def refine_signed_depth(
    payload: dict[str, Any],
    analysis: dict[str, Any],
    dw: dict[str, Any],
    sam: dict[str, Any],
    sam_path: Path,
) -> dict[str, Any]:
    """Qualify signed camera-relative depth only where observation and anatomical labels agree.

    SAM3D reconstruction never establishes visibility here. DWPose must observe both
    members of the anatomical pair, the existing SAM3D support audit must qualify the
    component geometry, and SAM3D's same-labelled 2-D projections must correlate with
    the observed DWPose joints. Signed direction is withheld for mirrors, provenance
    risk, low-magnitude rotations, or any DWPose/SAM disagreement.
    """
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    if not isinstance(fusion, dict):
        return out

    mirror_sensitive = _mirror_sensitive(analysis)
    audit: dict[str, Any] = {
        "schema_version": "signed-depth-authority-audit-1.0",
        "mirror_sensitive": mirror_sensitive,
        "sign_convention": {
            "diagnostic": "left_to_right_depth_fraction = (right_z - left_z) / pair_length",
            "nearer_mapping": "positive => anatomical left nearer; negative => anatomical right nearer",
            "camera_depth_interpretation": "more-negative SAM3D camera-space z is nearer within the reconstructed body",
        },
        "thresholds": {
            "minimum_depth_rotation_deg": MIN_SIGNED_DEPTH_DEG,
            "minimum_abs_signed_depth_fraction": MIN_SIGNED_DEPTH_FRACTION,
        },
        "method_validation": {
            "status": "empirically_calibrated_on_clear_regression_cases",
            "note": (
                "Direction is promoted only after DWPose/SAM anatomical-label correlation. "
                "The sign mapping was checked against visually clear regression poses; it is not inferred from Analyze laterality."
            ),
        },
        "components": {},
        "synthetic_caption_parts": [],
    }

    if mirror_sensitive:
        audit["components"] = {
            "shoulder": {"component": "shoulder", "action": "withheld", "reason": "mirror_sensitive"},
            "hip": {"component": "hip", "action": "withheld", "reason": "mirror_sensitive"},
        }
    else:
        shoulder = _component(fusion, dw, sam, sam_path, component="shoulder")
        hip = _component(fusion, dw, sam, sam_path, component="hip")
        audit["components"] = {"shoulder": shoulder, "hip": hip}

        shoulder_side = shoulder.get("nearer_anatomical_side") if shoulder.get("action") == "qualified" else None
        hip_side = hip.get("nearer_anatomical_side") if hip.get("action") == "qualified" else None
        torso_confirmed = bool(shoulder_side and hip_side and shoulder_side == hip_side)
        audit["torso_direction"] = {
            "action": "qualified" if torso_confirmed else "withheld",
            "nearer_anatomical_side": shoulder_side if torso_confirmed else None,
            "reason": (
                "visibility-qualified shoulder and hip signed depth independently agree"
                if torso_confirmed
                else "shoulder and hip signed depth are not both qualified with the same nearer side"
            ),
        }

        if shoulder_side:
            synthetic = _signed_shoulder_part(shoulder, torso_confirmed)
            parts = fusion.setdefault("qualified_body_parts", [])
            parts[:] = [
                item for item in parts
                if not (isinstance(item, dict) and item.get("derived_signed_depth"))
            ]
            parts.append(synthetic)
            audit["synthetic_caption_parts"].append(copy.deepcopy(synthetic))

    fusion["schema_version"] = "analysis-fusion-2.3.3"
    fusion["signed_depth_authority_audit"] = audit
    fusion.setdefault("selection_policy", {})["signed_depth_authority"] = (
        "Signed camera-relative shoulder depth is caption-usable only when the pair is DWPose-observed, "
        "visibility-qualified by the existing SAM3D support audit, and SAM3D anatomical labels correlate "
        "with those observed joints. SAM3D reconstruction alone never establishes visibility."
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-signed-depth-refine",
        description="Add visibility-gated signed SAM3D shoulder-depth authority to cached Fusion-v2.3.2.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--sam3d-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.2" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    sam3d_dir = (args.sam3d_dir or (run_dir / "sam3d")).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "fusion-v2.3.3" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion-v2.3.2"), (dwpose_dir, "DWPose"), (sam3d_dir, "SAM3D")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = missing = signed_shoulders = signed_torsos = 0
    records: list[dict[str, Any]] = []

    for fusion_path in sorted(fusion_dir.glob("*.fused_v2_3.json")):
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_path = output_dir / fusion_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        dw_path = dwpose_dir / f"{key}.dwpose.json"
        sam_path = sam3d_dir / f"{key}.sam3d.json"
        payload = _read(fusion_path)
        analysis_path = Path(str(payload.get("analysis_path") or ""))
        if not analysis_path.is_file():
            analysis_path = run_dir / slug / f"{key}.analysis.json"
        if not dw_path.is_file() or not sam_path.is_file() or not analysis_path.is_file():
            missing += 1
            records.append({"image_key": key, "status": "missing_source"})
            continue
        analysis_record = _read(analysis_path)
        analysis = analysis_record.get("analysis")
        if not isinstance(analysis, dict):
            missing += 1
            records.append({"image_key": key, "status": "missing_analysis"})
            continue

        refined = refine_signed_depth(payload, analysis, _read(dw_path), _read(sam_path), sam_path)
        _write(out_path, refined)
        written += 1
        audit = ((refined.get("fusion") or {}).get("signed_depth_authority_audit") or {})
        shoulder = ((audit.get("components") or {}).get("shoulder") or {})
        torso = audit.get("torso_direction") or {}
        shoulder_qualified = shoulder.get("action") == "qualified"
        torso_qualified = torso.get("action") == "qualified"
        signed_shoulders += int(shoulder_qualified)
        signed_torsos += int(torso_qualified)
        records.append({
            "image_key": key,
            "status": "written",
            "shoulder_signed": shoulder_qualified,
            "shoulder_nearer_side": shoulder.get("nearer_anatomical_side"),
            "torso_signed": torso_qualified,
            "torso_nearer_side": torso.get("nearer_anatomical_side"),
            "shoulder_reason": shoulder.get("reason"),
        })

    index = {
        "schema_version": "analysis-fusion-2.3.3-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "source_fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "sam3d_dir": str(sam3d_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_sources": missing,
        "signed_shoulder_records": signed_shoulders,
        "signed_torso_records": signed_torsos,
        "records": records,
    }
    _write(output_dir / "signed_depth_refine.index.json", index)
    print(f"Fusion-v2.3.3 output: {output_dir}")
    print(
        f"Written: {written}; reused: {skipped}; missing: {missing}; "
        f"signed shoulders: {signed_shoulders}; signed torsos: {signed_torsos}"
    )
    return 0 if written or skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
