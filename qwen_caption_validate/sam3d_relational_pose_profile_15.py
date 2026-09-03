from __future__ import annotations

"""v0.15 assertion-authority and body-orientation refinement.

v0.14 gets the physical family ranking into a useful state, but a highly cropped
reclined reconstruction can still cross the generic 10% crop-support threshold
and become a public pose even when only head/shoulders are actually observed.

This layer deliberately does *not* change posture scores.  It separates:

* best reconstructed pose: which governed SAM3D family fits best;
* assertion authority: whether the visible crop contains enough directly
  supported anatomy to publish that family as an observed pose.

For recline there are three admissible evidence paths:

1. observed whole-body/torso path: shoulder->hip or hip->upper-leg support plus
   a genuine whole-body recline diagnostic;
2. observed upper-body path: strong direction-aware upper-body recline plus
   visible upper-body authority;
3. broad observed path: already substantial pose-weighted crop support.

A reconstructed recline with none of those paths remains the best candidate but
is publicly withheld as ``uncertain`` for Fusion to resolve with scene semantics.
Non-reclined public pose decisions are intentionally left unchanged.

v0.15 also adds report-only camera-relative body-rotation diagnostics.  Shoulder
and hip axes in camera X/Z estimate unsigned yaw from frontal/back-facing (0 deg)
toward profile (90 deg).  This supports modifiers such as ``body turned about
45 degrees`` without changing the primary pose family.  It intentionally does
not infer face direction or front-vs-back orientation.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_14 as v14


v13 = v14.v13
v12 = v13.v12
v09 = v13.v09
MHR = v12.MHR

# Recline assertion paths.  These thresholds govern *publication*, not the
# reconstruction score.  The high-crop fallback preserves already well-observed
# reclines without demanding a particular landmark combination.
RECLINE_WHOLE_MIN_SCORE = 0.45
RECLINE_WHOLE_MIN_PATH_AUTHORITY = 0.20
RECLINE_UPPER_MIN_SCORE = 0.55
RECLINE_UPPER_MIN_RETREAT = 0.35
RECLINE_UPPER_MIN_PATH_AUTHORITY = 0.35
RECLINE_BROAD_MIN_CROP_AUTHORITY = 0.30


def _round(value: float | None, digits: int = 3) -> float | None:
    return v14._round(value, digits)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point3(keypoints: np.ndarray, name: str) -> np.ndarray | None:
    idx = MHR.get(name)
    if idx is None or idx >= len(keypoints):
        return None
    p = np.asarray(keypoints[idx, :3], dtype=np.float64)
    if p.size < 3 or not np.all(np.isfinite(p)):
        return None
    return p


def _axis_yaw_from_frontal_deg(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    x = abs(float(d[0]))
    z = abs(float(d[2]))
    if x <= 1e-9 and z <= 1e-9:
        return None
    # A shoulder/hip axis lying in the image plane (mostly X) is compatible with
    # frontal *or back-facing* presentation.  An axis mostly in depth (Z) is
    # profile.  The axis alone cannot distinguish front from back.
    return float(np.degrees(np.arctan2(z, x)))


def _axis_roll_deg(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if abs(float(d[0])) <= 1e-9 and abs(float(d[1])) <= 1e-9:
        return None
    angle = float(np.degrees(np.arctan2(float(d[1]), float(d[0]))))
    # Axis direction is arbitrary for a line. Fold to [-90, 90].
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def _yaw_label(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value < 15.0:
        return "near_frontal_or_back"
    if value < 32.0:
        return "slight_turn"
    if value < 58.0:
        return "three_quarter"
    if value < 78.0:
        return "near_profile"
    return "profile"


def _body_orientation_diagnostic(profile: dict[str, Any], keypoints: np.ndarray) -> dict[str, Any]:
    ls = _point3(keypoints, "left_shoulder")
    rs = _point3(keypoints, "right_shoulder")
    lh = _point3(keypoints, "left_hip")
    rh = _point3(keypoints, "right_hip")

    shoulder_yaw = _axis_yaw_from_frontal_deg(ls, rs)
    hip_yaw = _axis_yaw_from_frontal_deg(lh, rh)
    shoulder_roll = _axis_roll_deg(ls, rs)
    hip_roll = _axis_roll_deg(lh, rh)

    projected = profile.get("sam3d_projected_pose") or {}
    region = {str(k): float(v or 0.0) for k, v in (projected.get("region_support") or {}).items()}
    shoulder_authority = float(region.get("shoulders", 0.0))
    hip_authority = float(region.get("hips", 0.0))

    weighted: list[tuple[float, float]] = []
    if shoulder_yaw is not None:
        weighted.append((shoulder_yaw, max(0.05, shoulder_authority)))
    if hip_yaw is not None:
        weighted.append((hip_yaw, max(0.05, hip_authority)))
    if weighted:
        total = sum(weight for _, weight in weighted)
        body_yaw = sum(value * weight for value, weight in weighted) / max(total, 1e-9)
    else:
        body_yaw = None

    twist = None
    if shoulder_yaw is not None and hip_yaw is not None:
        twist = abs(float(shoulder_yaw) - float(hip_yaw))

    if shoulder_authority > 0.0 and hip_authority > 0.0:
        whole_authority = min(shoulder_authority, hip_authority)
        authority_path = "shoulders_and_hips"
    elif shoulder_authority > 0.0:
        whole_authority = shoulder_authority
        authority_path = "shoulders_only"
    elif hip_authority > 0.0:
        whole_authority = hip_authority
        authority_path = "hips_only"
    else:
        whole_authority = 0.0
        authority_path = "reconstruction_only"

    modifier = None
    if body_yaw is not None:
        if 35.0 <= body_yaw <= 55.0:
            modifier = "body_turned_about_45_degrees"
        elif body_yaw >= 78.0:
            modifier = "body_near_profile_to_camera"
        elif body_yaw >= 58.0:
            modifier = "body_turned_strongly_from_camera"
        elif body_yaw >= 20.0:
            modifier = "body_turned_slightly_from_camera"

    return {
        "report_only": True,
        "available": body_yaw is not None,
        "camera_relative_only": True,
        "front_vs_back_resolved": False,
        "shoulder_yaw_from_frontal_deg": _round(shoulder_yaw),
        "hip_yaw_from_frontal_deg": _round(hip_yaw),
        "body_yaw_from_frontal_deg": _round(body_yaw),
        "body_yaw_label": _yaw_label(body_yaw),
        "shoulder_roll_deg": _round(shoulder_roll),
        "hip_roll_deg": _round(hip_roll),
        "shoulder_hip_yaw_disagreement_deg": _round(twist),
        "possible_torso_twist": bool(twist is not None and twist >= 20.0),
        "suggested_modifier": modifier,
        "observed_authority": _round(whole_authority, 4),
        "observed_authority_percent": int(round(100.0 * whole_authority)),
        "authority_path": authority_path,
        "shoulder_authority_percent": int(round(100.0 * shoulder_authority)),
        "hip_authority_percent": int(round(100.0 * hip_authority)),
        "interpretation": (
            "Unsigned body yaw is estimated from shoulder/hip depth-vs-image-horizontal axes. "
            "0 degrees means the transverse axis is image-parallel (frontal or back-facing); "
            "90 degrees means profile. This is a pose modifier, not a primary posture family, "
            "and does not determine face direction."
        ),
    }


def _apply_pose_assertion_authority(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    authority = governance.get("authority") or {}

    public_before = str(projected.get("pose") or "uncertain")
    best = str(projected.get("best_candidate_pose") or "uncertain")
    candidate_before_authority = str(governance.get("governed_pose_before_authority") or public_before)
    crop_support = float(authority.get("crop_support") or projected.get("crop_support") or 0.0)
    region = {str(k): float(v or 0.0) for k, v in (projected.get("region_support") or {}).items()}

    projected["v14_public_pose_before_assertion_authority"] = public_before
    projected["v14_best_candidate_before_assertion_authority"] = best
    projected["v14_crop_support_before_assertion_authority"] = _round(crop_support, 4)

    whole = projected.get("whole_body_recline_override") or {}
    directional = projected.get("directional_recline_diagnostic") or {}
    upper = projected.get("upper_body_recline_diagnostic") or {}

    shoulder_hip = min(float(region.get("shoulders", 0.0)), float(region.get("hips", 0.0)))
    hip_leg = min(
        float(region.get("hips", 0.0)),
        max(float(region.get("thighs", 0.0)), float(region.get("knees", 0.0))),
    )
    whole_path_authority = max(shoulder_hip, hip_leg)
    whole_score = float(whole.get("whole_body_recline_score") or 0.0)
    upper_score = float(directional.get("directional_upper_recline_score") or 0.0)
    retreat = float(directional.get("retreat_from_support_score") or 0.0)
    upper_path_authority = float(upper.get("path_authority") or 0.0)

    whole_path_ok = bool(
        whole_score >= RECLINE_WHOLE_MIN_SCORE
        and whole_path_authority >= RECLINE_WHOLE_MIN_PATH_AUTHORITY
    )
    upper_path_ok = bool(
        upper_score >= RECLINE_UPPER_MIN_SCORE
        and retreat >= RECLINE_UPPER_MIN_RETREAT
        and upper_path_authority >= RECLINE_UPPER_MIN_PATH_AUTHORITY
    )
    broad_path_ok = bool(crop_support >= RECLINE_BROAD_MIN_CROP_AUTHORITY)

    selected_path = "not_applicable"
    selected_authority = crop_support
    withheld = False
    withheld_reason = authority.get("withheld_reason")

    # Only tighten an already-published reclined result.  Do not alter non-recline
    # public decisions and do not promote v0.14 uncertain cases.
    if public_before == "reclined" and best == "reclined":
        if whole_path_ok:
            selected_path = "observed_whole_body_recline"
            selected_authority = whole_path_authority
        elif upper_path_ok:
            selected_path = "observed_directional_upper_body_recline"
            selected_authority = upper_path_authority
        elif broad_path_ok:
            selected_path = "broad_pose_crop_support"
            selected_authority = crop_support
        else:
            selected_path = "reconstruction_only_recline"
            selected_authority = max(whole_path_authority, 0.0)
            projected["pose"] = "uncertain"
            withheld = True
            withheld_reason = "insufficient_observed_support_for_reclined_pose"
            authority["usable_as_projected_pose"] = False
            authority["reconstruction_dominant"] = True
            authority["withheld_reason"] = withheld_reason
    elif public_before == "uncertain":
        selected_path = "already_withheld_before_v15"
    else:
        selected_path = str(authority.get("authority_path") or "existing_non_recline_policy")

    assertion = {
        "policy_version": "v0.15_pose_specific_assertion_authority",
        "public_pose_before": public_before,
        "public_pose_after": projected.get("pose"),
        "best_reconstruction_candidate": best,
        "governed_pose_before_authority": candidate_before_authority,
        "crop_support": _round(crop_support, 4),
        "crop_support_percent": int(round(100.0 * crop_support)),
        "selected_path": selected_path,
        "selected_path_authority": _round(selected_authority, 4),
        "selected_path_authority_percent": int(round(100.0 * selected_authority)),
        "withheld_by_v15": withheld,
        "withheld_reason": withheld_reason,
        "recline_paths": {
            "whole_body": {
                "score": _round(whole_score, 4),
                "score_percent": int(round(100.0 * whole_score)),
                "shoulder_hip_authority": _round(shoulder_hip, 4),
                "hip_leg_authority": _round(hip_leg, 4),
                "path_authority": _round(whole_path_authority, 4),
                "path_authority_percent": int(round(100.0 * whole_path_authority)),
                "qualifies": whole_path_ok,
                "minimum_score": RECLINE_WHOLE_MIN_SCORE,
                "minimum_path_authority": RECLINE_WHOLE_MIN_PATH_AUTHORITY,
            },
            "upper_body": {
                "directional_recline_score": _round(upper_score, 4),
                "directional_recline_score_percent": int(round(100.0 * upper_score)),
                "retreat_from_support": _round(retreat, 4),
                "path_authority": _round(upper_path_authority, 4),
                "path_authority_percent": int(round(100.0 * upper_path_authority)),
                "qualifies": upper_path_ok,
                "minimum_score": RECLINE_UPPER_MIN_SCORE,
                "minimum_retreat": RECLINE_UPPER_MIN_RETREAT,
                "minimum_path_authority": RECLINE_UPPER_MIN_PATH_AUTHORITY,
            },
            "broad_crop": {
                "path_authority": _round(crop_support, 4),
                "path_authority_percent": int(round(100.0 * crop_support)),
                "qualifies": broad_path_ok,
                "minimum_path_authority": RECLINE_BROAD_MIN_CROP_AUTHORITY,
            },
        },
        "interpretation": (
            "The governed SAM3D family may remain reclined as the best reconstruction while "
            "the public pose is withheld when the crop does not directly support a valid "
            "recline evidence path. Scene semantics may later restore the claim in Fusion."
        ),
    }

    authority["assertion_authority"] = assertion
    authority["assertion_authority_path"] = selected_path
    governance["authority"] = authority
    governance["assertion_authority"] = assertion
    projected["physical_governance"] = governance
    projected["assertion_authority"] = assertion
    profile["sam3d_projected_pose"] = projected


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v14.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.15"

    keypoints = np.asarray(arrays.get("pred_keypoints_3d", np.empty((0, 3))), dtype=np.float64)
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    projected = profile.get("sam3d_projected_pose") or {}
    projected["body_orientation_diagnostic"] = _body_orientation_diagnostic(profile, keypoints)
    profile["sam3d_projected_pose"] = projected

    _apply_pose_assertion_authority(profile)

    policy = profile.get("policy") or {}
    policy.update({
        "v15_reconstruction_candidate_is_separate_from_public_pose_assertion": True,
        "v15_recline_assertion_requires_pose_specific_observed_path": True,
        "v15_non_recline_public_pose_governance_unchanged": True,
        "v15_body_yaw_is_report_only_camera_relative_modifier": True,
        "v15_body_yaw_does_not_resolve_front_vs_back_or_face_direction": True,
    })
    profile["policy"] = policy
    return profile


def _audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    recline_withheld: list[dict[str, Any]] = []
    v14_public = 0
    v15_public = 0

    for record in rows:
        key = str(record.get("image_key") or "")
        projected = (record.get("profile") or {}).get("sam3d_projected_pose") or {}
        old = str(projected.get("v14_public_pose_before_assertion_authority") or "uncertain")
        new = str(projected.get("pose") or "uncertain")
        best = str(projected.get("best_candidate_pose") or "uncertain")
        assertion = projected.get("assertion_authority") or {}
        if old != "uncertain":
            v14_public += 1
        if new != "uncertain":
            v15_public += 1
        if old != new:
            row = {
                "image_key": key,
                "v14_public_pose": old,
                "v15_public_pose": new,
                "best_candidate_pose": best,
                "reconstruction_match_percent": projected.get("reconstruction_match_percent"),
                "winner_margin_percent": projected.get("winner_margin_percent"),
                "crop_support_percent": projected.get("crop_support_percent"),
                "assertion_path": assertion.get("selected_path"),
                "withheld_reason": assertion.get("withheld_reason"),
            }
            changed.append(row)
            if old == "reclined" and new == "uncertain" and best == "reclined":
                recline_withheld.append(row)
            else:
                unexpected.append(row)

    return {
        "schema_version": "sam3d-v15-authority-regression-audit-0.1",
        "record_count": len(rows),
        "v14_public_pose_count": v14_public,
        "v15_public_pose_count": v15_public,
        "changed_public_pose_count": len(changed),
        "intended_recline_withheld_count": len(recline_withheld),
        "unexpected_public_pose_change_count": len(unexpected),
        "broad_non_recline_regression_free": len(unexpected) == 0,
        "changed_public_poses": changed,
        "intended_recline_withheld": recline_withheld,
        "unexpected_public_pose_changes": unexpected,
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# v0.15 assertion-authority regression audit",
        "",
        f"- Records: {audit['record_count']}",
        f"- v0.14 public poses: {audit['v14_public_pose_count']}",
        f"- v0.15 public poses: {audit['v15_public_pose_count']}",
        f"- Public pose changes: {audit['changed_public_pose_count']}",
        f"- Intended reclined -> uncertain withholds: {audit['intended_recline_withheld_count']}",
        f"- Unexpected changes: {audit['unexpected_public_pose_change_count']}",
        f"- Broad non-recline regression free: {'YES' if audit['broad_non_recline_regression_free'] else 'NO'}",
        "",
    ]
    changed = audit.get("changed_public_poses") or []
    if changed:
        lines.extend([
            "## Changed public poses",
            "",
            "| image | v0.14 | v0.15 | best candidate | reconstruction | margin | crop | assertion path |",
            "|---|---|---|---|---:|---:|---:|---|",
        ])
        for row in changed:
            lines.append(
                f"| {row['image_key']} | {row['v14_public_pose']} | {row['v15_public_pose']} | "
                f"{row['best_candidate_pose']} | {row.get('reconstruction_match_percent',0)}% | "
                f"{row.get('winner_margin_percent',0)}% | {row.get('crop_support_percent',0)}% | "
                f"{row.get('assertion_path') or '-'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-15",
        description=(
            "Build v0.15 governed pose profiles with pose-specific recline assertion authority "
            "and report-only camera-relative body rotation diagnostics."
        ),
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")
    dwpose_dir = args.dwpose_dir.expanduser().resolve() if args.dwpose_dir else sam3d_dir.parent / "dwpose"
    images_dir = args.images_dir.expanduser().resolve() if args.images_dir else sam3d_dir.parent / "images"
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.15")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file())
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [p for p in paths if any(token in p.stem.lower() for token in wanted)]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    helpers = v09.v08.v07.v06.v05.v04.v03
    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        dwpose = helpers._read_json(dwpose_path)
        width = int(dwpose.get("image_width") or 0)
        height = int(dwpose.get("image_height") or 0)
        if width <= 0 or height <= 0:
            image_matches = [p for p in images_dir.rglob(f"{key}.*") if p.is_file()] if images_dir.is_dir() else []
            if not image_matches:
                raise SystemExit(f"Cannot determine image size for {key}")
            with Image.open(image_matches[0]) as im:
                width, height = im.size

        profile = build_profile(arrays, dwpose or None, width, height)
        record = {
            "image_key": key,
            "sam3d_arrays": str(path),
            "dwpose": str(dwpose_path) if dwpose_path.is_file() else None,
            "image_width": width,
            "image_height": height,
            "profile": profile,
        }
        out_path = output / f"{key}.sam3d_relational_pose.json"
        helpers._write_json(out_path, record)
        rows.append(record)

        projected = profile["sam3d_projected_pose"]
        scores = projected.get("posture_score_percent") or {}
        assertion = projected.get("assertion_authority") or {}
        orient = projected.get("body_orientation_diagnostic") or {}
        print(
            f"{key}: pose={projected.get('pose')} best={projected.get('best_candidate_pose')} "
            f"scores=stand:{scores.get('standing',0)} crouch:{scores.get('crouching',0)} "
            f"squat:{scores.get('squatting',0)} sit:{scores.get('sitting',0)} recl:{scores.get('reclined',0)} "
            f"crop:{projected.get('crop_support_percent',0)}% "
            f"assert:{assertion.get('selected_path','-')} "
            f"withheld:{'Y' if assertion.get('withheld_by_v15') else '-'} "
            f"yaw:{orient.get('body_yaw_from_frontal_deg')}deg[{orient.get('body_yaw_label','-')}]"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.15",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    helpers._write_json(output / "sam3d_relational_pose.index.json", index)

    audit = _audit_rows(rows)
    helpers._write_json(output / "v15_authority_regression_audit.json", audit)
    (output / "v15_authority_regression_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")

    print(f"Index: {output / 'sam3d_relational_pose.index.json'}")
    print(
        "v0.15 audit: "
        f"changed={audit['changed_public_pose_count']} "
        f"intended_recline_withheld={audit['intended_recline_withheld_count']} "
        f"unexpected={audit['unexpected_public_pose_change_count']} "
        f"non_recline_regression_free={'YES' if audit['broad_non_recline_regression_free'] else 'NO'}"
    )
    print(f"Audit: {output / 'v15_authority_regression_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
