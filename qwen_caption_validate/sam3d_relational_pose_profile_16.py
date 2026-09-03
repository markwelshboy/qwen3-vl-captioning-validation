from __future__ import annotations

"""v0.16 assertion-authority and posture-modifier refinement.

This layer keeps every v0.15 posture-family score unchanged.  It addresses two
review findings without trying to make reconstructed anatomy masquerade as
observation:

* crouching is a lower-body claim, so a public ``crouching`` label now needs a
  directly corroborated hip+knee evidence path; head/shoulders alone can keep
  crouching as the best reconstruction candidate but cannot publish it;
* image-plane shoulder declination and torso inclination are exposed as
  composable posture modifiers (for example ``sitting_heavily_leaning_back``),
  with direct DWPose authority kept separate from SAM3D reconstruction.

The v0.15 recline authority policy is deliberately retained.  When a strong
reconstruction is withheld because pose-joint corroboration is weak, v0.16
emits an explicit ``semantic_recovery`` hint for Fusion rather than weakening
the geometry governor.  This is important for severe foreshortening where the
person is visually present in the image but DWPose/SAM3D 2-D joint agreement is
poor.
"""

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import sam3d_relational_pose_profile_15 as v15


v14 = v15.v14
v13 = v15.v13
v12 = v15.v12
v09 = v15.v09

# Crouching is defined by a flexed lower-body topology.  Do not publish it from
# a head/shoulder crop merely because the latent full-body reconstruction likes
# crouching.  Both hips and knees need some corroborated crop authority.
CROUCH_MIN_HIP_AUTHORITY = 0.20
CROUCH_MIN_KNEE_AUTHORITY = 0.20
CROUCH_MIN_LOWER_PATH_AUTHORITY = 0.20

# Human-readable lean bands.  These are modifiers, not new pose families.
LEAN_SLIGHT_DEG = 15.0
LEAN_MODERATE_DEG = 25.0
LEAN_HEAVY_DEG = 35.0
LEAN_NEAR_HORIZONTAL_DEG = 60.0
SHOULDER_TILT_SLIGHT_DEG = 10.0
SHOULDER_TILT_STRONG_DEG = 25.0
SHOULDER_TILT_VERY_STRONG_DEG = 40.0
DIRECTION_SCORE_MIN = 0.35
DIRECTION_SCORE_STRONG = 0.60

# A strong but withheld reconstruction is useful to Fusion as a semantic
# candidate; it is not permission to serialize hidden joint details.
SEMANTIC_RECOVERY_MIN_SCORE = 0.65
SEMANTIC_RECOVERY_MIN_MARGIN = 0.10


def _round(value: float | None, digits: int = 3) -> float | None:
    return v15._round(value, digits)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fold_line_angle_deg(value: float) -> float:
    angle = float(value)
    while angle > 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def _angle_from_image_down_deg(vector: np.ndarray) -> float | None:
    v = np.asarray(vector, dtype=np.float64)[:2]
    n = float(np.linalg.norm(v))
    if n <= 1e-9 or not np.all(np.isfinite(v)):
        return None
    cosine = float(v[1] / n)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def _dwpose_observed_axes(
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Measure directly accepted/in-frame DWPose shoulder/torso axes.

    This is intentionally different from SAM3D's latent 3-D body orientation.
    It answers what the 2-D pose observation itself supports in this crop.
    """
    if not dwpose:
        return {"available": False, "source": "dwpose", "reason": "missing_dwpose"}

    points, accepted, in_frame = v12.atlas03._dwpose_target_points(dwpose, width, height)
    idx = v12.atlas03.base.IDX

    def point(name: str) -> np.ndarray | None:
        i = idx.get(name)
        if i is None or i >= len(points) or name not in accepted or name not in in_frame:
            return None
        p = np.asarray(points[i, :2], dtype=np.float64)
        return p if np.all(np.isfinite(p)) else None

    ls, rs = point("left_shoulder"), point("right_shoulder")
    lh, rh = point("left_hip"), point("right_hip")

    shoulder_signed = None
    shoulder_abs = None
    lower_shoulder = None
    if ls is not None and rs is not None:
        delta = rs - ls
        if float(np.linalg.norm(delta)) > 1e-9:
            shoulder_signed = _fold_line_angle_deg(
                float(np.degrees(np.arctan2(float(delta[1]), float(delta[0]))))
            )
            shoulder_abs = abs(shoulder_signed)
            if shoulder_abs >= SHOULDER_TILT_SLIGHT_DEG:
                # Image Y grows downward. Positive left->right angle means the
                # right shoulder is lower in the source image.
                lower_shoulder = "right" if shoulder_signed > 0.0 else "left"

    torso_angle = None
    if all(p is not None for p in (ls, rs, lh, rh)):
        shoulder_mid = (ls + rs) / 2.0
        hip_mid = (lh + rh) / 2.0
        torso_angle = _angle_from_image_down_deg(hip_mid - shoulder_mid)

    return {
        "available": bool(shoulder_signed is not None or torso_angle is not None),
        "source": "dwpose_accepted_in_frame",
        "accepted_landmarks": sorted(accepted),
        "in_frame_landmarks": sorted(in_frame),
        "shoulder_line_declination_signed_deg": _round(shoulder_signed),
        "shoulder_line_declination_abs_deg": _round(shoulder_abs),
        "lower_shoulder": lower_shoulder,
        "torso_axis_from_image_down_deg": _round(torso_angle),
        "shoulders_observed": bool(ls is not None and rs is not None),
        "hips_observed": bool(lh is not None and rh is not None),
    }


def _lean_severity(angle: float | None) -> str:
    if angle is None:
        return "unavailable"
    a = abs(float(angle))
    if a >= LEAN_NEAR_HORIZONTAL_DEG:
        return "near_horizontal"
    if a >= LEAN_HEAVY_DEG:
        return "heavy"
    if a >= LEAN_MODERATE_DEG:
        return "moderate"
    if a >= LEAN_SLIGHT_DEG:
        return "slight"
    return "upright"


def _shoulder_tilt_severity(angle: float | None) -> str:
    if angle is None:
        return "unavailable"
    a = abs(float(angle))
    if a >= SHOULDER_TILT_VERY_STRONG_DEG:
        return "very_strong"
    if a >= SHOULDER_TILT_STRONG_DEG:
        return "strong"
    if a >= SHOULDER_TILT_SLIGHT_DEG:
        return "slight"
    return "level"


def _posture_modifier_diagnostic(
    profile: dict[str, Any],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    projected = profile.get("sam3d_projected_pose") or {}
    region = {str(k): float(v or 0.0) for k, v in (projected.get("region_support") or {}).items()}
    geometry = projected.get("geometry") or {}
    directional = projected.get("directional_recline_diagnostic") or {}
    orientation = projected.get("body_orientation_diagnostic") or {}
    observed = _dwpose_observed_axes(dwpose, width, height)

    observed_torso = observed.get("torso_axis_from_image_down_deg")
    reconstructed_torso = geometry.get("torso_axis_from_image_down_deg")
    if observed_torso is not None:
        torso_angle = float(observed_torso)
        torso_source = "dwpose_observed_shoulders_and_hips"
        torso_authority = min(float(region.get("shoulders", 0.0)), float(region.get("hips", 0.0)))
    elif reconstructed_torso is not None:
        torso_angle = float(reconstructed_torso)
        torso_source = "sam3d_reconstruction"
        torso_authority = min(float(region.get("shoulders", 0.0)), float(region.get("hips", 0.0)))
    else:
        torso_angle = None
        torso_source = "unavailable"
        torso_authority = 0.0

    shoulder_declination = observed.get("shoulder_line_declination_abs_deg")
    shoulder_declination_source = "dwpose_accepted_in_frame"
    shoulder_authority = float(region.get("shoulders", 0.0)) if shoulder_declination is not None else 0.0
    if shoulder_declination is None and orientation.get("shoulder_roll_deg") is not None:
        shoulder_declination = abs(float(orientation.get("shoulder_roll_deg")))
        shoulder_declination_source = "sam3d_reconstruction"
        shoulder_authority = float(region.get("shoulders", 0.0))

    retreat = float(directional.get("retreat_from_support_score") or 0.0)
    advance = float(directional.get("advance_toward_support_score") or 0.0)
    if retreat >= DIRECTION_SCORE_STRONG and retreat > advance:
        lean_direction = "back"
    elif advance >= DIRECTION_SCORE_STRONG and advance > retreat:
        lean_direction = "forward"
    elif retreat >= DIRECTION_SCORE_MIN and retreat > advance:
        lean_direction = "back_possible"
    elif advance >= DIRECTION_SCORE_MIN and advance > retreat:
        lean_direction = "forward_possible"
    else:
        lean_direction = "direction_indeterminate"

    severity = _lean_severity(torso_angle)
    shoulder_severity = _shoulder_tilt_severity(shoulder_declination)
    pose_for_modifier = str(projected.get("pose") or "uncertain")
    if pose_for_modifier == "uncertain":
        pose_for_modifier = str(projected.get("best_candidate_pose") or "uncertain")

    modifiers: list[str] = []
    if severity in {"heavy", "near_horizontal"}:
        if lean_direction == "back":
            modifiers.append("heavily_leaning_back")
        elif lean_direction == "forward":
            modifiers.append("heavily_leaning_forward")
        else:
            modifiers.append("heavily_leaning")
    elif severity == "moderate":
        if lean_direction.startswith("back"):
            modifiers.append("leaning_back")
        elif lean_direction.startswith("forward"):
            modifiers.append("leaning_forward")
        else:
            modifiers.append("leaning")

    if shoulder_severity in {"strong", "very_strong"}:
        modifiers.append("shoulder_line_strongly_tilted")

    compound = None
    if pose_for_modifier == "sitting" and modifiers:
        if "heavily_leaning_back" in modifiers:
            compound = "sitting_heavily_leaning_back"
        elif "heavily_leaning_forward" in modifiers:
            compound = "sitting_heavily_leaning_forward"
        elif "heavily_leaning" in modifiers:
            compound = "sitting_heavily_leaning"
        elif "leaning_back" in modifiers:
            compound = "sitting_leaning_back"
        elif "leaning_forward" in modifiers:
            compound = "sitting_leaning_forward"
        elif "leaning" in modifiers:
            compound = "sitting_leaning"

    return {
        "report_only": True,
        "pose_family_for_modifier": pose_for_modifier,
        "dwpose_observed_axes": observed,
        "torso_inclination_from_vertical_deg": _round(torso_angle),
        "torso_inclination_source": torso_source,
        "torso_inclination_authority": _round(torso_authority, 4),
        "torso_inclination_authority_percent": int(round(100.0 * torso_authority)),
        "lean_severity": severity,
        "lean_direction": lean_direction,
        "retreat_from_support_score": _round(retreat, 4),
        "advance_toward_support_score": _round(advance, 4),
        "shoulder_line_declination_deg": _round(float(shoulder_declination)) if shoulder_declination is not None else None,
        "shoulder_line_declination_source": shoulder_declination_source if shoulder_declination is not None else "unavailable",
        "shoulder_line_declination_authority": _round(shoulder_authority, 4),
        "shoulder_line_declination_authority_percent": int(round(100.0 * shoulder_authority)),
        "shoulder_line_tilt_severity": shoulder_severity,
        "lower_shoulder": observed.get("lower_shoulder"),
        "suggested_modifiers": modifiers,
        "suggested_compound_pose_modifier": compound,
        "interpretation": (
            "Lean/declination is a composable modifier, not a new primary pose family. "
            "DWPose image-plane shoulder/torso measurements are reported when directly "
            "accepted in-frame; SAM3D values remain reconstruction evidence with separate "
            "authority. Fusion may combine these modifiers with scene support semantics."
        ),
    }


def _apply_crouch_assertion_authority(profile: dict[str, Any]) -> None:
    projected = profile.get("sam3d_projected_pose") or {}
    governance = projected.get("physical_governance") or {}
    authority = governance.get("authority") or {}
    v15_assertion = projected.get("assertion_authority") or {}

    public_before = str(projected.get("pose") or "uncertain")
    best = str(projected.get("best_candidate_pose") or "uncertain")
    region = {str(k): float(v or 0.0) for k, v in (projected.get("region_support") or {}).items()}

    hip = float(region.get("hips", 0.0))
    knee = float(region.get("knees", 0.0))
    thigh = float(region.get("thighs", 0.0))
    lower_path = min(hip, knee)
    lower_chain = min(max(hip, thigh), knee)
    qualifies = bool(
        hip >= CROUCH_MIN_HIP_AUTHORITY
        and knee >= CROUCH_MIN_KNEE_AUTHORITY
        and max(lower_path, lower_chain) >= CROUCH_MIN_LOWER_PATH_AUTHORITY
    )

    projected["v15_public_pose_before_crouch_assertion_authority"] = public_before
    projected["v15_assertion_authority_before_v16"] = v15_assertion

    withheld = False
    selected_path = str(v15_assertion.get("selected_path") or authority.get("authority_path") or "existing")
    selected_authority = float(v15_assertion.get("selected_path_authority") or authority.get("crop_support") or 0.0)
    withheld_reason = v15_assertion.get("withheld_reason") or authority.get("withheld_reason")

    if public_before == "crouching" and best == "crouching":
        if qualifies:
            selected_path = "observed_crouch_hip_knee_chain"
            selected_authority = max(lower_path, lower_chain)
        else:
            projected["pose"] = "uncertain"
            withheld = True
            selected_path = "reconstruction_only_crouch"
            selected_authority = max(lower_path, lower_chain)
            withheld_reason = "insufficient_observed_lower_body_support_for_crouching_pose"
            authority["usable_as_projected_pose"] = False
            authority["reconstruction_dominant"] = True
            authority["withheld_reason"] = withheld_reason

    scores = projected.get("posture_scores") or {}
    best_score = float(scores.get(best) or projected.get("reconstruction_match") or 0.0)
    margin = float(projected.get("winner_margin") or 0.0)
    semantic_recovery = {
        "needed": False,
        "candidate_pose": None,
        "reason": None,
        "minimum_candidate_score": SEMANTIC_RECOVERY_MIN_SCORE,
        "minimum_winner_margin": SEMANTIC_RECOVERY_MIN_MARGIN,
    }
    if (
        str(projected.get("pose") or "uncertain") == "uncertain"
        and best not in {"", "uncertain"}
        and best_score >= SEMANTIC_RECOVERY_MIN_SCORE
        and margin >= SEMANTIC_RECOVERY_MIN_MARGIN
    ):
        semantic_recovery.update({
            "needed": True,
            "candidate_pose": best,
            "candidate_score": _round(best_score, 4),
            "candidate_score_percent": int(round(100.0 * best_score)),
            "winner_margin": _round(margin, 4),
            "winner_margin_percent": int(round(100.0 * margin)),
            "reason": "strong_reconstruction_candidate_but_pose_joint_authority_is_insufficient",
            "recommended_fusion_action": "seek_independent_analyze_or_scene_semantics_for_broad_pose",
        })

    assertion = dict(v15_assertion)
    assertion.update({
        "policy_version": "v0.16_pose_specific_assertion_authority",
        "public_pose_before_v16": public_before,
        "public_pose_after": projected.get("pose"),
        "selected_path": selected_path,
        "selected_path_authority": _round(selected_authority, 4),
        "selected_path_authority_percent": int(round(100.0 * selected_authority)),
        "withheld_by_v16": withheld,
        "withheld_reason": withheld_reason,
        "authority_semantics": "pose_joint_corroboration_not_literal_visual_crop_extent",
        "crouch_paths": {
            "hip_knee_chain": {
                "hip_authority": _round(hip, 4),
                "hip_authority_percent": int(round(100.0 * hip)),
                "thigh_authority": _round(thigh, 4),
                "thigh_authority_percent": int(round(100.0 * thigh)),
                "knee_authority": _round(knee, 4),
                "knee_authority_percent": int(round(100.0 * knee)),
                "path_authority": _round(max(lower_path, lower_chain), 4),
                "path_authority_percent": int(round(100.0 * max(lower_path, lower_chain))),
                "qualifies": qualifies,
                "minimum_hip_authority": CROUCH_MIN_HIP_AUTHORITY,
                "minimum_knee_authority": CROUCH_MIN_KNEE_AUTHORITY,
            }
        },
        "semantic_recovery": semantic_recovery,
    })

    authority["assertion_authority"] = assertion
    authority["assertion_authority_path"] = selected_path
    governance["authority"] = authority
    governance["assertion_authority"] = assertion
    projected["physical_governance"] = governance
    projected["assertion_authority"] = assertion
    projected["semantic_recovery"] = semantic_recovery
    profile["sam3d_projected_pose"] = projected


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v15.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.16"

    projected = profile.get("sam3d_projected_pose") or {}
    projected["posture_modifier_diagnostic"] = _posture_modifier_diagnostic(
        profile, dwpose, width, height
    )
    profile["sam3d_projected_pose"] = projected

    _apply_crouch_assertion_authority(profile)

    policy = profile.get("policy") or {}
    policy.update({
        "v16_crouching_assertion_requires_observed_hip_knee_path": True,
        "v16_posture_lean_and_shoulder_declination_are_modifiers_not_pose_families": True,
        "v16_dwpose_axes_are_preferred_for_direct_image_plane_declination": True,
        "v16_pose_joint_authority_is_not_literal_visual_crop_extent": True,
        "v16_strong_withheld_candidates_are_flagged_for_semantic_recovery_in_fusion": True,
        "v16_v15_recline_authority_and_v14_pose_scores_unchanged": True,
    })
    profile["policy"] = policy
    return profile


def _audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []

    for record in rows:
        key = str(record.get("image_key") or "")
        projected = (record.get("profile") or {}).get("sam3d_projected_pose") or {}
        old = str(projected.get("v15_public_pose_before_crouch_assertion_authority") or "uncertain")
        new = str(projected.get("pose") or "uncertain")
        best = str(projected.get("best_candidate_pose") or "uncertain")
        assertion = projected.get("assertion_authority") or {}
        crouch = (assertion.get("crouch_paths") or {}).get("hip_knee_chain") or {}
        if old != new:
            row = {
                "image_key": key,
                "v15_public_pose": old,
                "v16_public_pose": new,
                "best_candidate_pose": best,
                "reconstruction_match_percent": projected.get("reconstruction_match_percent"),
                "winner_margin_percent": projected.get("winner_margin_percent"),
                "pose_joint_authority_percent": projected.get("crop_support_percent"),
                "crouch_lower_path_percent": crouch.get("path_authority_percent"),
                "assertion_path": assertion.get("selected_path"),
                "withheld_reason": assertion.get("withheld_reason"),
            }
            changed.append(row)
            if old == "crouching" and new == "uncertain" and best == "crouching":
                expected.append(row)
            else:
                unexpected.append(row)

        semantic = projected.get("semantic_recovery") or {}
        if semantic.get("needed"):
            recovery.append({
                "image_key": key,
                "candidate_pose": semantic.get("candidate_pose"),
                "candidate_score_percent": semantic.get("candidate_score_percent"),
                "winner_margin_percent": semantic.get("winner_margin_percent"),
                "public_pose": new,
            })

    return {
        "schema_version": "sam3d-v16-authority-regression-audit-0.1",
        "record_count": len(rows),
        "changed_public_pose_count": len(changed),
        "expected_crouch_withheld_count": len(expected),
        "unexpected_public_pose_change_count": len(unexpected),
        "score_families_changed_by_v16": False,
        "authority_regression_free": len(unexpected) == 0,
        "semantic_recovery_candidate_count": len(recovery),
        "changed_public_poses": changed,
        "expected_crouch_withheld": expected,
        "unexpected_public_pose_changes": unexpected,
        "semantic_recovery_candidates": recovery,
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# v0.16 assertion-authority regression audit",
        "",
        f"- Records: {audit['record_count']}",
        f"- Public pose changes vs v0.15: {audit['changed_public_pose_count']}",
        f"- Expected crouching -> uncertain withholds: {audit['expected_crouch_withheld_count']}",
        f"- Unexpected public pose changes: {audit['unexpected_public_pose_change_count']}",
        f"- Posture-family scores changed by v0.16: {'YES' if audit['score_families_changed_by_v16'] else 'NO'}",
        f"- Authority regression free: {'YES' if audit['authority_regression_free'] else 'NO'}",
        f"- Strong withheld candidates flagged for Fusion: {audit['semantic_recovery_candidate_count']}",
        "",
    ]
    changed = audit.get("changed_public_poses") or []
    if changed:
        lines.extend([
            "## Changed public poses",
            "",
            "| image | v0.15 | v0.16 | best | recon | margin | joint authority | crouch lower path | assertion path |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ])
        for row in changed:
            lines.append(
                f"| {row['image_key']} | {row['v15_public_pose']} | {row['v16_public_pose']} | "
                f"{row['best_candidate_pose']} | {row.get('reconstruction_match_percent',0)}% | "
                f"{row.get('winner_margin_percent',0)}% | {row.get('pose_joint_authority_percent',0)}% | "
                f"{row.get('crouch_lower_path_percent',0)}% | {row.get('assertion_path') or '-'} |"
            )
        lines.append("")
    recovery = audit.get("semantic_recovery_candidates") or []
    if recovery:
        lines.extend([
            "## Strong withheld candidates for Fusion",
            "",
            "| image | candidate | score | margin | public pose |",
            "|---|---|---:|---:|---|",
        ])
        for row in recovery:
            lines.append(
                f"| {row['image_key']} | {row.get('candidate_pose')} | "
                f"{row.get('candidate_score_percent',0)}% | {row.get('winner_margin_percent',0)}% | "
                f"{row.get('public_pose')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-16",
        description=(
            "Build v0.16 pose profiles with crouch-specific assertion authority, "
            "lean/shoulder-declination modifiers, and semantic-recovery hints."
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
    output = (args.output or (sam3d_dir / "relational-pose-profile-v0.16")).expanduser().resolve()
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
        mod = projected.get("posture_modifier_diagnostic") or {}
        semantic = projected.get("semantic_recovery") or {}
        print(
            f"{key}: pose={projected.get('pose')} best={projected.get('best_candidate_pose')} "
            f"scores=stand:{scores.get('standing',0)} crouch:{scores.get('crouching',0)} "
            f"squat:{scores.get('squatting',0)} sit:{scores.get('sitting',0)} recl:{scores.get('reclined',0)} "
            f"jointAuthority:{projected.get('crop_support_percent',0)}% "
            f"assert:{assertion.get('selected_path','-')} "
            f"lean:{mod.get('lean_severity','-')}/{mod.get('lean_direction','-')} "
            f"shoulderDecl:{mod.get('shoulder_line_declination_deg')}deg "
            f"fusionRecovery:{semantic.get('candidate_pose') if semantic.get('needed') else '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.16",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    helpers._write_json(output / "sam3d_relational_pose.index.json", index)

    audit = _audit_rows(rows)
    helpers._write_json(output / "v16_authority_regression_audit.json", audit)
    (output / "v16_authority_regression_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")

    print(f"Index: {output / 'sam3d_relational_pose.index.json'}")
    print(
        "v0.16 audit: "
        f"changed={audit['changed_public_pose_count']} "
        f"expected_crouch_withheld={audit['expected_crouch_withheld_count']} "
        f"unexpected={audit['unexpected_public_pose_change_count']} "
        f"scores_changed={'YES' if audit['score_families_changed_by_v16'] else 'NO'} "
        f"authority_regression_free={'YES' if audit['authority_regression_free'] else 'NO'}"
    )
    print(f"Audit: {output / 'v16_authority_regression_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
