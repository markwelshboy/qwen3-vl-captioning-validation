from __future__ import annotations

"""Shadow-only specialist for direct raised-leg / support-leg authority.

This module is intentionally conservative.  It does not mutate a caption fact
sheet.  It inspects the already-produced geometry bundle and reports whether a
unilateral raised-leg relation can be bound to anatomical laterality from
*direct* local evidence rather than by taking the opposite side of a weak foot
contact guess.

The first target is the failure mode exposed by imageblind-01_00064: the image
clearly contains a high raised knee and a single supporting leg, but v0.2.11
correctly withheld the old binding because that binding was chained from weak
ankle-height support inference.  This shadow pass asks a narrower question:
can the raised side and support side be established independently from the two
leg chains themselves?

Publication is deliberately not part of v01.  Population review comes first.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from . import bilateral_knee_flexion_shadow_v01 as knee_shadow


DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.12"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "raised-leg-support-shadow-v0.1"

# The shadow gate favours direct 2-D topology.  SAM3D may corroborate the
# decision, but reconstruction alone is never enough to bind anatomical side.
MIN_DWPOSE_LEG_CHAIN_POINTS = 3
MIN_RAISED_KNEE_HEIGHT_BODY = 0.14
MIN_RAISED_KNEE_HEIGHT_ADVANTAGE_BODY = 0.10
MAX_RAISED_KNEE_ANGLE_2D = 115.0
MIN_SUPPORT_KNEE_ANGLE_2D = 135.0
MIN_SUPPORT_ANKLE_DOWN_ADVANTAGE_BODY = 0.10

# SAM3D is corroboration only.  Keep this lower than the bilateral-knee
# publication threshold because severe foreshortening can reduce region
# authority while still preserving a very strong left/right flexion contrast.
MIN_SAM3D_KNEE_AUTHORITY = 0.30
MIN_SAM3D_FLEXION_ADVANTAGE_DEG = 35.0
MAX_SAM3D_RAISED_KNEE_ANGLE = 105.0
MIN_SAM3D_SUPPORT_KNEE_ANGLE = 135.0

POSE_BEARING_ROUTES = {"pose_allowed", "pose_guided"}


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _analysis_payload(sheet: dict[str, Any]) -> dict[str, Any]:
    """Return the richest nested analysis/provenance dictionary available.

    Fact-sheet versions have accumulated data under slightly different names;
    the shadow pass tolerates those layouts so it can be run over the current
    workspace without a migration.
    """
    provenance = _first_dict(sheet.get("provenance"), sheet.get("evidence"))
    return _first_dict(
        sheet.get("analysis"),
        provenance.get("analysis"),
        sheet.get("source_analysis"),
        provenance,
        sheet,
    )


def _route(sheet: dict[str, Any], analysis: dict[str, Any]) -> str | None:
    for container in (sheet, analysis, _first_dict(sheet.get("perception_policy"))):
        for key in ("route", "perception_route", "mode"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    policy = _first_dict(analysis.get("perception_policy"))
    value = policy.get("route") or policy.get("mode")
    return str(value) if value else None


def _dwpose_joint_source(analysis: dict[str, Any]) -> dict[str, Any]:
    # Prefer a normalized joint dictionary when one is present.
    for key in (
        "dwpose_named_joints",
        "dwpose_joints",
        "pose_joints",
        "dwpose",
    ):
        value = analysis.get(key)
        if isinstance(value, dict):
            return value
    for parent_key in ("geometry", "pose_geometry", "body_geometry"):
        parent = analysis.get(parent_key)
        if isinstance(parent, dict):
            for key in ("dwpose_named_joints", "dwpose_joints", "dwpose"):
                value = parent.get(key)
                if isinstance(value, dict):
                    return value
    return {}


def _joint_xy(joints: dict[str, Any], name: str) -> tuple[float, float] | None:
    value = joints.get(name)
    if value is None:
        return None
    if isinstance(value, dict):
        accepted = value.get("accepted")
        in_frame = value.get("in_frame")
        if accepted is False or in_frame is False:
            return None
        x = _finite(value.get("x"))
        y = _finite(value.get("y"))
        if x is None or y is None:
            point = value.get("point") or value.get("xy") or value.get("coords")
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = _finite(point[0]), _finite(point[1])
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = _finite(value[0]), _finite(value[1])
    else:
        return None
    if x is None or y is None:
        return None
    return x, y


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    nba = math.hypot(*ba)
    nbc = math.hypot(*bc)
    if nba <= 1e-9 or nbc <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (nba * nbc)))
    return math.degrees(math.acos(cosine))


def _leg_measurements(joints: dict[str, Any], side: str) -> dict[str, Any]:
    hip = _joint_xy(joints, f"{side}_hip")
    knee = _joint_xy(joints, f"{side}_knee")
    ankle = _joint_xy(joints, f"{side}_ankle")
    observed = sum(point is not None for point in (hip, knee, ankle))
    angle = _angle(hip, knee, ankle) if all(point is not None for point in (hip, knee, ankle)) else None
    return {
        "side": side,
        "hip": hip,
        "knee": knee,
        "ankle": ankle,
        "observed_chain_points": observed,
        "knee_angle_2d_deg": angle,
    }


def _body_scale(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    candidates: list[float] = []
    for leg in (left, right):
        hip, knee, ankle = leg["hip"], leg["knee"], leg["ankle"]
        if hip is not None and knee is not None:
            candidates.append(_euclidean(hip, knee))
        if knee is not None and ankle is not None:
            candidates.append(_euclidean(knee, ankle))
    if not candidates:
        return None
    return max(sum(candidates) / len(candidates), 1e-6)


def _sam3d_knee_evidence(analysis: dict[str, Any]) -> dict[str, Any]:
    profile = _first_dict(
        analysis.get("sam3d_relational_pose_profile_v16"),
        analysis.get("sam3d_relational_pose_profile"),
        analysis.get("sam3d_profile"),
        analysis.get("sam3d"),
    )
    projected = _first_dict(profile.get("sam3d_projected_pose"), profile.get("projected_pose"), profile)
    geometry = _first_dict(projected.get("geometry"), profile.get("geometry"))
    region = _first_dict(projected.get("region_support"), profile.get("region_support"))

    # Accommodate the field names used across the SAM3D profile revisions.
    left = None
    right = None
    for container in (geometry, projected, profile):
        left = left if left is not None else _finite(
            container.get("left_knee_angle_deg")
            or container.get("left_knee_flexion_angle_deg")
            or container.get("knee_angle_left_deg")
        )
        right = right if right is not None else _finite(
            container.get("right_knee_angle_deg")
            or container.get("right_knee_flexion_angle_deg")
            or container.get("knee_angle_right_deg")
        )

    authority = _finite(region.get("knees"))
    if authority is None:
        governance = _first_dict(projected.get("physical_governance"))
        auth = _first_dict(governance.get("authority"))
        authority = _finite(auth.get("knees") or auth.get("knee") or auth.get("crop_support"))

    return {
        "left_knee_angle_deg": left,
        "right_knee_angle_deg": right,
        "knee_authority": authority,
    }


def _semantic_relation(sheet: dict[str, Any]) -> dict[str, Any]:
    body = _first_dict(sheet.get("body"))
    config = body.get("configuration")
    if isinstance(config, list):
        texts = [str(item.get("text") if isinstance(item, dict) else item) for item in config]
    elif isinstance(config, dict):
        texts = [str(value.get("text") if isinstance(value, dict) else value) for value in config.values()]
    else:
        texts = []
    normalized = " | ".join(texts).lower()
    return {
        "texts": texts,
        "has_one_knee_raised": "one knee raised" in normalized,
        "has_one_leg_raised": "one leg raised" in normalized,
        "has_high_magnitude": any(token in normalized for token in ("raised high", "lifted high", "high knee")),
    }


def evaluate_sheet(sheet: dict[str, Any]) -> dict[str, Any]:
    analysis = _analysis_payload(sheet)
    route = _route(sheet, analysis)
    semantic = _semantic_relation(sheet)

    result: dict[str, Any] = {
        "report_only": True,
        "route": route,
        "semantic_relation": semantic,
        "decision": "abstain",
        "reason": None,
        "raised_side": None,
        "support_side": None,
        "raised_high": None,
    }

    if route not in POSE_BEARING_ROUTES:
        result["reason"] = "route_not_pose_bearing"
        return result
    if not (semantic["has_one_knee_raised"] or semantic["has_one_leg_raised"]):
        result["reason"] = "no_unilateral_raised_leg_semantic_candidate"
        return result

    joints = _dwpose_joint_source(analysis)
    left = _leg_measurements(joints, "left")
    right = _leg_measurements(joints, "right")
    result["dwpose"] = {"left": left, "right": right}

    if min(left["observed_chain_points"], right["observed_chain_points"]) < MIN_DWPOSE_LEG_CHAIN_POINTS:
        result["reason"] = "both_full_dwpose_leg_chains_not_observed"
        return result

    scale = _body_scale(left, right)
    if scale is None:
        result["reason"] = "body_scale_unavailable"
        return result

    # Image Y increases downward.  A raised knee sits above its same-side hip;
    # the support ankle should extend lower than the raised-side ankle.  These
    # are independent direct measurements, not opposite-side inference.
    direct: dict[str, dict[str, float | None]] = {}
    for side, leg in (("left", left), ("right", right)):
        direct[side] = {
            "knee_height_above_hip_body": (leg["hip"][1] - leg["knee"][1]) / scale,
            "knee_angle_2d_deg": leg["knee_angle_2d_deg"],
        }
    direct["left"]["ankle_down_advantage_body"] = (left["ankle"][1] - right["ankle"][1]) / scale
    direct["right"]["ankle_down_advantage_body"] = (right["ankle"][1] - left["ankle"][1]) / scale
    result["direct_leg_geometry"] = direct

    candidates: list[str] = []
    for side, other in (("left", "right"), ("right", "left")):
        raised_height = _finite(direct[side]["knee_height_above_hip_body"])
        other_height = _finite(direct[other]["knee_height_above_hip_body"])
        raised_angle = _finite(direct[side]["knee_angle_2d_deg"])
        support_angle = _finite(direct[other]["knee_angle_2d_deg"])
        support_down = _finite(direct[other]["ankle_down_advantage_body"])
        if None in (raised_height, other_height, raised_angle, support_angle, support_down):
            continue
        if (
            raised_height >= MIN_RAISED_KNEE_HEIGHT_BODY
            and (raised_height - other_height) >= MIN_RAISED_KNEE_HEIGHT_ADVANTAGE_BODY
            and raised_angle <= MAX_RAISED_KNEE_ANGLE_2D
            and support_angle >= MIN_SUPPORT_KNEE_ANGLE_2D
            and support_down >= MIN_SUPPORT_ANKLE_DOWN_ADVANTAGE_BODY
        ):
            candidates.append(side)

    sam = _sam3d_knee_evidence(analysis)
    result["sam3d_corroboration"] = sam

    if len(candidates) != 1:
        result["reason"] = "direct_leg_geometry_not_unambiguous"
        return result

    raised = candidates[0]
    support = "right" if raised == "left" else "left"
    sam_raised = _finite(sam.get(f"{raised}_knee_angle_deg"))
    sam_support = _finite(sam.get(f"{support}_knee_angle_deg"))
    sam_auth = _finite(sam.get("knee_authority"))

    sam_supports = False
    if None not in (sam_raised, sam_support, sam_auth):
        sam_supports = bool(
            sam_auth >= MIN_SAM3D_KNEE_AUTHORITY
            and sam_raised <= MAX_SAM3D_RAISED_KNEE_ANGLE
            and sam_support >= MIN_SAM3D_SUPPORT_KNEE_ANGLE
            and (sam_support - sam_raised) >= MIN_SAM3D_FLEXION_ADVANTAGE_DEG
        )
    result["sam3d_corroboration"]["supports_direct_binding"] = sam_supports

    # Direct DWPose topology is the authority.  SAM3D is deliberately only a
    # corroboration flag in v01 so population review can reveal whether making
    # it mandatory would create avoidable abstentions.
    result.update(
        {
            "decision": "candidate_would_bind",
            "reason": "direct_bilateral_leg_geometry_identifies_raised_and_support_sides",
            "raised_side": raised,
            "support_side": support,
            "raised_high": bool(
                _finite(direct[raised]["knee_height_above_hip_body"]) is not None
                and float(direct[raised]["knee_height_above_hip_body"]) >= 0.30
            ),
            "publication_candidate": {
                "raised_leg": f"{raised} knee raised" + (
                    " high" if float(direct[raised]["knee_height_above_hip_body"]) >= 0.30 else ""
                ),
                "support_leg": f"{support} leg supporting",
                "authority": "direct_dwpose_bilateral_leg_topology",
                "sam3d_corroborated": sam_supports,
            },
        }
    )
    return result


def _sheet_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def run(input_root: Path, output_root: Path) -> dict[str, int]:
    counts = {
        "total": 0,
        "candidate_would_bind": 0,
        "abstain": 0,
        "sam3d_corroborated": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for path in _sheet_paths(input_root):
        try:
            sheet = _load_json(path)
        except Exception:
            continue
        result = evaluate_sheet(sheet)
        counts["total"] += 1
        counts[result["decision"]] = counts.get(result["decision"], 0) + 1
        if result.get("sam3d_corroboration", {}).get("supports_direct_binding"):
            counts["sam3d_corroborated"] += 1
        relative = path.name if input_root.is_file() else path.relative_to(input_root)
        out = output_root / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = run(args.input, args.output)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
