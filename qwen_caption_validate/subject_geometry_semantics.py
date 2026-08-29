from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from typing import Any

from .analysis_v2_normalize import normalize_analysis_v2
from .runner import model_slug, resolve_model_id


FACT = "FACT"
CANDIDATE = "CANDIDATE"
WITHHELD = "WITHHELD"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _analysis_root(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else payload
    return value if isinstance(value, dict) else {}


def _fusion_root(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get("fusion") if isinstance(payload.get("fusion"), dict) else payload
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _axis_payload(analysis: dict[str, Any], field: str) -> dict[str, Any]:
    subject = analysis.get("target_subject") or {}
    orientation = subject.get("orientation") or {}
    value = orientation.get(field)
    return value if isinstance(value, dict) else {}


def _gaze_target(analysis: dict[str, Any]) -> str | None:
    subject = analysis.get("target_subject") or {}
    gaze = subject.get("gaze") or {}
    value = str(gaze.get("target") or "").strip().lower()
    return value or None


def _yaw_axis_validation(axis: dict[str, Any], band: str | None, *, face: bool = False) -> dict[str, Any]:
    direction = str(axis.get("direction") or "unknown").strip().lower()
    magnitude = str(axis.get("magnitude") or "unknown").strip().lower()
    confidence = _safe_float(axis.get("confidence"))
    usable = confidence is not None and confidence >= 0.60 and direction != "unknown"

    if not usable or not band:
        verdict = "unknown"
    elif face:
        if band == "toward_camera":
            agree = direction == "frontal" or magnitude in {"none", "slight"}
            disagree = direction == "back_to_camera" or (direction in {"anatomical_left", "anatomical_right"} and magnitude == "strong")
        elif band == "three_quarter":
            agree = direction in {"anatomical_left", "anatomical_right"} and magnitude in {"moderate", "strong"}
            disagree = direction in {"frontal", "back_to_camera"} and magnitude in {"none", "strong"}
        elif band == "profile":
            agree = direction in {"anatomical_left", "anatomical_right"} and magnitude == "strong"
            disagree = direction == "frontal" and magnitude in {"none", "slight"}
        elif band == "away_from_camera":
            agree = direction == "back_to_camera"
            disagree = direction == "frontal"
        else:
            agree = disagree = False
        verdict = "agree" if agree else "disagree" if disagree else "weak"
    else:
        if band == "frontal":
            agree = direction == "frontal" or magnitude == "none"
            disagree = direction == "back_to_camera" or (direction in {"anatomical_left", "anatomical_right"} and magnitude == "strong")
        elif band == "slightly_angled":
            agree = direction in {"anatomical_left", "anatomical_right"} and magnitude in {"slight", "moderate"}
            disagree = direction == "back_to_camera"
        elif band == "three_quarter":
            agree = direction in {"anatomical_left", "anatomical_right"} and magnitude in {"moderate", "strong"}
            disagree = direction in {"frontal", "back_to_camera"} and magnitude in {"none", "strong"}
        elif band == "side_on":
            agree = direction in {"anatomical_left", "anatomical_right"} and magnitude == "strong"
            disagree = direction == "frontal" and magnitude in {"none", "slight"}
        elif band in {"rear_three_quarter", "rear"}:
            agree = direction == "back_to_camera"
            disagree = direction == "frontal"
        else:
            agree = disagree = False
        verdict = "agree" if agree else "disagree" if disagree else "weak"

    return {
        "direction": direction,
        "magnitude": magnitude,
        "confidence": confidence,
        "verdict": verdict,
    }


def _fusion_provenance_usable(fusion: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    audit = fusion.get("sam3d_geometry_audit") or {}
    provenance = audit.get("target_provenance") or {}
    risk = provenance.get("context_risk")
    return risk != "requires_review", provenance if isinstance(provenance, dict) else {}


def _fusion_body_validation(fusion: dict[str, Any], body_band: str | None) -> dict[str, Any]:
    upper = fusion.get("qualified_upper_torso_depth_relation")
    if not isinstance(upper, dict):
        return {"available": False, "verdict": "unknown"}
    authority = str(upper.get("authority") or "")
    magnitude = _safe_float(upper.get("source_magnitude_deg"))
    relation = str(upper.get("relation") or "")
    qualified = authority == "qualified_visible_shoulder_depth_rotation"
    if not qualified or not body_band:
        verdict = "unknown"
    elif body_band == "side_on":
        verdict = "agree" if ((magnitude is not None and magnitude >= 65.0) or "side-on" in relation.lower()) else "weak"
    elif body_band == "three_quarter":
        verdict = "agree" if (magnitude is None or magnitude >= 35.0) else "weak"
    elif body_band == "slightly_angled":
        # The legacy shoulder-depth qualifier was intentionally tuned for strong
        # turns.  Its presence can support a modest angle, but absence is not
        # counterevidence.
        verdict = "agree" if magnitude is not None and 20.0 <= magnitude <= 55.0 else "weak"
    elif body_band == "frontal":
        verdict = "disagree" if (magnitude is not None and magnitude >= 45.0) or "side-on" in relation.lower() else "weak"
    else:
        verdict = "weak"
    return {
        "available": True,
        "authority": authority,
        "source_magnitude_deg": magnitude,
        "relation": relation or None,
        "verdict": verdict,
    }


def _fusion_head_validation(fusion: dict[str, Any]) -> dict[str, Any]:
    relation = fusion.get("qualified_head_torso_relation")
    if not isinstance(relation, dict):
        return {"available": False, "verdict": "unknown"}
    camera_relation = str(relation.get("camera_relation") or "").strip().lower()
    return {
        "available": True,
        "camera_relation": camera_relation or None,
        "magnitude": relation.get("magnitude"),
        "relation": relation.get("relation"),
        "authority": relation.get("authority"),
        "verdict": "agree" if camera_relation == "toward_camera" else "weak",
    }


def _status_payload(
    status: str,
    value: dict[str, Any] | None,
    *,
    authority: str,
    reasons: list[str],
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "value": value,
        "authority": authority,
        "reasons": reasons,
        "limitations": list(limitations or []),
    }


def build_subject_geometry_semantics(
    diagnostic: dict[str, Any],
    analysis_payload: dict[str, Any] | None,
    fusion_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Govern calibrated SAM3D subject geometry into FACT/CANDIDATE/WITHHELD.

    The stage is deliberately narrow.  It promotes camera-relative body/face yaw
    only when reconstructed geometry is visibility-gated by DWPose and has an
    independent semantic/geometric corroborator.  Subject-relative camera
    position/aim remains CANDIDATE in v0.1 because capture mode and world-up are
    outside this stage.
    """
    analysis = _analysis_root(analysis_payload)
    fusion = _fusion_root(fusion_payload)
    body = diagnostic.get("body_camera_relation") or {}
    face = diagnostic.get("face_camera_relation") or {}
    camera = diagnostic.get("camera_relative_subject") or {}
    visibility = diagnostic.get("dwpose_visibility_gate") or {}
    compound = diagnostic.get("compound_pose_hint")
    landmark_check = ((diagnostic.get("body_frame_landmarks") or {}).get("canonical_lateral_axis_check") or {})

    body_band = body.get("orientation_band")
    face_band = face.get("orientation_band")
    analyze_body = _yaw_axis_validation(_axis_payload(analysis, "torso_yaw"), body_band, face=False)
    analyze_face = _yaw_axis_validation(_axis_payload(analysis, "head_yaw"), face_band, face=True)
    gaze_target = _gaze_target(analysis)
    gaze_camera_support = gaze_target in {"camera_lens", "near_camera"}
    fusion_body = _fusion_body_validation(fusion, body_band)
    fusion_head = _fusion_head_validation(fusion)
    provenance_usable, provenance = _fusion_provenance_usable(fusion)

    body_reasons: list[str] = []
    body_limits: list[str] = []
    if not visibility.get("body_yaw_observation_gate"):
        body_status = WITHHELD
        body_reasons.append("bilateral shoulders are not independently observed by DWPose")
    elif not provenance_usable:
        body_status = WITHHELD
        body_reasons.append("Fusion marks SAM3D target provenance as requiring review")
    elif analyze_body["verdict"] == "disagree" or fusion_body.get("verdict") == "disagree":
        body_status = CANDIDATE
        body_reasons.append("SAM3D body yaw conflicts with an independent orientation source")
        body_limits.append("manual review recommended before caption use")
    elif analyze_body["verdict"] == "agree" or fusion_body.get("verdict") == "agree":
        body_status = FACT
        body_reasons.append("DWPose-observed shoulders gate the SAM3D root/body yaw")
        if analyze_body["verdict"] == "agree":
            body_reasons.append("Analyze torso-yaw semantics independently agree with the orientation band")
        if fusion_body.get("verdict") == "agree":
            body_reasons.append("qualified Fusion shoulder-depth geometry independently agrees with the body turn")
    else:
        body_status = CANDIDATE
        body_reasons.append("SAM3D body yaw is visibility-gated but lacks independent corroboration")

    body_value = {
        "yaw_deg": body.get("yaw_deg"),
        "orientation": body_band,
        "faces_frame": body.get("faces_frame"),
    } if body_band else None

    face_reasons: list[str] = []
    face_limits: list[str] = []
    if not visibility.get("face_yaw_observation_gate"):
        face_status = WITHHELD
        face_reasons.append("nose, both eyes, and both ears are not independently observed by DWPose")
    elif not provenance_usable:
        face_status = WITHHELD
        face_reasons.append("Fusion marks SAM3D target provenance as requiring review")
    elif analyze_face["verdict"] == "disagree":
        face_status = CANDIDATE
        face_reasons.append("SAM3D face yaw conflicts with Analyze head-yaw semantics")
        face_limits.append("manual review recommended before caption use")
    elif analyze_face["verdict"] == "agree" or (face_band == "toward_camera" and gaze_camera_support):
        face_status = FACT
        face_reasons.append("DWPose-observed facial landmarks gate the reconstructed face-forward yaw")
        if analyze_face["verdict"] == "agree":
            face_reasons.append("Analyze head-yaw semantics independently agree with the face orientation")
        if face_band == "toward_camera" and gaze_camera_support:
            face_reasons.append("Analyze gaze target independently places the subject at or near the camera lens")
    else:
        face_status = CANDIDATE
        face_reasons.append("SAM3D face yaw is visibility-gated but lacks independent semantic corroboration")

    face_value = {
        "yaw_deg": face.get("yaw_deg"),
        "orientation": face_band,
    } if face_band else None

    head_reasons: list[str] = []
    head_limits: list[str] = []
    if not isinstance(compound, dict) or compound.get("head_relation") != "turned_toward_camera":
        head_status = WITHHELD
        head_value = None
        head_reasons.append("no salient compensating head-turn relation was proposed by the calibrated geometry")
    elif body_status == FACT and face_status == FACT:
        head_status = FACT
        head_value = {
            "relation": "turned_toward_camera",
            "turn_toward_camera_deg": face.get("head_turn_toward_camera_deg"),
            "body_orientation": body_band,
            "body_faces_frame": body.get("faces_frame"),
            "face_orientation": face_band,
        }
        head_reasons.append("qualified body and face yaw jointly establish a compensating turn toward the camera")
        if fusion_head.get("verdict") == "agree":
            head_reasons.append("Fusion independently qualifies the same head-to-torso camera relation")
    else:
        head_status = CANDIDATE
        head_value = {
            "relation": "turned_toward_camera",
            "turn_toward_camera_deg": face.get("head_turn_toward_camera_deg"),
            "body_orientation": body_band,
            "body_faces_frame": body.get("faces_frame"),
            "face_orientation": face_band,
        }
        head_reasons.append("compound head turn is geometrically plausible but one or more prerequisite orientations are not FACT")
        head_limits.append("candidate remains audit-only")

    camera_reasons: list[str] = []
    if visibility.get("face_yaw_observation_gate"):
        camera_status = CANDIDATE
        camera_reasons.append("subject-relative camera center/aim is calibrated geometry with observed facial reference landmarks")
        camera_reasons.append("capture mode and world-up are not established by this stage, so camera geometry is not promoted to caption FACT")
    else:
        camera_status = WITHHELD
        camera_reasons.append("eye-relative camera position is withheld without independently observed facial landmarks")
    camera_value = {
        "vertical_vs_eye_m": camera.get("vertical_vs_eye"),
        "vertical_vs_shoulders_m": camera.get("vertical_vs_shoulders"),
        "side": camera.get("side") if landmark_check.get("plus_x_is_subject_left") else None,
        "optical_axis_pitch_deg": camera.get("optical_axis_pitch_deg"),
        "optical_axis_yaw_deg": camera.get("optical_axis_yaw_deg"),
        "camera_pose_pattern": camera.get("camera_pose_pattern") or camera.get("selfie_like_geometry"),
    }

    body_semantic = _status_payload(
        body_status,
        body_value,
        authority="dwpose_visibility_plus_sam3d_root_geometry_plus_cross_source_validation",
        reasons=body_reasons,
        limitations=body_limits,
    )
    face_semantic = _status_payload(
        face_status,
        face_value,
        authority="dwpose_face_visibility_plus_sam3d_face_proxy_plus_analyze_validation",
        reasons=face_reasons,
        limitations=face_limits,
    )
    head_semantic = _status_payload(
        head_status,
        head_value,
        authority="qualified_body_face_relative_geometry",
        reasons=head_reasons,
        limitations=head_limits,
    )
    camera_semantic = _status_payload(
        camera_status,
        camera_value if camera_status != WITHHELD else None,
        authority="sam3d_subject_relative_camera_geometry",
        reasons=camera_reasons,
        limitations=["never interpret as world high/low or selfie/external capture mode in subject-geometry-semantics-0.1"],
    )

    preferred = {
        "body_orientation": body_value if body_status == FACT else None,
        "face_orientation": face_value if face_status == FACT else None,
        "head_body_relation": head_value if head_status == FACT else None,
        "camera_subject_relation": None,
    }
    compound_fact = None
    if head_status == FACT and isinstance(head_value, dict):
        compound_fact = {
            "body_orientation": head_value.get("body_orientation"),
            "body_faces_frame": head_value.get("body_faces_frame"),
            "head_relation": "turned_toward_camera",
            "face_orientation": head_value.get("face_orientation"),
        }

    return {
        "schema_version": "subject-geometry-semantics-0.1",
        "body_orientation": body_semantic,
        "face_orientation": face_semantic,
        "head_body_relation": head_semantic,
        "camera_subject_relation": camera_semantic,
        "preferred_orientation": preferred,
        "compound_pose_fact": compound_fact,
        "cross_source_validation": {
            "analyze_torso_yaw": analyze_body,
            "analyze_head_yaw": analyze_face,
            "analyze_gaze_target": gaze_target,
            "analyze_gaze_camera_support": gaze_camera_support,
            "fusion_upper_torso_depth": fusion_body,
            "fusion_head_torso_relation": fusion_head,
            "fusion_target_provenance": provenance,
            "fusion_target_provenance_usable": provenance_usable,
        },
        "observation_gates": visibility,
        "policy": {
            "fact_candidate_withheld": True,
            "body_fact_requires_dwpose_shoulders_and_independent_corroboration": True,
            "face_fact_requires_dwpose_face_and_independent_corroboration": True,
            "compound_head_turn_requires_fact_body_and_face": True,
            "camera_subject_relation_is_candidate_only": True,
            "face_pitch_is_not_semantic_authority": True,
            "body_pitch_is_not_semantic_authority": True,
        },
    }


def _matches(key: str, only: list[str]) -> bool:
    if not only:
        return True
    return any(fnmatch.fnmatch(key, pattern) or pattern in key for pattern in only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-subject-geometry-semantics",
        description=(
            "Govern cached SAM3D subject geometry into FACT/CANDIDATE/WITHHELD body, face, head/body, and subject-relative camera semantics."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-model", default="32b-fp8")
    parser.add_argument("--subject-diagnostic", type=Path)
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    analysis_model_id = resolve_model_id(args.analysis_model)
    slug = model_slug(analysis_model_id)
    analysis_dir = run_dir / slug
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.7" / slug)).expanduser().resolve()
    diagnostic_path = (
        args.subject_diagnostic or (run_dir / "sam3d" / "sam3d_subject_geometry_diagnostic.json")
    ).expanduser().resolve()
    output_dir = (
        args.output_dir or (run_dir / "subject-geometry-semantics-v0.1" / slug)
    ).expanduser().resolve()

    diagnostic_run = _read_json(diagnostic_path)
    if diagnostic_run is None:
        print(f"Subject geometry diagnostic not found: {diagnostic_path}", file=sys.stderr)
        return 2

    records = [
        item
        for item in (diagnostic_run.get("records") or [])
        if isinstance(item, dict) and item.get("image_key") and _matches(str(item.get("image_key")), args.only)
    ]
    if not records:
        print("No matching subject geometry diagnostic records found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    index_records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for item in records:
        key = str(item.get("image_key"))
        out_path = output_dir / f"{key}.subject_geometry_semantics.json"
        if out_path.exists() and not args.overwrite:
            payload = _read_json(out_path) or {}
            semantics = payload.get("subject_geometry_semantics") or {}
            state = "reused"
        else:
            diagnostic = item.get("diagnostic") if isinstance(item.get("diagnostic"), dict) else {}
            analysis_record = _read_json(analysis_dir / f"{key}.analysis.json")
            if analysis_record is not None:
                raw = _analysis_root(analysis_record)
                analysis, _ = normalize_analysis_v2(raw)
            else:
                analysis = {}
            fusion = _read_json(fusion_dir / f"{key}.fused_v2_3.json")
            semantics = build_subject_geometry_semantics(diagnostic, analysis, fusion)
            payload = {
                "image_key": key,
                "source_subject_diagnostic": str(diagnostic_path),
                "source_analysis": str(analysis_dir / f"{key}.analysis.json") if analysis_record is not None else None,
                "source_fusion": str(fusion_dir / f"{key}.fused_v2_3.json") if fusion is not None else None,
                "subject_geometry_semantics": semantics,
            }
            _write_json(out_path, payload)
            state = "written"

        body_status = str((semantics.get("body_orientation") or {}).get("status") or "UNKNOWN")
        face_status = str((semantics.get("face_orientation") or {}).get("status") or "UNKNOWN")
        head_status = str((semantics.get("head_body_relation") or {}).get("status") or "UNKNOWN")
        for label in (body_status, face_status, head_status):
            status_counts[label] = status_counts.get(label, 0) + 1
        body_value = (semantics.get("body_orientation") or {}).get("value") or {}
        face_value = (semantics.get("face_orientation") or {}).get("value") or {}
        index_records.append(
            {
                "image_key": key,
                "status": state,
                "body_status": body_status,
                "body_orientation": body_value.get("orientation"),
                "body_faces_frame": body_value.get("faces_frame"),
                "face_status": face_status,
                "face_orientation": face_value.get("orientation"),
                "head_body_status": head_status,
                "compound_pose_fact": semantics.get("compound_pose_fact"),
                "camera_subject_status": (semantics.get("camera_subject_relation") or {}).get("status"),
            }
        )
        print(
            f"{key}: body={body_status}:{body_value.get('orientation') or '-'} "
            f"{body_value.get('faces_frame') or '-'}; face={face_status}:{face_value.get('orientation') or '-'}; "
            f"head={head_status}; compound={'yes' if semantics.get('compound_pose_fact') else 'no'}"
        )

    index = {
        "schema_version": "subject-geometry-semantics-run-0.1",
        "run_dir": str(run_dir),
        "analysis_model": analysis_model_id,
        "analysis_source": str(analysis_dir),
        "fusion_source": str(fusion_dir),
        "subject_diagnostic": str(diagnostic_path),
        "output_dir": str(output_dir),
        "record_count": len(index_records),
        "status_counts": status_counts,
        "records": index_records,
    }
    index_path = output_dir / "subject_geometry_semantics.index.json"
    _write_json(index_path, index)
    print(f"Subject geometry semantics: {output_dir}")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
