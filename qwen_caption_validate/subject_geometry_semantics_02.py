from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

from .analysis_v2_normalize import normalize_analysis_v2
from .runner import model_slug, resolve_model_id
from . import subject_geometry_semantics as v01


FACT = v01.FACT
CANDIDATE = v01.CANDIDATE
WITHHELD = v01.WITHHELD


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    """Promote calibrated subject-relative geometry without making Analyze its veto.

    v0.1 required Analyze/Fusion corroboration before calibrated SAM3D yaw could
    become FACT.  That defeated the purpose of the new channel in exactly the
    cases it is meant to fix: a camera-facing head can cause Analyze to call a
    nearly side-on torso frontal.  v0.2 therefore makes calibrated SAM3D geometry
    the estimator and DWPose visibility the observation gate.  Analyze is a
    validator/audit comparator, not a veto.  A contradictory independently
    qualified Fusion geometry relation can still demote a body claim.

    Camera center/aim is promoted only as a *subject-relative* geometric FACT.
    It is explicitly forbidden from becoming world high/low or capture-mode
    semantics until a later posture/capture governor supplies that context.
    """
    analysis = v01._analysis_root(analysis_payload)
    fusion = v01._fusion_root(fusion_payload)
    body = diagnostic.get("body_camera_relation") or {}
    face = diagnostic.get("face_camera_relation") or {}
    camera = diagnostic.get("camera_relative_subject") or {}
    visibility = diagnostic.get("dwpose_visibility_gate") or {}
    compound = diagnostic.get("compound_pose_hint")
    landmark_check = ((diagnostic.get("body_frame_landmarks") or {}).get("canonical_lateral_axis_check") or {})

    body_band = body.get("orientation_band")
    face_band = face.get("orientation_band")
    analyze_body = v01._yaw_axis_validation(v01._axis_payload(analysis, "torso_yaw"), body_band, face=False)
    analyze_face = v01._yaw_axis_validation(v01._axis_payload(analysis, "head_yaw"), face_band, face=True)
    gaze_target = v01._gaze_target(analysis)
    gaze_camera_support = gaze_target in {"camera_lens", "near_camera"}
    fusion_body = v01._fusion_body_validation(fusion, body_band)
    fusion_head = v01._fusion_head_validation(fusion)
    provenance_usable, provenance = v01._fusion_provenance_usable(fusion)

    cross_source_conflicts: list[dict[str, Any]] = []

    body_value = {
        "yaw_deg": body.get("yaw_deg"),
        "orientation": body_band,
        "faces_frame": body.get("faces_frame"),
    } if body_band else None

    body_reasons: list[str] = []
    body_limits: list[str] = []
    if not body_band:
        body_status = WITHHELD
        body_reasons.append("SAM3D body/root yaw is unavailable")
    elif not visibility.get("body_yaw_observation_gate"):
        body_status = WITHHELD
        body_reasons.append("bilateral shoulders are not independently observed by DWPose")
    elif not provenance_usable:
        body_status = WITHHELD
        body_reasons.append("Fusion marks SAM3D target provenance as requiring review")
    elif fusion_body.get("verdict") == "disagree":
        body_status = CANDIDATE
        body_reasons.append("independently qualified Fusion shoulder-depth geometry conflicts with calibrated SAM3D body yaw")
        body_limits.append("manual review recommended before caption use")
        cross_source_conflicts.append({
            "field": "body_orientation",
            "source": "fusion.qualified_upper_torso_depth_relation",
            "sam3d_value": body_value,
            "other_source": fusion_body,
            "effect": "demote_to_candidate",
        })
    else:
        body_status = FACT
        body_reasons.append("calibrated SAM3D root/body yaw estimates camera-relative torso orientation")
        body_reasons.append("DWPose independently observes both shoulders, gating the reconstructed body orientation to visible anatomy")
        if fusion_body.get("verdict") == "agree":
            body_reasons.append("qualified Fusion shoulder-depth geometry independently corroborates the body turn")
        if analyze_body.get("verdict") == "agree":
            body_reasons.append("Analyze torso-yaw semantics independently corroborate the orientation band")
        elif analyze_body.get("verdict") == "disagree":
            body_limits.append("Analyze torso-yaw semantics disagree; calibrated geometry is retained and the disagreement is audit-visible")
            cross_source_conflicts.append({
                "field": "body_orientation",
                "source": "analyze.target_subject.orientation.torso_yaw",
                "sam3d_value": body_value,
                "other_source": analyze_body,
                "effect": "audit_only_geometry_retained",
            })
        elif analyze_body.get("verdict") in {"weak", "unknown"}:
            body_limits.append("Analyze does not independently resolve this body-yaw magnitude")

    face_value = {
        "yaw_deg": face.get("yaw_deg"),
        "orientation": face_band,
    } if face_band else None

    face_reasons: list[str] = []
    face_limits: list[str] = []
    if not face_band:
        face_status = WITHHELD
        face_reasons.append("SAM3D reconstructed face yaw is unavailable")
    elif not visibility.get("face_yaw_observation_gate"):
        face_status = WITHHELD
        face_reasons.append("nose, both eyes, and both ears are not independently observed by DWPose")
    elif not provenance_usable:
        face_status = WITHHELD
        face_reasons.append("Fusion marks SAM3D target provenance as requiring review")
    else:
        face_status = FACT
        face_reasons.append("calibrated reconstructed face-forward yaw estimates face orientation relative to camera")
        face_reasons.append("DWPose independently observes nose, both eyes, and both ears, gating the face proxy to visible anatomy")
        if analyze_face.get("verdict") == "agree":
            face_reasons.append("Analyze head-yaw semantics independently corroborate the face orientation")
        elif analyze_face.get("verdict") == "disagree":
            face_limits.append("Analyze head-yaw semantics disagree; calibrated geometry is retained and the disagreement is audit-visible")
            cross_source_conflicts.append({
                "field": "face_orientation",
                "source": "analyze.target_subject.orientation.head_yaw",
                "sam3d_value": face_value,
                "other_source": analyze_face,
                "effect": "audit_only_geometry_retained",
            })
        if face_band == "toward_camera" and gaze_camera_support:
            face_reasons.append("Analyze gaze target independently places the subject at or near the camera lens")

    head_reasons: list[str] = []
    head_limits: list[str] = []
    if not isinstance(compound, dict) or compound.get("head_relation") != "turned_toward_camera":
        head_status = WITHHELD
        head_value = None
        head_reasons.append("no salient compensating head-turn relation was proposed by the calibrated geometry")
    else:
        head_value = {
            "relation": "turned_toward_camera",
            "turn_toward_camera_deg": face.get("head_turn_toward_camera_deg"),
            "body_orientation": body_band,
            "body_faces_frame": body.get("faces_frame"),
            "face_orientation": face_band,
        }
        if body_status == FACT and face_status == FACT:
            head_status = FACT
            head_reasons.append("FACT body and face yaw jointly establish a salient compensating head turn toward the camera")
            if fusion_head.get("verdict") == "agree":
                head_reasons.append("Fusion independently qualifies the same head-to-torso camera relation")
        else:
            head_status = CANDIDATE
            head_reasons.append("compound head turn is geometrically plausible but one or more prerequisite orientations are not FACT")
            head_limits.append("candidate remains audit-only")

    lateral_valid = bool(landmark_check.get("plus_x_is_subject_left"))
    camera_value = {
        "vertical_vs_eye_m": camera.get("vertical_vs_eye"),
        "vertical_vs_shoulders_m": camera.get("vertical_vs_shoulders"),
        "side": camera.get("side") if lateral_valid else None,
        "optical_axis_pitch_deg": camera.get("optical_axis_pitch_deg"),
        "optical_axis_yaw_deg": camera.get("optical_axis_yaw_deg"),
        "camera_pose_pattern": camera.get("camera_pose_pattern") or camera.get("selfie_like_geometry"),
        "interpretation_scope": "subject_relative_only",
    }
    camera_reasons: list[str] = []
    camera_limits: list[str] = [
        "must not be converted directly into world high/low camera elevation",
        "must not infer selfie/external capture mode",
        "requires a later posture/capture governor before caption-level photographic interpretation",
    ]
    if not visibility.get("body_yaw_observation_gate"):
        camera_status = WITHHELD
        camera_reasons.append("subject-relative camera transform is withheld without independently observed bilateral shoulders")
    elif not visibility.get("face_yaw_observation_gate"):
        camera_status = WITHHELD
        camera_reasons.append("eye-relative camera transform is withheld without independently observed facial landmarks")
    elif not provenance_usable:
        camera_status = WITHHELD
        camera_reasons.append("Fusion marks SAM3D target provenance as requiring review")
    else:
        camera_status = FACT
        camera_reasons.append("SAM3D camera translation and calibrated body/root rotation establish camera center and optical axis in the subject frame")
        camera_reasons.append("DWPose independently observes the shoulders and facial landmarks used to gate the subject-relative transform")
        if not lateral_valid:
            camera_limits.append("anatomical camera side withheld because reconstructed lateral-axis sign check failed")

    body_semantic = _status_payload(
        body_status,
        body_value if body_status != WITHHELD else None,
        authority="calibrated_sam3d_root_geometry_visibility_gated_by_dwpose",
        reasons=body_reasons,
        limitations=body_limits,
    )
    face_semantic = _status_payload(
        face_status,
        face_value if face_status != WITHHELD else None,
        authority="calibrated_sam3d_face_proxy_visibility_gated_by_dwpose",
        reasons=face_reasons,
        limitations=face_limits,
    )
    head_semantic = _status_payload(
        head_status,
        head_value if head_status != WITHHELD else None,
        authority="qualified_body_face_relative_geometry",
        reasons=head_reasons,
        limitations=head_limits,
    )
    camera_semantic = _status_payload(
        camera_status,
        camera_value if camera_status != WITHHELD else None,
        authority="sam3d_subject_relative_camera_geometry_visibility_gated_by_dwpose",
        reasons=camera_reasons,
        limitations=camera_limits,
    )

    preferred_orientation = {
        "body_orientation": body_value if body_status == FACT else None,
        "face_orientation": face_value if face_status == FACT else None,
        "head_body_relation": head_value if head_status == FACT else None,
        # Deliberately excluded from caption-facing orientation.  The next
        # camera governor must interpret this subject-relative fact first.
        "camera_subject_relation": None,
    }
    preferred_subject_geometry = {
        "camera_subject_relation": camera_value if camera_status == FACT else None,
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
        "schema_version": "subject-geometry-semantics-0.2",
        "body_orientation": body_semantic,
        "face_orientation": face_semantic,
        "head_body_relation": head_semantic,
        "camera_subject_relation": camera_semantic,
        "preferred_orientation": preferred_orientation,
        "preferred_subject_geometry": preferred_subject_geometry,
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
            "conflicts": cross_source_conflicts,
        },
        "observation_gates": visibility,
        "policy": {
            "fact_candidate_withheld": True,
            "sam3d_is_calibrated_geometry_estimator": True,
            "dwpose_visibility_is_observation_gate_not_yaw_estimator": True,
            "analyze_is_validator_not_veto": True,
            "qualified_fusion_geometry_may_demote_body_on_conflict": True,
            "compound_head_turn_requires_fact_body_and_face": True,
            "camera_subject_relation_can_be_fact_only_in_subject_frame": True,
            "camera_subject_fact_is_not_caption_photographic_elevation": True,
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
        prog="qwen-subject-geometry-semantics-02",
        description=(
            "Govern calibrated SAM3D subject geometry into FACT/CANDIDATE/WITHHELD body, face, head/body, and subject-relative camera semantics."
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
        args.output_dir or (run_dir / "subject-geometry-semantics-v0.2" / slug)
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
                raw = v01._analysis_root(analysis_record)
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

        body_sem = semantics.get("body_orientation") or {}
        face_sem = semantics.get("face_orientation") or {}
        head_sem = semantics.get("head_body_relation") or {}
        cam_sem = semantics.get("camera_subject_relation") or {}
        body_status = str(body_sem.get("status") or "UNKNOWN")
        face_status = str(face_sem.get("status") or "UNKNOWN")
        head_status = str(head_sem.get("status") or "UNKNOWN")
        camera_status = str(cam_sem.get("status") or "UNKNOWN")
        for label in (body_status, face_status, head_status, camera_status):
            status_counts[label] = status_counts.get(label, 0) + 1
        body_value = body_sem.get("value") or {}
        face_value = face_sem.get("value") or {}
        index_records.append({
            "image_key": key,
            "status": state,
            "body_status": body_status,
            "body_orientation": body_value.get("orientation"),
            "body_faces_frame": body_value.get("faces_frame"),
            "face_status": face_status,
            "face_orientation": face_value.get("orientation"),
            "head_body_status": head_status,
            "camera_subject_status": camera_status,
            "compound_pose_fact": semantics.get("compound_pose_fact"),
            "cross_source_conflict_count": len(((semantics.get("cross_source_validation") or {}).get("conflicts") or [])),
        })
        print(
            f"{key}: body={body_status}:{body_value.get('orientation') or '-'} {body_value.get('faces_frame') or '-'}; "
            f"face={face_status}:{face_value.get('orientation') or '-'}; "
            f"head={head_status}; camera={camera_status}; "
            f"compound={'yes' if semantics.get('compound_pose_fact') else 'no'}; "
            f"conflicts={len(((semantics.get('cross_source_validation') or {}).get('conflicts') or []))}"
        )

    index = {
        "schema_version": "subject-geometry-semantics-run-0.2",
        "run_dir": str(run_dir),
        "analysis_model": analysis_model_id,
        "subject_diagnostic": str(diagnostic_path),
        "fusion_dir": str(fusion_dir),
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
