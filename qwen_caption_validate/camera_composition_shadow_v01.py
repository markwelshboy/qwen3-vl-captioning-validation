from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .sam3d_subject_geometry_diagnostic_02 import build_subject_geometry

SCHEMA_VERSION = "camera-composition-shadow-0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_FRAMING_SUBDIR = Path("semantic-v3") / "framing-semantics-v1.0"
DEFAULT_OBSERVER_SUBDIR = Path("semantic-v3") / "selfie-composition-observer-v0.1"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "camera-composition-shadow-v0.1"

MHR_LEFT_SHOULDER = 5
MHR_RIGHT_SHOULDER = 6

# Provisional validation thresholds, intentionally not production authority.
ELEVATED_MIN_VERTICAL_VS_EYE = 0.15
ELEVATED_MAX_DOWNWARD_PITCH = -10.0
DOWNWARD_MIN_VERTICAL_VS_EYE = -0.05
DOWNWARD_MAX_PITCH = -8.0
SHOULDER_CLEAR_DELTA_WIDTHS = 0.10
SHOULDER_WEAK_DELTA_WIDTHS = 0.05


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _observed_shoulders(dwpose: dict[str, Any] | None) -> dict[str, bool]:
    target = (((dwpose or {}).get("derived") or {}).get("target") or {})
    visible = {
        str(v)
        for v in (target.get("visible_body_landmarks") or [])
        if isinstance(v, str)
    }
    return {
        "left": "left_shoulder" in visible,
        "right": "right_shoulder" in visible,
    }


def _camera_viewpoint(geometry: dict[str, Any]) -> dict[str, Any]:
    camera = (
        geometry.get("camera_relative_subject")
        if isinstance(geometry.get("camera_relative_subject"), dict)
        else {}
    )
    vertical = camera.get("vertical_vs_eye")
    pitch = camera.get("optical_axis_pitch_deg")
    vertical = float(vertical) if isinstance(vertical, (int, float)) else None
    pitch = float(pitch) if isinstance(pitch, (int, float)) else None

    classification = "withheld"
    composer_text = None
    publishable_candidate = False
    reason = "geometry_does_not_clear_provisional_viewpoint_thresholds"

    if vertical is not None and pitch is not None:
        if vertical >= ELEVATED_MIN_VERTICAL_VS_EYE and pitch <= ELEVATED_MAX_DOWNWARD_PITCH:
            classification = "elevated_downward"
            composer_text = "from a slightly elevated, downward-angled camera viewpoint"
            publishable_candidate = True
            reason = "camera_above_eye_line_and_optical_axis_clearly_aimed_down"
        elif vertical >= DOWNWARD_MIN_VERTICAL_VS_EYE and pitch <= DOWNWARD_MAX_PITCH:
            classification = "downward_aimed"
            composer_text = "with the camera angled slightly downward"
            publishable_candidate = True
            reason = "optical_axis_clearly_aimed_down_without_strong_elevation"
        elif vertical <= -0.15 and pitch >= 8.0:
            classification = "below_upward_diagnostic_only"
            reason = "reverse_geometry_present_but_not_calibrated_for_caption_promotion"

    return {
        "classification": classification,
        "vertical_vs_eye": vertical,
        "vertical_band": camera.get("vertical_band"),
        "optical_axis_pitch_deg": pitch,
        "camera_pose_pattern": camera.get("camera_pose_pattern"),
        "publishable_candidate": publishable_candidate,
        "composer_text": composer_text,
        "reason": reason,
        "authority": "sam3d_subject_relative_geometry_shadow",
    }


def _shoulder_depth(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
) -> dict[str, Any]:
    observed = _observed_shoulders(dwpose)
    if not (observed["left"] and observed["right"]):
        return {
            "status": "withheld",
            "reason": "bilateral_shoulders_not_observed_by_dwpose",
            "observed_shoulders": observed,
            "publishable_candidate": False,
        }

    points = np.asarray(arrays.get("pred_keypoints_3d"), dtype=np.float64)
    cam_t = np.asarray(arrays.get("pred_cam_t"), dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[0] <= MHR_RIGHT_SHOULDER or points.shape[1] < 3 or cam_t.size < 3:
        return {
            "status": "withheld",
            "reason": "sam3d_shoulder_geometry_unavailable",
            "observed_shoulders": observed,
            "publishable_candidate": False,
        }

    left_root = points[MHR_LEFT_SHOULDER, :3]
    right_root = points[MHR_RIGHT_SHOULDER, :3]
    left_cam = left_root + cam_t[:3]
    right_cam = right_root + cam_t[:3]
    if not (np.all(np.isfinite(left_cam)) and np.all(np.isfinite(right_cam))):
        return {
            "status": "withheld",
            "reason": "nonfinite_sam3d_shoulder_geometry",
            "observed_shoulders": observed,
            "publishable_candidate": False,
        }

    shoulder_width = float(np.linalg.norm(left_root - right_root))
    left_distance = float(np.linalg.norm(left_cam))
    right_distance = float(np.linalg.norm(right_cam))
    distance_delta = right_distance - left_distance
    distance_delta_widths = (
        distance_delta / shoulder_width
        if shoulder_width > 1e-9
        else None
    )

    left_z = float(left_cam[2])
    right_z = float(right_cam[2])
    z_delta = right_z - left_z
    z_nearer = "left" if z_delta > 0 else ("right" if z_delta < 0 else None)
    distance_nearer = (
        "left"
        if distance_delta > 0
        else ("right" if distance_delta < 0 else None)
    )
    agree = bool(distance_nearer and distance_nearer == z_nearer)

    magnitude = abs(float(distance_delta_widths)) if distance_delta_widths is not None else 0.0
    if agree and magnitude >= SHOULDER_CLEAR_DELTA_WIDTHS:
        status = "clear"
        reason = "camera_distance_and_depth_order_agree_beyond_clear_deadband"
    elif agree and magnitude >= SHOULDER_WEAK_DELTA_WIDTHS:
        status = "weak"
        reason = "camera_distance_and_depth_order_agree_but_only_weakly"
    elif not agree:
        status = "conflict"
        reason = "camera_distance_and_optical_depth_order_do_not_agree"
    else:
        status = "ambiguous"
        reason = "shoulder_depth_difference_inside_deadband"

    composer_text = (
        f"the {distance_nearer} shoulder is nearer the camera"
        if status == "clear" and distance_nearer
        else None
    )

    return {
        "status": status,
        "observed_shoulders": observed,
        "anatomical_side_nearer": distance_nearer if status in {"clear", "weak"} else None,
        "left_camera_distance": round(left_distance, 6),
        "right_camera_distance": round(right_distance, 6),
        "distance_delta_right_minus_left": round(distance_delta, 6),
        "distance_delta_shoulder_widths": round(float(distance_delta_widths), 4)
        if distance_delta_widths is not None
        else None,
        "left_camera_z": round(left_z, 6),
        "right_camera_z": round(right_z, 6),
        "z_delta_right_minus_left": round(z_delta, 6),
        "distance_nearer_side": distance_nearer,
        "z_nearer_side": z_nearer,
        "distance_and_z_order_agree": agree,
        "publishable_candidate": status == "clear",
        "composer_text": composer_text,
        "reason": reason,
        "authority": "sam3d_reconstructed_shoulder_depth_plus_dwpose_observation_gate_shadow",
    }


def _canonical_foreground_element(element: dict[str, Any]) -> dict[str, Any] | None:
    region = str(element.get("body_region") or "")
    extension = str(element.get("extension") or "")
    frame_region = str(element.get("frame_region") or "")
    salience = str(element.get("salience") or "")
    if region not in {"arm", "forearm", "hand", "other"}:
        return None
    if frame_region not in {
        "lower_frame_left",
        "lower_frame_right",
        "frame_left",
        "frame_right",
        "lower_center",
        "center",
        "other",
    }:
        return None
    if salience not in {"large", "medium"}:
        return None

    region_phrase = {
        "arm": "arm",
        "forearm": "forearm",
        "hand": "hand",
        "other": "body element",
    }[region]
    extension_phrase = {
        "outstretched": "outstretched ",
        "extended": "extended ",
        "bent": "bent ",
        "unclear": "",
        "": "",
    }.get(extension, "")

    frame_phrase = {
        "lower_frame_left": "lower frame-left foreground",
        "lower_frame_right": "lower frame-right foreground",
        "frame_left": "frame-left foreground",
        "frame_right": "frame-right foreground",
        "lower_center": "lower-center foreground",
        "center": "central foreground",
        "other": "foreground",
    }[frame_region]

    verb = "fills much of" if salience == "large" else "occupies"
    text = f"an {extension_phrase}{region_phrase} {verb} the {frame_phrase}"
    if region_phrase in {"arm", "forearm"} and not extension_phrase:
        text = f"a {region_phrase} {verb} the {frame_phrase}"

    return {
        "body_region": region,
        "extension": extension or "unclear",
        "frame_region": frame_region,
        "salience": salience,
        "anatomical_side": None,
        "composer_text": text,
        "authority": "direct_qwen_frame_relative_visual_observation_shadow",
        "source_text": element.get("composer_text"),
    }


def _portrait_context(framing: dict[str, Any] | None, policy: dict[str, Any]) -> dict[str, Any]:
    span = (
        framing.get("anatomical_span")
        if isinstance(framing, dict) and isinstance(framing.get("anatomical_span"), dict)
        else {}
    )
    upper = span.get("upper_anchor")
    lower = span.get("lower_anchor")
    broad = bool(
        ((policy.get("visibility") or {}).get("broad_pose_supported"))
        if isinstance(policy.get("visibility"), dict)
        else False
    )
    eligible = bool(
        upper == "head"
        and lower in {"head", "shoulders", "hips"}
        and not broad
    )
    return {
        "eligible": eligible,
        "upper_anchor": upper,
        "lower_anchor": lower,
        "broad_pose_supported": broad,
        "reason": (
            "head_anchored_portrait_crop_without_broad_pose"
            if eligible
            else "outside_initial_portrait_viewpoint_validation_governor"
        ),
    }


def evaluate(
    policy: dict[str, Any],
    framing: dict[str, Any] | None,
    observer: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
) -> dict[str, Any]:
    geometry = build_subject_geometry(arrays, dwpose)
    viewpoint = _camera_viewpoint(geometry)
    shoulder = _shoulder_depth(arrays, dwpose)
    portrait = _portrait_context(framing, policy)

    observation = (
        observer.get("observation")
        if isinstance(observer, dict) and isinstance(observer.get("observation"), dict)
        else {}
    )
    capture = (
        observation.get("capture_style")
        if isinstance(observation.get("capture_style"), dict)
        else {}
    )
    capture_label = str(capture.get("label") or "unclear")

    foreground: list[dict[str, Any]] = []
    for raw in observation.get("foreground_body_elements") or []:
        if not isinstance(raw, dict):
            continue
        normalized = _canonical_foreground_element(raw)
        if normalized:
            foreground.append(normalized)

    foreground_present = bool(foreground)
    selfie_style = capture_label == "selfie_style"
    compositionally_selfie_like = selfie_style or foreground_present

    viewpoint_publishable = bool(viewpoint.get("publishable_candidate") and portrait["eligible"])
    shoulder_publishable = bool(
        shoulder.get("publishable_candidate")
        and portrait["eligible"]
        and compositionally_selfie_like
    )

    promoted_facts: list[str] = []
    if viewpoint_publishable and viewpoint.get("composer_text"):
        promoted_facts.append(str(viewpoint["composer_text"]))
    promoted_facts.extend(str(e["composer_text"]) for e in foreground)
    if shoulder_publishable and shoulder.get("composer_text"):
        promoted_facts.append(str(shoulder["composer_text"]))

    cue_count = sum(
        [
            1 if selfie_style else 0,
            1 if viewpoint_publishable else 0,
            1 if foreground_present else 0,
            1 if shoulder_publishable else 0,
        ]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": policy.get("image_key"),
        "portrait_context": portrait,
        "capture_style": {
            "label": capture_label,
            "basis": capture.get("basis") if isinstance(capture.get("basis"), list) else [],
            "authority": "qwen_visual_style_candidate_shadow",
            "caption_required": False,
        },
        "camera_viewpoint": {
            **viewpoint,
            "publishable_after_governor": viewpoint_publishable,
        },
        "foreground_body_composition": {
            "elements": foreground,
            "present": foreground_present,
            "frame_relative_language_required": True,
            "anatomical_side_assignment_forbidden": True,
        },
        "nearer_shoulder": {
            **shoulder,
            "publishable_after_governor": shoulder_publishable,
            "promotion_requires_selfie_style_or_salient_foreground_body": True,
        },
        "hypothesis": {
            "compositionally_selfie_like": compositionally_selfie_like,
            "cue_count": cue_count,
            "package_candidate": cue_count >= 2,
            "promoted_fact_candidates": promoted_facts,
            "note": (
                "Selfie style is not a prerequisite for camera viewpoint or foreground-body facts. "
                "Nearer-shoulder wording is promoted only when a selfie-like or salient foreground-body "
                "composition makes the depth cue materially useful."
            ),
        },
        "invariants": {
            "selfie_label_does_not_imply_camera_elevation": True,
            "foreground_arm_does_not_imply_camera_holding": True,
            "foreground_body_uses_frame_left_right_not_anatomical_side": True,
            "nearer_shoulder_uses_anatomical_side_only_after_dwpose_observation_gate": True,
            "nearer_shoulder_is_not_used_to_assign_foreground_arm_side": True,
            "camera_viewpoint_is_subject_relative_geometry_not_world_terrain_slope": True,
            "below_upward_viewpoint_remains_diagnostic_only_in_v01": True,
        },
        "diagnostic_geometry": {
            "body_camera_relation": geometry.get("body_camera_relation"),
            "camera_relative_subject": geometry.get("camera_relative_subject"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fuse SAM3D camera/shoulder geometry with direct foreground-body observation for selfie-composition validation."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--framing-dir", type=Path)
    p.add_argument("--observer-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    framing_dir = args.framing_dir.expanduser().resolve() if args.framing_dir else run_dir / DEFAULT_FRAMING_SUBDIR
    observer_dir = args.observer_dir.expanduser().resolve() if args.observer_dir else run_dir / DEFAULT_OBSERVER_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    if not run_dir.is_dir() or not policy_dir.is_dir() or not framing_dir.is_dir() or not observer_dir.is_dir():
        print(
            f"Required input missing: run={run_dir} policy={policy_dir} framing={framing_dir} observer={observer_dir}",
            file=sys.stderr,
        )
        return 2

    requested = set(args.only)
    policy_paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        policy_paths = [
            p for p in policy_paths
            if p.name.removesuffix(".perception_policy.json") in requested
        ]
    if not policy_paths:
        print("No matching production v0.2 perception-policy records found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for policy_path in policy_paths:
        key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output_dir / f"{key}.camera_composition_shadow.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
            records.append(record)
            continue

        policy = _read_json(policy_path)
        sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
        sam_path = Path(str(sources.get("sam3d_arrays") or "")).expanduser()
        dw_path = Path(str(sources.get("dwpose") or "")).expanduser()
        framing_path = framing_dir / f"{key}.framing.json"
        observer_path = observer_dir / f"{key}.selfie_composition.json"

        if not sam_path.is_file() or not dw_path.is_file() or not framing_path.is_file() or not observer_path.is_file():
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "error": "missing_required_shadow_input",
                "sources": {
                    "policy": str(policy_path),
                    "sam3d_arrays": str(sam_path),
                    "dwpose": str(dw_path),
                    "framing": str(framing_path),
                    "observer": str(observer_path),
                },
            }
            _write_json(out_path, record)
            records.append(record)
            continue

        record = evaluate(
            policy,
            _read_json(framing_path),
            _read_json(observer_path),
            _load_arrays(sam_path),
            _read_json(dw_path),
        )
        record["sources"] = {
            "policy": str(policy_path),
            "sam3d_arrays": str(sam_path),
            "dwpose": str(dw_path),
            "framing": str(framing_path),
            "observer": str(observer_path),
        }
        _write_json(out_path, record)
        records.append(record)

        print(
            f"{key}: capture={record['capture_style']['label']} "
            f"view={record['camera_viewpoint']['classification']} "
            f"foreground={len(record['foreground_body_composition']['elements'])} "
            f"shoulder={record['nearer_shoulder'].get('anatomical_side_nearer') or '-'} "
            f"package={record['hypothesis']['package_candidate']}"
        )

    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    capture_counts = Counter(
        str((r.get("capture_style") or {}).get("label") or "unknown")
        for r in records
        if r.get("status") == "ok"
    )
    viewpoint_counts = Counter(
        str((r.get("camera_viewpoint") or {}).get("classification") or "unknown")
        for r in records
        if r.get("status") == "ok"
    )
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "capture_style_counts": dict(sorted(capture_counts.items())),
        "camera_viewpoint_counts": dict(sorted(viewpoint_counts.items())),
        "foreground_body_present_count": sum(
            bool((r.get("foreground_body_composition") or {}).get("present"))
            for r in records
            if r.get("status") == "ok"
        ),
        "package_candidate_count": sum(
            bool((r.get("hypothesis") or {}).get("package_candidate"))
            for r in records
            if r.get("status") == "ok"
        ),
        "calibration": {
            "elevated_min_vertical_vs_eye": ELEVATED_MIN_VERTICAL_VS_EYE,
            "elevated_max_downward_pitch": ELEVATED_MAX_DOWNWARD_PITCH,
            "downward_min_vertical_vs_eye": DOWNWARD_MIN_VERTICAL_VS_EYE,
            "downward_max_pitch": DOWNWARD_MAX_PITCH,
            "shoulder_clear_delta_widths": SHOULDER_CLEAR_DELTA_WIDTHS,
            "shoulder_weak_delta_widths": SHOULDER_WEAK_DELTA_WIDTHS,
        },
        "records": records,
    }
    _write_json(output_dir / "camera_composition_shadow.index.json", index)
    print(f"Index: {output_dir / 'camera_composition_shadow.index.json'}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
