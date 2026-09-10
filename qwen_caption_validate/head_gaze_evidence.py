from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "head-gaze-evidence-0.1"


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _find(directory: Path | None, key: str, suffix: str) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    direct = directory / f"{key}{suffix}"
    if direct.is_file():
        return direct
    matches = sorted(directory.rglob(f"{key}{suffix}"))
    return matches[0] if matches else None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _horizontal_from_yaw(yaw_deg: float | None, dead_zone: float = 10.0) -> str | None:
    if yaw_deg is None:
        return None
    if yaw_deg > dead_zone:
        return "frame_left"
    if yaw_deg < -dead_zone:
        return "frame_right"
    return "center"


def _vertical_from_pitch(pitch_deg: float | None, dead_zone: float = 10.0) -> str | None:
    if pitch_deg is None:
        return None
    if pitch_deg > dead_zone:
        return "up"
    if pitch_deg < -dead_zone:
        return "down"
    return "center"


def _head_yaw_strength(yaw_deg: float | None) -> str | None:
    if yaw_deg is None:
        return None
    a = abs(yaw_deg)
    if a < 15.0:
        return "frontal"
    if a < 35.0:
        return "turned"
    if a < 60.0:
        return "strong_turn"
    return "profile"


def _camera_relationship(magnitude_deg: float | None) -> str | None:
    if magnitude_deg is None:
        return None
    if magnitude_deg <= 15.0:
        return "toward_camera"
    if magnitude_deg >= 25.0:
        return "off_camera"
    return "uncertain"


def _iris_regions(uniface: dict[str, Any]) -> list[dict[str, Any]]:
    value = (
        uniface.get("face_parsing", {})
        .get("eye_region_diagnostics", {})
        .get("iris_regions", [])
    )
    return value if isinstance(value, list) else []


def _region_occlusion(item: dict[str, Any]) -> float | None:
    if not item.get("available"):
        return None
    fractions = item.get("fractions") if isinstance(item.get("fractions"), dict) else {}
    hair = _safe_float(fractions.get("hair")) or 0.0
    background = _safe_float(fractions.get("background")) or 0.0
    return min(1.0, hair + background)


def _gaze_observability(uniface: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []

    if uniface.get("status") != "ok":
        return {
            "eligible": False,
            "authority": "unavailable",
            "eye_expectation": None,
            "reasons": ["uniface_face_unavailable"],
            "warnings": [],
        }

    face = uniface.get("face") if isinstance(uniface.get("face"), dict) else {}
    state = uniface.get("face_state") if isinstance(uniface.get("face_state"), dict) else {}
    quality = uniface.get("face_quality") if isinstance(uniface.get("face_quality"), dict) else {}
    head = uniface.get("head_pose") if isinstance(uniface.get("head_pose"), dict) else {}
    mesh = uniface.get("face_mesh") if isinstance(uniface.get("face_mesh"), dict) else {}

    face_score = _safe_float(face.get("score"))
    q = _safe_float(quality.get("score"))
    sunglasses = _safe_float(state.get("sunglasses"))
    eye_l = _safe_float(state.get("left_eye_open"))
    eye_r = _safe_float(state.get("right_eye_open"))
    yaw = _safe_float(head.get("yaw_deg_raw"))

    abs_yaw = abs(yaw) if yaw is not None else 0.0
    eye_expectation = "both_eyes_expected" if abs_yaw < 25.0 else "one_clean_eye_sufficient"

    if face_score is not None and face_score < 0.75:
        reasons.append("low_face_detection_confidence")
    if sunglasses is not None and sunglasses >= 0.80:
        reasons.append("sunglasses_obscure_eye_gaze")

    eye_values = [v for v in (eye_l, eye_r) if v is not None]
    stronger_eye = max(eye_values) if eye_values else None
    weaker_eye = min(eye_values) if eye_values else None
    if stronger_eye is None:
        warnings.append("eye_state_unavailable")
    elif stronger_eye < 0.35:
        reasons.append("no_reliably_observable_eye")
    elif abs_yaw < 25.0 and weaker_eye is not None and weaker_eye < 0.35:
        warnings.append("one_expected_frontal_eye_has_weak_state")

    occlusions = [v for item in _iris_regions(uniface) if (v := _region_occlusion(item)) is not None]
    clean_regions = [v for v in occlusions if v <= 0.55]
    if not occlusions:
        warnings.append("eye_region_parsing_unavailable")
    elif not clean_regions:
        reasons.append("all_predicted_eye_regions_occluded")
    elif abs_yaw < 25.0 and len(occlusions) >= 2 and max(occlusions) > 0.55:
        warnings.append("one_expected_frontal_eye_region_occluded")

    bbox = face.get("bbox_xyxy") if isinstance(face.get("bbox_xyxy"), list) else None
    inter_iris = _safe_float(mesh.get("inter_iris_distance_px"))
    iris_ratio = None
    if bbox and len(bbox) >= 4 and inter_iris is not None:
        face_width = max(1.0, float(bbox[2]) - float(bbox[0]))
        iris_ratio = inter_iris / face_width
        if abs_yaw < 25.0 and iris_ratio < 0.15:
            reasons.append("collapsed_inter_iris_geometry_on_frontal_head")
        elif 25.0 <= abs_yaw < 45.0 and iris_ratio < 0.10:
            warnings.append("small_inter_iris_geometry_for_moderate_turn")

    if q is not None and q < 0.30:
        warnings.append("low_general_face_quality")

    if reasons:
        authority = "unavailable"
        eligible = False
    elif warnings:
        authority = "reduced"
        eligible = True
    else:
        authority = "high"
        eligible = True

    return {
        "eligible": eligible,
        "authority": authority,
        "eye_expectation": eye_expectation,
        "head_abs_yaw_deg": abs_yaw,
        "face_score": face_score,
        "quality_score": q,
        "sunglasses_probability": sunglasses,
        "eye_open": {"left": eye_l, "right": eye_r, "stronger": stronger_eye, "weaker": weaker_eye},
        "eye_region_occlusion": occlusions,
        "inter_iris_to_face_width": iris_ratio,
        "reasons": reasons,
        "warnings": warnings,
    }


def _head_evidence(uniface: dict[str, Any], pyfeat: dict[str, Any]) -> dict[str, Any]:
    source = None
    pitch = yaw = roll = None
    if uniface.get("status") == "ok":
        hp = uniface.get("head_pose") if isinstance(uniface.get("head_pose"), dict) else {}
        pitch = _safe_float(hp.get("pitch_deg_raw"))
        yaw = _safe_float(hp.get("yaw_deg_raw"))
        roll = _safe_float(hp.get("roll_deg_raw"))
        if pitch is not None and yaw is not None:
            source = "uniface_resnet50"
    if source is None and pyfeat.get("status") == "ok":
        hp = pyfeat.get("head_pose") if isinstance(pyfeat.get("head_pose"), dict) else {}
        pitch = _safe_float(hp.get("pitch_deg"))
        yaw = _safe_float(hp.get("yaw_deg"))
        roll = _safe_float(hp.get("roll_deg"))
        if pitch is not None and yaw is not None:
            source = "pyfeat_v28_fallback"

    comparison: dict[str, Any] | None = None
    if uniface.get("status") == "ok" and pyfeat.get("status") == "ok":
        uh = uniface.get("head_pose") or {}
        ph = pyfeat.get("head_pose") or {}
        up = _safe_float(uh.get("pitch_deg_raw"))
        uy = _safe_float(uh.get("yaw_deg_raw"))
        ur = _safe_float(uh.get("roll_deg_raw"))
        pp = _safe_float(ph.get("pitch_deg"))
        py = _safe_float(ph.get("yaw_deg"))
        pr = _safe_float(ph.get("roll_deg"))
        if None not in (up, uy, ur, pp, py, pr):
            assert up is not None and uy is not None and ur is not None
            assert pp is not None and py is not None and pr is not None
            pitch_err = abs(up - pp)
            yaw_err = abs(uy - py)
            roll_err = abs((-ur) - pr)
            same_yaw_side = _horizontal_from_yaw(uy) == _horizontal_from_yaw(py)
            if pitch_err <= 12.0 and yaw_err <= 10.0 and roll_err <= 15.0:
                status = "agree"
            elif abs(uy) >= 55.0 and same_yaw_side:
                status = "profile_magnitude_divergence"
            else:
                status = "diverge"
            comparison = {
                "status": status,
                "pitch_abs_error_deg": pitch_err,
                "yaw_abs_error_deg": yaw_err,
                "roll_abs_error_deg_after_uniface_sign_flip": roll_err,
                "same_frame_yaw_direction": same_yaw_side,
            }

    return {
        "available": source is not None,
        "primary_source": source,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "roll_deg_uniface_raw_or_pyfeat": roll,
        "frame_horizontal": _horizontal_from_yaw(yaw, 12.0),
        "vertical": _vertical_from_pitch(pitch, 10.0),
        "yaw_strength": _head_yaw_strength(yaw),
        "pyfeat_comparison": comparison,
        "convention": "frame coordinates: positive yaw -> frame_left; negative yaw -> frame_right; negative pitch -> down",
    }


def _ptgaze_compare(l2_pitch: float, l2_yaw: float, ptgaze: dict[str, Any]) -> dict[str, Any] | None:
    if ptgaze.get("status") != "ok":
        return None
    gaze = ptgaze.get("gaze") if isinstance(ptgaze.get("gaze"), dict) else {}
    pp = _safe_float(gaze.get("pitch_deg"))
    py = _safe_float(gaze.get("yaw_deg"))
    if pp is None or py is None:
        return None

    l_h = _horizontal_from_yaw(l2_yaw)
    p_h = _horizontal_from_yaw(py)
    l_v = _vertical_from_pitch(l2_pitch)
    p_v = _vertical_from_pitch(pp)
    horizontal_opposite = {l_h, p_h} == {"frame_left", "frame_right"}
    vertical_opposite = {l_v, p_v} == {"up", "down"}
    if horizontal_opposite or vertical_opposite:
        status = "conflict"
    elif l_h == p_h and l_v == p_v:
        status = "agree"
    else:
        status = "partial_agreement"
    return {
        "status": status,
        "pitch_deg": pp,
        "yaw_deg": py,
        "horizontal": p_h,
        "vertical": p_v,
        "component_error_deg": math.hypot(l2_pitch - pp, l2_yaw - py),
    }


def _gaze_evidence(l2cs: dict[str, Any], observability: dict[str, Any], ptgaze: dict[str, Any]) -> dict[str, Any]:
    if not observability.get("eligible"):
        return {
            "available": False,
            "publishable": False,
            "primary_source": "l2cs",
            "reason": "eye_gaze_not_observable",
            "authority": "unavailable",
        }
    if l2cs.get("status") != "ok":
        return {
            "available": False,
            "publishable": False,
            "primary_source": "l2cs",
            "reason": "l2cs_gaze_unavailable",
            "authority": "unavailable",
        }

    gaze = l2cs.get("gaze") if isinstance(l2cs.get("gaze"), dict) else {}
    pitch = _safe_float(gaze.get("pitch_deg"))
    yaw = _safe_float(gaze.get("yaw_deg"))
    magnitude = _safe_float(gaze.get("forward_deviation_deg"))
    if pitch is None or yaw is None:
        return {
            "available": False,
            "publishable": False,
            "primary_source": "l2cs",
            "reason": "l2cs_gaze_incomplete",
            "authority": "unavailable",
        }

    tie = _ptgaze_compare(pitch, yaw, ptgaze)
    authority = observability.get("authority", "high")
    publishable = True
    if tie is not None and tie.get("status") == "conflict":
        authority = "conflict"
        publishable = False
    elif tie is not None and tie.get("status") == "agree" and authority == "high":
        authority = "corroborated"

    return {
        "available": True,
        "publishable": publishable,
        "primary_source": "l2cs",
        "authority": authority,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "horizontal": _horizontal_from_yaw(yaw),
        "vertical": _vertical_from_pitch(pitch),
        "camera_relationship": _camera_relationship(magnitude),
        "forward_deviation_deg": magnitude,
        "ptgaze_tie_break": tie,
        "convention": "frame coordinates: +yaw=frame_left, -yaw=frame_right, +pitch=up, -pitch=down",
    }


def _keys(directory: Path | None, suffix: str) -> set[str]:
    if directory is None or not directory.is_dir():
        return set()
    return {p.name[: -len(suffix)] for p in directory.rglob(f"*{suffix}") if p.is_file()}


def build_record(key: str, uniface: dict[str, Any], pyfeat: dict[str, Any], l2cs: dict[str, Any], ptgaze: dict[str, Any]) -> dict[str, Any]:
    observability = _gaze_observability(uniface)
    return {
        "schema_version": SCHEMA_VERSION,
        "image_key": key,
        "status": "ok",
        "head": _head_evidence(uniface, pyfeat),
        "gaze_observability": observability,
        "gaze": _gaze_evidence(l2cs, observability, ptgaze),
        "sources": {
            "uniface_schema": uniface.get("schema_version"),
            "pyfeat_schema": pyfeat.get("schema_version"),
            "l2cs_schema": l2cs.get("schema_version"),
            "ptgaze_schema": ptgaze.get("schema_version") if ptgaze else None,
        },
        "notes": [
            "Head direction and gaze direction use explicit frame_left/frame_right vocabulary.",
            "Strong head yaw permits one clean eye; far-eye weakness is not itself a gaze penalty.",
            "Sunglasses are a hard gaze-observability null. Ordinary eyeglasses are not.",
            "eDifFIQA quality is corroborating evidence and is not a standalone hard gate.",
            "py-feat gaze is intentionally not used as camera-relative gaze authority.",
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join specialist head/gaze outputs into deterministic caption evidence.")
    p.add_argument("--uniface-dir", type=Path, required=True)
    p.add_argument("--pyfeat-dir", type=Path, required=True)
    p.add_argument("--l2cs-dir", type=Path, required=True)
    p.add_argument("--ptgaze-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--only", nargs="+", default=[])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    uniface_dir = args.uniface_dir.expanduser().resolve()
    pyfeat_dir = args.pyfeat_dir.expanduser().resolve()
    l2cs_dir = args.l2cs_dir.expanduser().resolve()
    ptgaze_dir = args.ptgaze_dir.expanduser().resolve() if args.ptgaze_dir else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = _keys(uniface_dir, ".uniface.json") | _keys(pyfeat_dir, ".pyfeat.json") | _keys(l2cs_dir, ".l2cs.json")
    if args.only:
        wanted = {k.lower() for k in args.only}
        keys = {k for k in keys if k.lower() in wanted}
    if not keys:
        raise SystemExit("No specialist records found")

    records = []
    for key in sorted(keys):
        uniface = _read_json(_find(uniface_dir, key, ".uniface.json"))
        pyfeat = _read_json(_find(pyfeat_dir, key, ".pyfeat.json"))
        l2cs = _read_json(_find(l2cs_dir, key, ".l2cs.json"))
        ptgaze = _read_json(_find(ptgaze_dir, key, ".gaze.json")) if ptgaze_dir else {}
        record = build_record(key, uniface, pyfeat, l2cs, ptgaze)
        _write_json(output_dir / f"{key}.head_gaze.json", record)
        records.append(record)
        head = record["head"]
        gaze = record["gaze"]
        print(
            f"{key}: head={head.get('frame_horizontal')}/{head.get('vertical')} "
            f"({head.get('yaw_strength')}) gaze="
            + (
                f"{gaze.get('horizontal')}/{gaze.get('vertical')}/{gaze.get('camera_relationship')} authority={gaze.get('authority')}"
                if gaze.get("available")
                else f"UNAVAILABLE reason={gaze.get('reason')}"
            )
        )

    index = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "gaze_available_count": sum(1 for r in records if r["gaze"].get("available")),
        "gaze_publishable_count": sum(1 for r in records if r["gaze"].get("publishable")),
        "head_available_count": sum(1 for r in records if r["head"].get("available")),
        "records": records,
    }
    _write_json(output_dir / "head_gaze_evidence.index.json", index)
    print(f"Head/gaze evidence bundle: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
