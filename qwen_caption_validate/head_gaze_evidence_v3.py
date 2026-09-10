from __future__ import annotations

from typing import Any

from . import head_gaze_evidence as base
from . import head_gaze_evidence_v2 as v2

SCHEMA_VERSION = "head-gaze-evidence-0.3"
_BASE_BUILD_RECORD = v2.build_record


def _axis_compare(primary: float | None, secondary: float | None, *, axis: str) -> dict[str, Any] | None:
    if primary is None or secondary is None:
        return None

    error = abs(primary - secondary)
    if axis == "yaw":
        p_class = base._horizontal_from_yaw(primary, 12.0)
        s_class = base._horizontal_from_yaw(secondary, 12.0)
        opposite = {p_class, s_class} == {"frame_left", "frame_right"}
        if opposite:
            authority = "conflict"
        elif p_class == s_class:
            authority = "corroborated" if error <= 10.0 else "corroborated_direction"
        else:
            # One estimator is inside the center dead-zone and the other just outside.
            authority = "reduced"
    elif axis == "pitch":
        p_class = base._vertical_from_pitch(primary, 10.0)
        s_class = base._vertical_from_pitch(secondary, 10.0)
        opposite = {p_class, s_class} == {"up", "down"}
        if opposite:
            authority = "conflict"
        elif p_class == s_class:
            authority = "corroborated" if error <= 12.0 else "corroborated_direction"
        else:
            authority = "reduced"
    else:
        raise ValueError(f"unsupported axis: {axis}")

    return {
        "authority": authority,
        "primary_deg": primary,
        "secondary_deg": secondary,
        "abs_error_deg": error,
        "primary_class": p_class,
        "secondary_class": s_class,
    }


def _roll_compare(uniface_roll: float | None, pyfeat_roll: float | None, head_abs_yaw: float | None) -> dict[str, Any] | None:
    if uniface_roll is None or pyfeat_roll is None:
        return None
    # Empirical convention established on the validation set: py-feat roll ~= -UniFace raw roll.
    normalized_uniface = -uniface_roll
    error = abs(normalized_uniface - pyfeat_roll)
    if head_abs_yaw is not None and head_abs_yaw >= 55.0:
        authority = "reduced_profile"
    elif error <= 15.0:
        authority = "corroborated"
    else:
        authority = "reduced"
    return {
        "authority": authority,
        "uniface_raw_deg": uniface_roll,
        "uniface_pyfeat_frame_deg": normalized_uniface,
        "pyfeat_deg": pyfeat_roll,
        "abs_error_deg": error,
        "note": "Roll is diagnostic and does not make yaw/pitch evidence conflict by itself.",
    }


def _head_axis_authority(record: dict[str, Any], uniface: dict[str, Any], pyfeat: dict[str, Any]) -> dict[str, Any]:
    head = record.get("head") if isinstance(record.get("head"), dict) else {}
    result: dict[str, Any] = {"yaw": None, "pitch": None, "roll": None}
    if not head.get("available"):
        return result

    if uniface.get("status") != "ok" or pyfeat.get("status") != "ok":
        return result

    uh = uniface.get("head_pose") if isinstance(uniface.get("head_pose"), dict) else {}
    ph = pyfeat.get("head_pose") if isinstance(pyfeat.get("head_pose"), dict) else {}
    up = base._safe_float(uh.get("pitch_deg_raw"))
    uy = base._safe_float(uh.get("yaw_deg_raw"))
    ur = base._safe_float(uh.get("roll_deg_raw"))
    pp = base._safe_float(ph.get("pitch_deg"))
    py = base._safe_float(ph.get("yaw_deg"))
    pr = base._safe_float(ph.get("roll_deg"))

    result["yaw"] = _axis_compare(uy, py, axis="yaw")
    result["pitch"] = _axis_compare(up, pp, axis="pitch")
    result["roll"] = _roll_compare(ur, pr, abs(uy) if uy is not None else None)
    return result


def _aggregate_head_authority(record: dict[str, Any], uniface: dict[str, Any], axes: dict[str, Any]) -> str:
    head = record.get("head") if isinstance(record.get("head"), dict) else {}
    if not head.get("available"):
        return "unavailable"
    if head.get("primary_source") == "pyfeat_v28_fallback":
        return "reduced"

    quality = None
    if uniface.get("status") == "ok":
        quality = base._safe_float((uniface.get("face_quality") or {}).get("score"))

    yaw = axes.get("yaw") if isinstance(axes.get("yaw"), dict) else None
    pitch = axes.get("pitch") if isinstance(axes.get("pitch"), dict) else None
    axis_states = [x.get("authority") for x in (yaw, pitch) if x]

    if "conflict" in axis_states:
        authority = "conflict"
    elif "reduced" in axis_states:
        authority = "reduced"
    elif axis_states and all(v == "corroborated" for v in axis_states):
        authority = "corroborated"
    elif axis_states and all(v in {"corroborated", "corroborated_direction"} for v in axis_states):
        authority = "corroborated_direction"
    elif axis_states:
        authority = "reduced"
    else:
        authority = "high"

    if quality is not None and quality < 0.30 and authority not in {"unavailable", "conflict"}:
        authority = "reduced"
    return authority


def build_record(key: str, uniface: dict[str, Any], pyfeat: dict[str, Any], l2cs: dict[str, Any], ptgaze: dict[str, Any]) -> dict[str, Any]:
    record = _BASE_BUILD_RECORD(key, uniface, pyfeat, l2cs, ptgaze)
    record["schema_version"] = SCHEMA_VERSION
    axes = _head_axis_authority(record, uniface, pyfeat)
    record["head"]["axis_authority"] = axes
    record["head"]["authority"] = _aggregate_head_authority(record, uniface, axes)
    record["notes"].append(
        "Head authority is axis-aware: roll disagreement cannot poison yaw/pitch; only true left/right or up/down contradiction is a head conflict."
    )
    return record


def main() -> int:
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.build_record = build_record
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
