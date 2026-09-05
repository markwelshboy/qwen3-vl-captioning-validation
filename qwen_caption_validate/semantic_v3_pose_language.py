from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .semantic_v3_pose_adapter import adapt_pose_v016


SUMMARY_VERSION = "semantic-v3-pose-language-0.1"
RUN_VERSION = "semantic-v3-pose-language-0.1-run"

_POSTURE_PHRASES = {
    "seated": "seated",
    "standing": "standing",
    "reclining": "reclining",
    "lying": "lying down",
    "crouching": "crouching",
    "squatting": "squatting",
    "kneeling": "kneeling",
    "walking": "walking",
}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sentence(value: str) -> str:
    text = value.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _posture_phrase(value: str) -> str | None:
    return _POSTURE_PHRASES.get(value)


def _orientation_component(projected: dict[str, Any]) -> dict[str, Any] | None:
    """Translate camera-relative body yaw into natural language without front/back claims."""
    orientation = _dict(projected.get("body_orientation_diagnostic"))
    if not orientation.get("available"):
        return None

    shoulder_authority = _float(orientation.get("shoulder_authority_percent")) / 100.0
    hip_authority = _float(orientation.get("hip_authority_percent")) / 100.0
    if shoulder_authority < 0.45:
        return None

    if hip_authority >= 0.45:
        scope = "torso"
        yaw = _float(orientation.get("body_yaw_from_frontal_deg"))
        authority = min(shoulder_authority, hip_authority)
    else:
        # A cropped image can still have strong shoulder evidence even when the hips are
        # absent or weak. In that case describe only the upper body, not the whole body.
        scope = "upper body"
        yaw = _float(orientation.get("shoulder_yaw_from_frontal_deg"))
        authority = shoulder_authority

    if yaw >= 70.0:
        label = "near_profile"
        phrase = f"{scope} nearly side-on to the camera"
    elif yaw >= 50.0:
        label = "strong_turn"
        phrase = f"{scope} strongly turned sideways to the camera"
    elif yaw >= 30.0:
        label = "three_quarter"
        phrase = f"{scope} partly turned sideways to the camera"
    else:
        # Small turns are usually better left to the rich VLM caption. Pose should add
        # information where geometry is meaningfully non-neutral, not narrate every angle.
        return None

    return {
        "label": label,
        "scope": scope,
        "phrase": phrase,
        "authority": round(authority, 4),
        "camera_relative_only": True,
    }


def _lean_components(projected: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Separate observed image-plane lean from depth-direction reconstruction hints."""
    modifiers = _dict(projected.get("posture_modifier_diagnostic"))
    severity = str(modifiers.get("lean_severity") or "")
    direction = str(modifiers.get("lean_direction") or "")
    authority = _float(modifiers.get("torso_inclination_authority"))

    if severity in ("", "upright"):
        return None, None

    caption_ready: dict[str, Any] | None = None
    if authority >= 0.45:
        phrase = {
            "slight": "leaning slightly",
            "moderate": "leaning noticeably",
            "heavy": "leaning heavily",
            "near_horizontal": "leaning heavily",
        }.get(severity)
        if phrase:
            caption_ready = {
                "severity": severity,
                "phrase": phrase,
                "authority": round(authority, 4),
                "direction_resolved": False,
            }

    conditional: dict[str, Any] | None = None
    if direction in ("forward_possible", "back"):
        direction_word = "forward" if direction == "forward_possible" else "back"
        adverb = {
            "slight": "slightly ",
            "moderate": "",
            "heavy": "heavily ",
            "near_horizontal": "far ",
        }.get(severity, "")
        conditional = {
            "kind": "lean_direction",
            "value": direction_word,
            "phrase": f"leaning {adverb}{direction_word}".strip(),
            "requires": "semantic_corroboration",
            "source": "pose_reconstruction_direction_hint",
        }

    return caption_ready, conditional


def _framing_component(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Emit framing only from explicit crop evidence, never from missing detections alone."""
    evidence = _dict(profile.get("evidence_support"))
    accepted = {str(value) for value in _list(evidence.get("dwpose_accepted_joint_names"))}
    in_frame = {str(value) for value in _list(evidence.get("dwpose_in_frame_accepted_joint_names"))}
    residual = _dict(evidence.get("projected_fit_residual"))
    extrapolated = {str(value) for value in _list(residual.get("extrapolated_dwpose_joint_names"))}

    hips = {"left_hip", "right_hip"}
    shoulders = {"left_shoulder", "right_shoulder"}
    if (
        shoulders & in_frame
        and hips.issubset(accepted)
        and not (hips & in_frame)
        and hips.issubset(extrapolated)
    ):
        return {
            "label": "tight_upper_body",
            "phrase": "tight crop around the head and upper torso",
            "basis": "both hips extrapolated outside the source frame",
        }
    return None


def _relation_components(adapted: dict[str, Any]) -> list[dict[str, Any]]:
    relations = _dict(adapted.get("relations"))
    result: list[dict[str, Any]] = []

    fist = _dict(relations.get("head_supported_by_fist"))
    hand = _dict(relations.get("head_supported_by_hand"))
    if fist.get("value") is True:
        side = fist.get("side")
        phrase = f"head resting on the {side} fist" if side in ("left", "right") else "head resting on a fist"
        result.append({
            "relation": "head_supported_by_fist",
            "phrase": phrase,
            "side": side,
            "crop_support": fist.get("crop_support"),
        })
    elif hand.get("value") is True:
        side = hand.get("side")
        phrase = f"head supported by the {side} hand" if side in ("left", "right") else "head supported by a hand"
        result.append({
            "relation": "head_supported_by_hand",
            "phrase": phrase,
            "side": side,
            "crop_support": hand.get("crop_support"),
        })

    hands_on_hips = _dict(relations.get("hands_on_hips"))
    if hands_on_hips.get("value") is True:
        result.append({
            "relation": "hands_on_hips",
            "phrase": "hands resting on the hips",
            "crop_support": hands_on_hips.get("crop_support"),
        })
    return result


def _injection_priority(components: dict[str, Any], conditional_hints: list[dict[str, Any]]) -> str:
    posture = _dict(components.get("posture")).get("value")
    if components.get("relations") or components.get("framing"):
        return "high"
    if posture in {"crouching", "squatting", "kneeling", "reclining", "lying"}:
        return "high"
    if components.get("orientation") or components.get("lean"):
        return "medium"
    if posture in {"seated", "walking"}:
        return "medium"
    if conditional_hints:
        return "conditional"
    return "low"


def summarize_pose_language(record: dict[str, Any]) -> dict[str, Any]:
    """Create caption-ready natural pose language from frozen Pose v0.16.

    This is deliberately a lossy translation layer. The detailed geometry remains in the
    source pose artifact; caption composition receives only human-scale posture, meaningful
    camera-relative turn, strongly supported relations, and explicit crop information.
    """
    profile = _dict(record.get("profile"))
    projected = _dict(profile.get("sam3d_projected_pose"))
    adapted = adapt_pose_v016(record)
    public = _dict(adapted.get("public"))
    recovery = _dict(adapted.get("semantic_recovery"))

    components: dict[str, Any] = {}
    conditional_hints: list[dict[str, Any]] = []

    posture_value = str(public.get("value") or "unknown")
    posture_text = _posture_phrase(posture_value)
    if posture_text:
        components["posture"] = {
            "value": posture_value,
            "phrase": posture_text,
            "authority": "pose_v0.16_public",
            "crop_support": public.get("crop_support"),
        }
    elif recovery.get("needed") is True:
        candidate = str(recovery.get("candidate") or "unknown")
        candidate_text = _posture_phrase(candidate)
        if candidate_text:
            conditional_hints.append({
                "kind": "posture",
                "value": candidate,
                "phrase": candidate_text,
                "requires": "semantic_corroboration",
                "candidate_score": recovery.get("candidate_score"),
                "winner_margin": recovery.get("winner_margin"),
            })

    orientation = _orientation_component(projected)
    if orientation:
        components["orientation"] = orientation

    lean, lean_direction_hint = _lean_components(projected)
    if lean:
        components["lean"] = lean
    if lean_direction_hint:
        conditional_hints.append(lean_direction_hint)

    framing = _framing_component(profile)
    if framing:
        components["framing"] = framing

    relations = _relation_components(adapted)
    if relations:
        components["relations"] = relations

    caption_ready: list[str] = []
    geometry_clauses: list[str] = []
    if posture_text:
        geometry_clauses.append(posture_text)
    if orientation:
        geometry_clauses.append(("with the " if geometry_clauses else "") + orientation["phrase"])
    if lean:
        geometry_clauses.append(("while " if geometry_clauses else "") + lean["phrase"])
    if geometry_clauses:
        caption_ready.append(_sentence(" ".join(geometry_clauses)))

    for relation in relations:
        caption_ready.append(_sentence(str(relation["phrase"])))
    if framing:
        caption_ready.append(_sentence(str(framing["phrase"])))

    priority = _injection_priority(components, conditional_hints)
    return {
        "schema_version": SUMMARY_VERSION,
        "source_schema_version": profile.get("schema_version"),
        "injection_priority": priority,
        "caption_ready_phrases": caption_ready,
        "conditional_hints": conditional_hints,
        "components": components,
        "editor_contract": {
            "use": "Use these phrases to simplify or correct VLM body geometry; preserve richer VLM appearance and scene detail.",
            "conditional": "Conditional hints may be promoted only when independently corroborated by the semantic VLM caption.",
            "do_not_expand": "Do not turn this summary back into diagnostic angles or infer front/back, contact, support, or hidden anatomy beyond the governed phrases.",
        },
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _key(path: Path) -> str:
    suffix = ".sam3d_relational_pose.json"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate frozen Pose v0.16 records into concise natural caption language.")
    parser.add_argument("pose_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tar", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pose_dir = args.pose_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not pose_dir.is_dir():
        print(f"Pose profile directory not found: {pose_dir}", file=sys.stderr)
        return 2

    selected = set(args.only)
    paths = sorted(pose_dir.glob("*.sam3d_relational_pose.json"))
    if selected:
        paths = [path for path in paths if _key(path) in selected]
    if not paths:
        print("No matching Pose v0.16 records found.", file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in paths:
        image_key = _key(path)
        out_path = output / f"{image_key}.pose_language.json"
        if out_path.exists() and not args.overwrite:
            result = _load(out_path)
            reused = True
        else:
            result = summarize_pose_language(_load(path))
            _write(out_path, result)
            reused = False
        records.append({
            "image_key": image_key,
            "path": str(out_path),
            "reused": reused,
            "injection_priority": result.get("injection_priority"),
            "caption_ready_phrases": result.get("caption_ready_phrases") or [],
            "conditional_hints": result.get("conditional_hints") or [],
        })
        print(f"{image_key}: priority={result.get('injection_priority')} phrases={len(result.get('caption_ready_phrases') or [])} conditional={len(result.get('conditional_hints') or [])}")

    index = {
        "schema_version": RUN_VERSION,
        "source_dir": str(pose_dir),
        "output_dir": str(output),
        "record_count": len(records),
        "records": records,
    }
    _write(output / "pose_language.index.json", index)

    if args.tar:
        import tarfile
        tar_path = output.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as archive:
            archive.add(output, arcname=output.name)
        print(f"Tar: {tar_path}")
    print(f"Pose language: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
