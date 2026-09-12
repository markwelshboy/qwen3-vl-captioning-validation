from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.1"
DEFAULT_HEAD_GAZE_SUBDIR = Path("head-gaze-evidence-v03")
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2"
SCHEMA_VERSION = "caption-fact-sheet-0.2"

LATERALITY_RE = re.compile(r"\b(left|right)\b", re.I)
ANATOMICAL_RE = re.compile(
    r"\b(left|right)\s+(hand|arm|forearm|wrist|elbow|shoulder|hip|knee|leg|ankle|foot|eye|ear)\b",
    re.I,
)
HEAD_ORIENTATION_RE = re.compile(
    r"(?:\b(?:head|neck|face)\b.{0,40}\b(?:tilt(?:ed)?|turn(?:ed)?|angle(?:d)?|orientation|yaw|pitch)\b"
    r"|\b(?:tilt(?:ed)?|turn(?:ed)?|angle(?:d)?|orientation|yaw|pitch)\b.{0,40}\b(?:head|neck|face)\b)",
    re.I,
)
PUBLISHABLE_HEAD_AXIS_AUTHORITIES = {"corroborated", "corroborated_direction", "high"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.strip().split())
    return value or None


def _neutralize_laterality(text: str) -> tuple[str, bool]:
    """Remove anatomical left/right labels. Never swap or infer sides."""
    changed = bool(ANATOMICAL_RE.search(text))
    out = ANATOMICAL_RE.sub(lambda m: m.group(2), text)
    if re.search(r"\bshoulder line\b", out, re.I):
        newer = re.sub(r"\s+on the (?:left|right) side\b", "", out, flags=re.I)
        changed = changed or newer != out
        out = newer
    return " ".join(out.split()).strip(" ,;"), changed


def _normalize_configuration(items: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = copy.deepcopy(raw)
        text = _clean(item.get("text"))
        if not text:
            continue
        item["text"] = text
        if HEAD_ORIENTATION_RE.search(text):
            item.update(
                promotion_status="held_for_head_pose_normalization",
                composer_text=None,
                note="Head/neck orientation is owned by head-gaze-evidence-v03.",
            )
            warnings.append("configuration_head_pose_leakage_held")
        else:
            normalized, changed = _neutralize_laterality(text)
            if changed:
                item.update(
                    normalized_text=normalized,
                    composer_text=normalized or None,
                    promotion_status="accepted_unlateralized_candidate",
                    note="Qwen anatomical left/right was removed rather than trusted or swapped; DWPose owns anatomical joint laterality.",
                )
                warnings.append("configuration_qwen_laterality_neutralized")
            else:
                item.update(composer_text=text, promotion_status="accepted_route_scoped_candidate")
        out.append(item)
    return out, warnings


def _normalize_visual(visual: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(visual)
    warnings: list[str] = []
    for field, values in out.items():
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            text = _clean(item.get("text"))
            if not text:
                continue
            item["text"] = text
            if item.get("promotion_status") == "held_for_laterality_normalization" or LATERALITY_RE.search(text):
                normalized, changed = _neutralize_laterality(text)
                if changed:
                    item.update(
                        normalized_text=normalized,
                        composer_text=normalized or None,
                        promotion_status="accepted_unlateralized_candidate",
                        note="Qwen anatomical laterality was stripped; DWPose visibility alone cannot attach an object/accessory semantic to a side.",
                    )
                    warnings.append(f"visual_{field}_qwen_laterality_neutralized")
                else:
                    item.update(
                        composer_text=None,
                        promotion_status="held_for_laterality_review",
                        note="Unresolved left/right wording is held rather than treated as authoritative.",
                    )
                    warnings.append(f"visual_{field}_unresolved_left_right_held")
            else:
                item["composer_text"] = text
    return out, warnings


def _framing(policy: dict[str, Any]) -> dict[str, Any]:
    vis = policy.get("visibility") if isinstance(policy.get("visibility"), dict) else {}
    extent = vis.get("extent_hint") or vis.get("extent_bucket")
    return {
        "available": bool(extent or vis),
        "source": "caption_perception_policy",
        "authority": "deterministic_observation",
        "extent": extent,
        "body_visibility": {k: vis.get(k) for k in ("head", "shoulders", "torso", "hips", "knees", "feet") if k in vis},
        "observed_bbox": vis.get("observed_bbox"),
        "broad_pose_supported": bool(vis.get("broad_pose_supported")),
        "note": "Observed crop extent constrains semantic bandwidth; it does not infer hidden anatomy or camera angle.",
    }


def _laterality_from_points(points: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    sides: dict[str, Any] = {}
    observed: list[str] = []
    for side in ("left", "right"):
        visible = []
        for part in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "eye", "ear"):
            name = f"{side}_{part}"
            if points.get(name) is not None:
                visible.append(part)
                observed.append(name)
        sides[side] = {"visible_joints": visible, "visible_joint_count": len(visible)}
    return {
        "available": bool(observed),
        "source": source,
        "authority": "dwpose_anatomical_joint_labels" if observed else "unavailable",
        "sides": sides,
        "observed_named_joints": observed,
        "note": "DWPose establishes anatomical joint identity/visibility only; it does not attach Qwen object/action semantics to a side.",
    }


def _load_laterality(policy: dict[str, Any]) -> dict[str, Any]:
    sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
    source, size = sources.get("dwpose"), policy.get("image_size")
    if not source or not isinstance(size, list) or len(size) < 2:
        return _laterality_from_points({}, str(source) if source else None)
    path = Path(str(source)).expanduser()
    if not path.is_file():
        out = _laterality_from_points({}, str(path)); out["reason"] = "dwpose_source_not_found"; return out
    try:
        from .caption_perception_policy import _dwpose_points
        return _laterality_from_points(_dwpose_points(_read_json(path), int(size[0]), int(size[1])), str(path))
    except Exception as exc:
        out = _laterality_from_points({}, str(path)); out["reason"] = f"{type(exc).__name__}: {exc}"; return out


def _torso_from_result(result: dict[str, Any]) -> dict[str, Any]:
    model = result.get("model_facts") if isinstance(result.get("model_facts"), dict) else {}
    orientation, yaw = model.get("torso_camera_orientation"), model.get("torso_yaw_magnitude_deg")
    available = orientation is not None or yaw is not None
    return {
        "available": available,
        "source": result.get("source"),
        "dwpose_source": result.get("dwpose_source"),
        "authority": "sam3d_reconstruction_observation_gated" if available else "unavailable",
        "torso_camera_orientation": orientation,
        "torso_yaw_magnitude_deg": yaw,
        "reconstruction_is_observation": False,
        "can_promote_broad_pose": False,
        "reason": result.get("reason"),
        "note": "Only the Phase-2 torso whitelist is exposed; DWPose visibility gates publication and SAM3D cannot create broad-pose authority.",
    }


def _load_torso(policy: dict[str, Any]) -> dict[str, Any]:
    try:
        from .fragment_probe_routed_v01 import _load_sam3d_guidance
        return _torso_from_result(_load_sam3d_guidance(policy))
    except Exception as exc:
        return _torso_from_result({"reason": f"{type(exc).__name__}: {exc}"})


def _head_axis(axis: dict[str, Any], semantic: Any, degrees: Any) -> dict[str, Any]:
    authority = str(axis.get("authority") or "unavailable")
    publishable = authority in PUBLISHABLE_HEAD_AXIS_AUTHORITIES and semantic is not None
    return {
        "value": semantic if publishable else None,
        "candidate_value": semantic,
        "degrees": degrees,
        "authority": authority,
        "publishable": publishable,
        "abs_error_deg": axis.get("abs_error_deg"),
        "primary_class": axis.get("primary_class"),
        "secondary_class": axis.get("secondary_class"),
    }


def _head_gaze(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if record.get("status") != "ok":
        return (
            {"available": False, "authority": "unavailable", "reason": "head_gaze_evidence_not_ok"},
            {"available": False, "publishable": False, "authority": "unavailable", "reason": "head_gaze_evidence_not_ok"},
        )
    raw_head = record.get("head") if isinstance(record.get("head"), dict) else {}
    axes = raw_head.get("axis_authority") if isinstance(raw_head.get("axis_authority"), dict) else {}
    if raw_head.get("available"):
        horizontal = _head_axis(axes.get("yaw") or {}, raw_head.get("frame_horizontal"), raw_head.get("yaw_deg"))
        vertical = _head_axis(axes.get("pitch") or {}, raw_head.get("vertical"), raw_head.get("pitch_deg"))
        head = {
            "available": True,
            "source": raw_head.get("primary_source"),
            "authority": raw_head.get("authority"),
            "horizontal": horizontal,
            "vertical": vertical,
            "yaw_strength": raw_head.get("yaw_strength") if horizontal["publishable"] else None,
            "roll": {"publishable": False, "authority": (axes.get("roll") or {}).get("authority"), "degrees": raw_head.get("roll_deg_uniface_raw_or_pyfeat")},
            "convention": raw_head.get("convention"),
        }
    else:
        head = {"available": False, "source": raw_head.get("primary_source"), "authority": "unavailable", "reason": raw_head.get("reason") or "head_unavailable"}

    obs = record.get("gaze_observability") if isinstance(record.get("gaze_observability"), dict) else {}
    raw = record.get("gaze") if isinstance(record.get("gaze"), dict) else {}
    publishable = bool(obs.get("eligible") and raw.get("available") and raw.get("publishable"))
    obs_out = {"eligible": bool(obs.get("eligible")), "authority": obs.get("authority") or "unavailable", "reasons": obs.get("reasons") or [], "warnings": obs.get("warnings") or []}
    if publishable:
        gaze = {
            "available": True,
            "publishable": True,
            "source": raw.get("primary_source"),
            "authority": raw.get("authority"),
            "horizontal": raw.get("horizontal"),
            "vertical": raw.get("vertical"),
            "camera_relationship": raw.get("camera_relationship"),
            "yaw_deg": raw.get("yaw_deg"),
            "pitch_deg": raw.get("pitch_deg"),
            "forward_deviation_deg": raw.get("forward_deviation_deg"),
            "observability": obs_out,
            "convention": raw.get("convention"),
        }
    else:
        gaze = {
            "available": bool(raw.get("available")),
            "publishable": False,
            "source": raw.get("primary_source"),
            "authority": "unavailable",
            "reason": raw.get("reason") or ("eye_gaze_not_observable" if not obs.get("eligible") else "gaze_not_publishable"),
            "observability": obs_out,
        }
    return head, gaze


def _build_fact_sheet(
    phase4a: dict[str, Any],
    policy: dict[str, Any],
    head_gaze_record: dict[str, Any],
    *,
    phase4a_path: Path,
    policy_path: Path,
    head_gaze_path: Path | None,
    laterality: dict[str, Any] | None = None,
    torso_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(phase4a)
    key = str(out.get("image_key") or policy.get("image_key") or "")
    old_audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    violations = list(old_audit.get("violations") or [])
    warnings = list(old_audit.get("warnings") or [])
    if phase4a.get("schema_version") != "caption-fact-sheet-0.1": violations.append("unexpected_phase4a_schema")
    if policy.get("image_key") and str(policy.get("image_key")) != key: violations.append("policy_image_key_mismatch")
    if str((phase4a.get("policy") or {}).get("mode") or "") != str((policy.get("policy") or {}).get("mode") or ""):
        violations.append("policy_mode_mismatch")

    framing = _framing(policy)
    laterality = copy.deepcopy(laterality) if laterality is not None else _load_laterality(policy)
    torso = copy.deepcopy(torso_geometry) if torso_geometry is not None else _load_torso(policy)
    head, gaze = _head_gaze(head_gaze_record)

    facts = out.setdefault("facts", {})
    body = facts.setdefault("body", {})
    body["configuration"], w = _normalize_configuration(body.get("configuration") if isinstance(body.get("configuration"), list) else [])
    warnings.extend(w)
    body["anatomical_laterality"] = laterality
    body["torso_geometry"] = torso
    facts["visual"], w = _normalize_visual(facts.get("visual") if isinstance(facts.get("visual"), dict) else {})
    warnings.extend(w)
    facts["framing"] = framing
    facts["head_pose"] = head
    facts["gaze"] = gaze

    if not laterality.get("available"): warnings.append("dwpose_laterality_unavailable")
    if not torso.get("available"): warnings.append("sam3d_torso_geometry_unavailable_or_observation_gated")
    if not head.get("available"): warnings.append("head_pose_unavailable")
    if not gaze.get("publishable"): warnings.append(f"gaze_resolved_null:{gaze.get('reason') or 'unavailable'}")

    sources = out.setdefault("sources", {})
    sources.update(
        phase4a_fact_sheet=str(phase4a_path),
        perception_policy=str(policy_path),
        head_gaze_evidence=str(head_gaze_path) if head_gaze_path else None,
        dwpose=laterality.get("source"),
        sam3d_torso=torso.get("source"),
    )
    out["schema_version"] = SCHEMA_VERSION
    out["status"] = "ok" if not violations else "needs_review"
    out["reserved_domains"] = {
        "framing": {"owner": "caption_perception_policy", "status": "resolved" if framing.get("available") else "resolved_null"},
        "anatomical_laterality": {"owner": "dwpose", "status": "resolved" if laterality.get("available") else "resolved_null"},
        "torso_geometry": {"owner": "sam3d_plus_dwpose_observation_gate", "status": "resolved" if torso.get("available") else "resolved_null"},
        "head_pose": {"owner": "head_gaze_evidence_v03", "status": "resolved" if head.get("available") else "resolved_null"},
        "gaze": {"owner": "head_gaze_evidence_v03", "status": "resolved" if gaze.get("publishable") else "resolved_null"},
    }
    out["audit"] = {
        "violations": sorted(set(str(x) for x in violations if x)),
        "warnings": sorted(set(str(x) for x in warnings if x)),
        "caption_ready": False,
        "phase": "4B",
        "invariants": {
            "no_model_calls": True,
            "sam3d_reconstruction_is_observation": False,
            "sam3d_can_promote_broad_pose": False,
            "qwen_anatomical_left_right_is_never_swapped_or_trusted": True,
            "head_and_gaze_are_separate_authority_domains": True,
            "gaze_requires_observability_and_publishable_l2cs_evidence": True,
            "head_axis_authority_is_independent": True,
        },
    }
    return out


def _input_files(directory: Path, only: set[str]) -> list[Path]:
    paths = sorted(directory.glob("*.fact_sheet.json"))
    return [p for p in paths if not only or p.name.removesuffix(".fact_sheet.json") in only]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-4B deterministic specialist fact-sheet normalizer.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--head-gaze-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr); return 2
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    head_dir = args.head_gaze_dir.expanduser().resolve() if args.head_gaze_dir else run_dir / DEFAULT_HEAD_GAZE_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    if not input_dir.is_dir() or not policy_dir.is_dir():
        print("Need Phase-4A fact-sheet and perception-policy directories.", file=sys.stderr); return 2
    paths = _input_files(input_dir, set(args.only))
    if not paths:
        print(f"No matching Phase-4A fact sheets found in {input_dir}", file=sys.stderr); return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for input_path in paths:
        key = input_path.name.removesuffix(".fact_sheet.json")
        out_path = output_dir / f"{key}.fact_sheet.json"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path)); continue
        policy_path = policy_dir / f"{key}.perception_policy.json"
        head_path = head_dir / f"{key}.head_gaze.json"
        if not policy_path.is_file():
            record = {"schema_version": SCHEMA_VERSION, "status": "error", "image_key": key, "error": "missing_perception_policy"}
        else:
            head = _read_json(head_path) if head_path.is_file() else {"schema_version": "head-gaze-evidence-0.3", "image_key": key, "status": "missing"}
            record = _build_fact_sheet(
                _read_json(input_path), _read_json(policy_path), head,
                phase4a_path=input_path, policy_path=policy_path, head_gaze_path=head_path if head_path.is_file() else None,
            )
        _write_json(out_path, record)
        records.append(record)
        print(f"{key}: {record.get('status')}")
    records.sort(key=lambda r: str(r.get("image_key") or ""))
    counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "policy_dir": str(policy_dir),
        "head_gaze_dir": str(head_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "invariants": {
            "no_model_calls": True,
            "phase4a_is_input_not_recomputed": True,
            "sam3d_reconstruction_is_observation": False,
            "sam3d_can_promote_broad_pose": False,
            "qwen_anatomical_laterality_is_neutralized_not_swapped": True,
            "head_axis_authority_is_independent": True,
            "gaze_requires_observability": True,
            "caption_ready": False,
        },
        "records": records,
    }
    index_path = output_dir / "caption_fact_sheet.index.json"
    _write_json(index_path, index)
    print(f"Index: {index_path}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
