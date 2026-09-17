from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "framing-semantics-1.0"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_PERSON_SUBDIR = Path("semantic-v3") / "person-detection-evidence-v0.1"
DEFAULT_UNIFACE_SUBDIRS = (
    Path("face-authority-uniface-v02-all"),
    Path("face-authority-uniface-all"),
)
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "framing-semantics-v1.0"

TIER_ORDER = ("head", "shoulders", "hips", "knees", "ankles")

# Final validation calibration after full 87-image census and visual boundary
# review. These are deliberately separated by abstention bands.
HEAD_SHOULDERS_MEDIUM_MAX = 0.36
HEAD_SHOULDERS_MEDIUM_CLOSE_MIN = 0.38
HEAD_SHOULDERS_MEDIUM_CLOSE_MAX = 0.66
HEAD_SHOULDERS_CLOSE_MIN = 0.68
HEAD_ONLY_CLOSE_MIN = 0.55


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _tier_states(policy: dict[str, Any]) -> dict[str, str]:
    vis = policy.get("visibility") if isinstance(policy.get("visibility"), dict) else {}
    tiers = vis.get("anatomical_tiers") if isinstance(vis.get("anatomical_tiers"), dict) else {}

    states: dict[str, str] = {}
    for tier in TIER_ORDER:
        record = tiers.get(tier) if isinstance(tiers.get(tier), dict) else {}
        state = str(record.get("state") or "absent")
        states[tier] = state if state in {"strong", "partial", "absent"} else "absent"
    return states


def _core_span(policy: dict[str, Any]) -> dict[str, Any]:
    states = _tier_states(policy)
    strong_indices = [i for i, tier in enumerate(TIER_ORDER) if states[tier] == "strong"]

    if not strong_indices:
        partial = [tier for tier in TIER_ORDER if states[tier] == "partial"]
        return {
            "status": "unavailable",
            "upper_anchor": None,
            "lower_anchor": None,
            "upper_partial": None,
            "lower_partial": None,
            "noncontiguous_observations": partial,
            "composer_text": None,
            "reason": "no_strong_anatomical_anchor",
        }

    start = strong_indices[0]
    end = start
    for idx in range(start + 1, len(TIER_ORDER)):
        if states[TIER_ORDER[idx]] != "strong":
            break
        end = idx

    upper = TIER_ORDER[start]
    lower = TIER_ORDER[end]
    upper_partial = TIER_ORDER[start - 1] if start > 0 and states[TIER_ORDER[start - 1]] == "partial" else None
    lower_partial = TIER_ORDER[end + 1] if end + 1 < len(TIER_ORDER) and states[TIER_ORDER[end + 1]] == "partial" else None

    noncontiguous: list[str] = []
    for idx in range(end + 1, len(TIER_ORDER)):
        tier = TIER_ORDER[idx]
        if tier == lower_partial:
            continue
        if states[tier] in {"strong", "partial"}:
            noncontiguous.append(tier)

    if upper == lower:
        text = "framed tightly around the head and face" if upper == "head" else f"framed around the {upper}"
    else:
        prefix = "framed from the" if upper == "head" else "framed from around the"
        text = f"{prefix} {upper} through the {lower}"

    return {
        "status": "available",
        "upper_anchor": upper,
        "lower_anchor": lower,
        "upper_partial": upper_partial,
        "lower_partial": lower_partial,
        "noncontiguous_observations": noncontiguous,
        "composer_text": text,
        "reason": "deepest_contiguous_strong_anatomical_span",
        "tier_states": states,
    }


def _uniface_dir(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_dir() else None
    for rel in DEFAULT_UNIFACE_SUBDIRS:
        p = run_dir / rel
        if p.is_dir():
            return p
    return None


def _face_geometry(uniface: dict[str, Any] | None, image_size: tuple[int, int] | None) -> dict[str, Any]:
    if not isinstance(uniface, dict) or not uniface:
        return {
            "status": "unavailable",
            "authority": "observed_face_detector",
            "source": "uniface_retinaface_target_bound",
            "reason": "uniface_record_missing",
        }
    source_status = str(uniface.get("status") or "missing")
    if source_status != "ok":
        return {
            "status": "unavailable",
            "authority": "observed_face_detector",
            "source": "uniface_retinaface_target_bound",
            "source_status": source_status,
            "reason": "uniface_face_unavailable",
        }

    face = uniface.get("face") if isinstance(uniface.get("face"), dict) else {}
    box = face.get("bbox_xyxy")
    if not isinstance(box, list) or len(box) < 4:
        return {
            "status": "unavailable",
            "authority": "observed_face_detector",
            "source": "uniface_retinaface_target_bound",
            "source_status": source_status,
            "reason": "uniface_bbox_missing",
        }
    if image_size is None:
        return {
            "status": "unavailable",
            "authority": "observed_face_detector",
            "source": "uniface_retinaface_target_bound",
            "source_status": source_status,
            "reason": "source_image_dimensions_unavailable",
        }

    width, height = image_size
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    cx1 = max(0.0, min(float(width), x1))
    cy1 = max(0.0, min(float(height), y1))
    cx2 = max(0.0, min(float(width), x2))
    cy2 = max(0.0, min(float(height), y2))
    fw = max(0.0, cx2 - cx1)
    fh = max(0.0, cy2 - cy1)

    if fw <= 0 or fh <= 0:
        return {
            "status": "unavailable",
            "authority": "observed_face_detector",
            "source": "uniface_retinaface_target_bound",
            "reason": "uniface_bbox_has_no_visible_intersection_with_image",
        }

    return {
        "status": "available",
        "authority": "observed_face_detector",
        "source": "uniface_retinaface_target_bound",
        "source_status": source_status,
        "detector_score": float(face.get("score")) if isinstance(face.get("score"), (int, float)) else None,
        "selection_strategy": face.get("selection_strategy"),
        "bbox_xyxy": [x1, y1, x2, y2],
        "visible_bbox_xyxy": [cx1, cy1, cx2, cy2],
        "image_width": width,
        "image_height": height,
        "width_fraction": round(fw / width, 6),
        "height_fraction": round(fh / height, 6),
        "bbox_area_fraction": round((fw * fh) / (width * height), 6),
        "bbox_clipped_to_image": any(
            abs(a - b) > 1e-6
            for a, b in zip((x1, y1, x2, y2), (cx1, cy1, cx2, cy2))
        ),
    }


def _person_geometry(person_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(person_record, dict):
        return {
            "status": "unavailable",
            "authority": "observed_person_detector",
            "source": "easy_dwpose_yolox_target_bound",
            "reason": "person_detection_record_missing",
        }
    geom = person_record.get("person_geometry") if isinstance(person_record.get("person_geometry"), dict) else {}
    return dict(geom) if geom else {
        "status": "unavailable",
        "authority": "observed_person_detector",
        "source": "easy_dwpose_yolox_target_bound",
        "reason": "person_geometry_missing",
    }


def _ratio(face: dict[str, Any], person: dict[str, Any]) -> float | None:
    if face.get("status") != "available" or person.get("status") != "available":
        return None
    fh = face.get("height_fraction")
    ph = person.get("visible_height_fraction")
    if not isinstance(fh, (int, float)) or not isinstance(ph, (int, float)) or ph <= 0:
        return None
    return float(fh) / float(ph)


def _scale_candidate(label: str, span: dict[str, Any], *, basis: list[str]) -> dict[str, Any]:
    surface = {
        "medium": "medium shot",
        "medium_close_up": "medium close-up",
        "close_up": "close-up",
        "medium_wide": "medium-wide shot",
    }[label]
    return {
        "status": "candidate",
        "label": label,
        "composer_text": surface,
        "basis": basis,
        "canonical_span": {
            "upper": span.get("upper_anchor"),
            "lower": span.get("lower_anchor"),
        },
        "note": (
            "Optional photographic language. The authoritative crop truth remains the "
            "pose-neutral anatomical span."
        ),
    }


def _withheld(span: dict[str, Any], reason: str, ratio: float | None = None) -> dict[str, Any]:
    basis = [reason, f"canonical_span={span.get('upper_anchor')}->{span.get('lower_anchor')}"]
    if ratio is not None:
        basis.append(f"face_person_height_ratio={ratio:.3f}")
    return {
        "status": "withheld",
        "label": None,
        "composer_text": None,
        "basis": basis,
        "note": "Conventional photographic scale withheld; use the canonical anatomical span.",
    }


def _standard_scale(span: dict[str, Any], face: dict[str, Any], person: dict[str, Any]) -> dict[str, Any]:
    key = (str(span.get("upper_anchor") or ""), str(span.get("lower_anchor") or ""))
    r = _ratio(face, person)

    if key == ("head", "shoulders"):
        if r is None:
            return _withheld(span, "target_bound_face_or_person_geometry_unavailable")
        ratio_basis = [
            "canonical_anatomical_span",
            "target_bound_uniface_retinaface_height_fraction",
            "target_bound_easy_dwpose_yolox_person_height_fraction",
            f"face_person_height_ratio={r:.3f}",
        ]
        if r <= HEAD_SHOULDERS_MEDIUM_MAX:
            return _scale_candidate("medium", span, basis=ratio_basis + ["calibration_band=head_shoulders_medium"])
        if r < HEAD_SHOULDERS_MEDIUM_CLOSE_MIN:
            return _withheld(span, "head_shoulders_medium_to_mcu_abstention_band", r)
        if r <= HEAD_SHOULDERS_MEDIUM_CLOSE_MAX:
            return _scale_candidate("medium_close_up", span, basis=ratio_basis + ["calibration_band=head_shoulders_medium_close"])
        if r < HEAD_SHOULDERS_CLOSE_MIN:
            return _withheld(span, "head_shoulders_mcu_to_close_abstention_band", r)
        return _scale_candidate("close_up", span, basis=ratio_basis + ["calibration_band=head_shoulders_close"])

    if key == ("head", "head"):
        if r is None:
            return _withheld(span, "target_bound_face_or_person_geometry_unavailable")
        if r >= HEAD_ONLY_CLOSE_MIN:
            return _scale_candidate(
                "close_up",
                span,
                basis=[
                    "canonical_anatomical_span",
                    "target_bound_face_person_relative_geometry",
                    f"face_person_height_ratio={r:.3f}",
                    "extreme_close_up_disabled_without_positive_calibration",
                ],
            )
        return _withheld(span, "head_only_face_person_ratio_below_close_calibration", r)

    if key == ("head", "hips"):
        return _scale_candidate("medium", span, basis=["canonical_head_through_hips_span"])

    if key == ("head", "knees"):
        return _scale_candidate("medium_wide", span, basis=["canonical_head_through_knees_span"])

    if key == ("head", "ankles"):
        return _withheld(span, "dwpose_observes_ankles_not_feet_full_body_language_not_authorized")

    return _withheld(span, "no_validated_standard_scale_rule_for_anatomical_span")


def _surface(span: dict[str, Any], scale: dict[str, Any]) -> dict[str, Any]:
    scale_text = _clean(scale.get("composer_text"))
    span_text = _clean(span.get("composer_text"))

    if scale.get("status") == "candidate" and scale_text:
        return {
            "source": "standard_shot_scale",
            "composer_text": scale_text,
            "opening_template": f"[[trigger]] is shown in a {scale_text}",
            "anatomical_span_retained_internally": True,
            "reason": "validated_standard_scale_available_do_not_mechanically_repeat_anatomical_span",
        }

    if span_text:
        opening = "[[trigger]] is " + span_text if span_text.startswith("framed ") else "[[trigger]] is shown " + span_text
        return {
            "source": "anatomical_span",
            "composer_text": span_text,
            "opening_template": opening,
            "anatomical_span_retained_internally": True,
            "reason": "standard_scale_withheld_use_literal_anatomical_crop",
        }

    return {
        "source": None,
        "composer_text": None,
        "opening_template": None,
        "anatomical_span_retained_internally": False,
        "reason": "framing_unavailable",
    }


def evaluate(
    policy: dict[str, Any],
    uniface: dict[str, Any] | None,
    person_record: dict[str, Any] | None,
) -> dict[str, Any]:
    size = policy.get("image_size")
    image_size = (
        (int(size[0]), int(size[1]))
        if isinstance(size, list) and len(size) >= 2 and all(isinstance(v, (int, float)) for v in size[:2])
        else None
    )
    span = _core_span(policy)
    face = _face_geometry(uniface, image_size)
    person = _person_geometry(person_record)
    scale = _standard_scale(span, face, person)
    ratio = _ratio(face, person)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if span.get("status") == "available" else "resolved_null",
        "image_key": policy.get("image_key"),
        "authority": "deterministic_observation",
        "anatomical_span": span,
        "face_scale_geometry": face,
        "person_scale_geometry": person,
        "face_person_height_ratio": round(ratio, 6) if ratio is not None else None,
        "standard_shot_scale": scale,
        "composer_framing": _surface(span, scale),
        "legacy_extent_hint": (
            (policy.get("visibility") or {}).get("extent_hint")
            if isinstance(policy.get("visibility"), dict)
            else None
        ),
        "broad_pose_supported": bool(
            (policy.get("visibility") or {}).get("broad_pose_supported")
            if isinstance(policy.get("visibility"), dict)
            else False
        ),
        "calibration": {
            "head_shoulders_medium_max": HEAD_SHOULDERS_MEDIUM_MAX,
            "head_shoulders_medium_close_min": HEAD_SHOULDERS_MEDIUM_CLOSE_MIN,
            "head_shoulders_medium_close_max": HEAD_SHOULDERS_MEDIUM_CLOSE_MAX,
            "head_shoulders_close_min": HEAD_SHOULDERS_CLOSE_MIN,
            "head_only_close_min": HEAD_ONLY_CLOSE_MIN,
            "extreme_close_up": "disabled_pending_positive_visual_calibration",
            "full_body": "withheld_because_dwpose_body18_has_ankles_not_feet",
        },
        "invariants": {
            "canonical_framing_is_pose_neutral_anatomical_span": True,
            "single_outlier_joint_cannot_extend_core_span": True,
            "face_scale_uses_target_bound_uniface_observation": True,
            "person_scale_uses_target_bound_yolox_observation": True,
            "sam3d_reconstruction_cannot_create_framing_authority": True,
            "standard_scale_is_optional": True,
            "standard_scale_never_overrides_anatomical_span": True,
            "confident_standard_scale_is_not_mechanically_repeated_with_span_in_caption": True,
            "extreme_close_up_disabled_without_positive_examples": True,
            "full_body_language_not_authorized_from_ankles_alone": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production crop-aware framing semantics.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--uniface-dir", type=Path)
    p.add_argument("--person-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    person_dir = args.person_dir.expanduser().resolve() if args.person_dir else run_dir / DEFAULT_PERSON_SUBDIR
    uniface_dir = _uniface_dir(run_dir, args.uniface_dir)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    if not run_dir.is_dir() or not policy_dir.is_dir() or not person_dir.is_dir() or uniface_dir is None:
        print(
            f"Required input missing: run={run_dir} policy={policy_dir} person={person_dir} uniface={uniface_dir}",
            file=sys.stderr,
        )
        return 2

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in requested]
    if not paths:
        print("No matching production v0.2 perception-policy records found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for policy_path in paths:
        key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output_dir / f"{key}.framing.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            policy = _read_json(policy_path)
            up = uniface_dir / f"{key}.uniface.json"
            pp = person_dir / f"{key}.person_detection.json"
            uniface = _read_json(up) if up.is_file() else None
            person = _read_json(pp) if pp.is_file() else None
            record = evaluate(policy, uniface, person)
            record["sources"] = {
                "perception_policy": str(policy_path),
                "uniface": str(up) if up.is_file() else None,
                "person_detection": str(pp) if pp.is_file() else None,
            }
            _write_json(out_path, record)

        records.append(record)
        span = record.get("anatomical_span") or {}
        scale = record.get("standard_shot_scale") or {}
        print(
            f"{key}: {span.get('upper_anchor')}->{span.get('lower_anchor')} "
            f"ratio={record.get('face_person_height_ratio') if record.get('face_person_height_ratio') is not None else '-'} "
            f"scale={scale.get('label') or '-'} "
            f"surface={((record.get('composer_framing') or {}).get('composer_text') or '-')}"
        )

    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    scale_counts = Counter(str((r.get("standard_shot_scale") or {}).get("label") or "withheld") for r in records)
    surface_counts = Counter(str((r.get("composer_framing") or {}).get("source") or "none") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "uniface_dir": str(uniface_dir),
        "person_dir": str(person_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "standard_shot_scale_counts": dict(sorted(scale_counts.items())),
        "composer_surface_source_counts": dict(sorted(surface_counts.items())),
        "records": records,
    }
    _write_json(output_dir / "framing_semantics.index.json", index)
    print(f"Index: {output_dir / 'framing_semantics.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
