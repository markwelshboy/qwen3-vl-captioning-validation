from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v05 as phase4b4

SCHEMA_VERSION = "local-relation-geometry-shadow-0.1"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "local-relation-geometry-shadow-v0.1"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FAMILY_REGISTRY = (
    PACKAGE_ROOT / "notes" / "semantic_v3_blind_crop_family_registry_v01.json"
)

_HEAD_RELATION_RE = re.compile(
    r"\b(?:hand|fist|palm|fingers?|wrist|forearm)\b.{0,55}\b(?:face|chin)\b"
    r"|\b(?:face|chin)\b.{0,55}\b(?:hand|fist|palm|fingers?|wrist|forearm)\b",
    re.I,
)
_HIP_RELATION_RE = re.compile(
    r"\b(?:hand|wrist|forearm)\b.{0,55}\b(?:hip|waist)\b"
    r"|\b(?:hip|waist)\b.{0,55}\b(?:hand|wrist|forearm)\b",
    re.I,
)
_CONTACT_WORD_RE = re.compile(
    r"\b(?:on|against|rest(?:s|ing|ed)?\s+on|press(?:ed|ing)?\s+against|"
    r"support(?:s|ing|ed)?|contact(?:ing)?|touch(?:es|ing|ed)?)\b",
    re.I,
)
_PROXIMITY_WORD_RE = re.compile(
    r"\b(?:near|nearby|beside|adjacent|close\s+to|positioned\s+near|placed\s+near)\b",
    re.I,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _distance(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_to_segment_distance(
    point: tuple[float, float] | None,
    start: tuple[float, float] | None,
    end: tuple[float, float] | None,
) -> float | None:
    """Euclidean distance from a point to the visible 2-D forearm segment."""
    if point is None or start is None or end is None:
        return None
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    px, py = float(point[0]), float(point[1])
    vx, vy = ex - sx, ey - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-8:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / denom))
    qx, qy = sx + t * vx, sy + t * vy
    return math.hypot(px - qx, py - qy)


def _safe_div(value: float | None, scale: float | None) -> float | None:
    if value is None or scale is None or scale <= 1e-8:
        return None
    return round(value / scale, 3)


def _configuration_items(sheet: dict[str, Any]) -> list[dict[str, Any]]:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    values = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    return [value for value in values if isinstance(value, dict)]


def _source_text(item: dict[str, Any]) -> str | None:
    binding = (
        item.get("laterality_binding")
        if isinstance(item.get("laterality_binding"), dict)
        else {}
    )
    return (
        _clean(binding.get("source_text"))
        or _clean(item.get("text"))
        or _clean(item.get("normalized_text"))
        or _clean(item.get("composer_text"))
    )


def _relation_candidates(sheet: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _configuration_items(sheet):
        source = _source_text(item)
        if not source:
            continue
        relation_types: list[str] = []
        if _HEAD_RELATION_RE.search(source):
            relation_types.append("hand_or_forearm_near_face_or_chin")
        if _HIP_RELATION_RE.search(source):
            relation_types.append("hand_or_forearm_near_hip_or_waist")
        if not relation_types:
            continue
        for relation_type in relation_types:
            out.append(
                {
                    "relation_type": relation_type,
                    "source_text": source,
                    "composer_text": _clean(item.get("composer_text")),
                    "promotion_status": _clean(item.get("promotion_status")),
                    "specialist_owner": _clean(item.get("specialist_owner")),
                    "surface_strength": (
                        "contact"
                        if _CONTACT_WORD_RE.search(source)
                        else "proximity"
                        if _PROXIMITY_WORD_RE.search(source)
                        else "unspecified"
                    ),
                    "laterality_binding": (
                        item.get("laterality_binding")
                        if isinstance(item.get("laterality_binding"), dict)
                        else None
                    ),
                }
            )
    return out


def _geometry(
    points: dict[str, tuple[float, float] | None],
) -> dict[str, Any]:
    body_scale = phase4b4._body_scale(points)
    shoulder_scale = phase4b4._distance(
        points.get("left_shoulder"),
        points.get("right_shoulder"),
    )
    head_points = [
        points.get(name)
        for name in (
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
        )
        if points.get(name) is not None
    ]

    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        wrist = points.get(f"{side}_wrist")
        hip = points.get(f"{side}_hip")
        elbow = points.get(f"{side}_elbow")
        shoulder = points.get(f"{side}_shoulder")

        wrist_face_distances = [
            _distance(wrist, head)
            for head in head_points
            if wrist is not None
        ]
        wrist_face_distances = [
            value for value in wrist_face_distances
            if value is not None
        ]
        min_wrist_face = min(wrist_face_distances) if wrist_face_distances else None

        elbow_face_distances = [
            _distance(elbow, head)
            for head in head_points
            if elbow is not None
        ]
        elbow_face_distances = [
            value for value in elbow_face_distances
            if value is not None
        ]
        min_elbow_face = min(elbow_face_distances) if elbow_face_distances else None

        forearm_face_distances = [
            _point_to_segment_distance(head, elbow, wrist)
            for head in head_points
            if elbow is not None and wrist is not None
        ]
        forearm_face_distances = [
            value for value in forearm_face_distances
            if value is not None
        ]
        min_forearm_face = min(forearm_face_distances) if forearm_face_distances else None

        # For relation-existence auditing, visible forearm geometry is the
        # strongest contradiction signal when both elbow and wrist exist. If
        # the elbow is unavailable, retain wrist-only evidence rather than
        # treating missing geometry as counterevidence.
        min_upper_limb_face = (
            min_forearm_face
            if min_forearm_face is not None
            else min_wrist_face
        )
        wrist_hip = _distance(wrist, hip)

        dx = (
            wrist[0] - hip[0]
            if wrist is not None and hip is not None
            else None
        )
        dy = (
            wrist[1] - hip[1]
            if wrist is not None and hip is not None
            else None
        )

        sides[side] = {
            "wrist_observed": wrist is not None,
            "elbow_observed": elbow is not None,
            "shoulder_observed": shoulder is not None,
            "hip_observed": hip is not None,
            "wrist_to_face_min_norm_body": _safe_div(min_wrist_face, body_scale),
            "wrist_to_face_min_norm_shoulders": _safe_div(
                min_wrist_face,
                shoulder_scale,
            ),
            "wrist_to_nose_norm_body": _safe_div(
                _distance(wrist, points.get("nose")),
                body_scale,
            ),
            "wrist_to_neck_norm_body": _safe_div(
                _distance(wrist, points.get("neck")),
                body_scale,
            ),
            "elbow_to_face_min_norm_body": _safe_div(
                min_elbow_face,
                body_scale,
            ),
            "forearm_segment_to_face_min_norm_body": _safe_div(
                min_forearm_face,
                body_scale,
            ),
            "upper_limb_to_face_min_norm_body": _safe_div(
                min_upper_limb_face,
                body_scale,
            ),
            "wrist_to_same_hip_norm_body": _safe_div(
                wrist_hip,
                body_scale,
            ),
            "wrist_minus_hip_dx_norm_body": _safe_div(
                abs(dx) if dx is not None else None,
                body_scale,
            ),
            "wrist_minus_hip_dy_norm_body": _safe_div(
                abs(dy) if dy is not None else None,
                body_scale,
            ),
            "wrist_minus_hip_dx_signed_norm_body": (
                round(dx / body_scale, 3)
                if dx is not None
                and body_scale is not None
                and body_scale > 1e-8
                else None
            ),
            "wrist_minus_hip_dy_signed_norm_body": (
                round(dy / body_scale, 3)
                if dy is not None
                and body_scale is not None
                and body_scale > 1e-8
                else None
            ),
            "elbow_angle_deg": (
                round(
                    phase4b4._joint_angle(
                        shoulder,
                        elbow,
                        wrist,
                    ),
                    1,
                )
                if phase4b4._joint_angle(
                    shoulder,
                    elbow,
                    wrist,
                )
                is not None
                else None
            ),
        }

    visible_face_distances = [
        (side, payload["wrist_to_face_min_norm_body"])
        for side, payload in sides.items()
        if payload["wrist_to_face_min_norm_body"] is not None
    ]
    nearest_face = (
        min(visible_face_distances, key=lambda value: value[1])
        if visible_face_distances
        else None
    )
    visible_upper_limb_distances = [
        (side, payload["upper_limb_to_face_min_norm_body"])
        for side, payload in sides.items()
        if payload["upper_limb_to_face_min_norm_body"] is not None
    ]
    nearest_upper_limb_face = (
        min(visible_upper_limb_distances, key=lambda value: value[1])
        if visible_upper_limb_distances
        else None
    )

    return {
        "body_scale_px": round(body_scale, 3) if body_scale is not None else None,
        "shoulder_scale_px": (
            round(shoulder_scale, 3)
            if shoulder_scale is not None
            else None
        ),
        "nearest_wrist_to_face_side": nearest_face[0] if nearest_face else None,
        "nearest_wrist_to_face_norm_body": nearest_face[1] if nearest_face else None,
        "nearest_upper_limb_to_face_side": (
            nearest_upper_limb_face[0] if nearest_upper_limb_face else None
        ),
        "nearest_upper_limb_to_face_norm_body": (
            nearest_upper_limb_face[1] if nearest_upper_limb_face else None
        ),
        "sides": sides,
        "production_hand_on_hip_binding": phase4b4._hand_on_hip_binding(points),
    }


def _family_lookup(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for family in registry.get("families") or []:
        if not isinstance(family, dict):
            continue
        order = [str(x) for x in family.get("wide_to_tight") or []]
        fov = (
            family.get("relative_fov_area")
            if isinstance(family.get("relative_fov_area"), dict)
            else {}
        )
        for key in family.get("members") or []:
            key = str(key)
            lookup[key] = {
                "family_id": family.get("family_id"),
                "kind": family.get("kind"),
                "wide_to_tight": order,
                "crop_rank": (
                    order.index(key)
                    if key in order
                    else None
                ),
                "relative_fov_area": fov.get(key),
            }
    return lookup


def analyze_sheet(
    sheet: dict[str, Any],
    *,
    family_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = str(sheet.get("image_key") or "")
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    framing = (
        facts.get("framing")
        if isinstance(facts.get("framing"), dict)
        else {}
    )
    composer_framing = (
        framing.get("composer_framing")
        if isinstance(framing.get("composer_framing"), dict)
        else {}
    )

    candidates = _relation_candidates(sheet)
    points, error = phase4b4._load_dwpose_points_for_sheet(sheet)
    geometry = _geometry(points) if points is not None else None

    return {
        "schema_version": SCHEMA_VERSION,
        "image_key": key,
        "status": "ok" if points is not None else "geometry_unavailable",
        "family": family_meta or {},
        "framing": {
            "composer_text": _clean(composer_framing.get("composer_text")),
            "broad_pose_supported": bool(framing.get("broad_pose_supported")),
            "anatomical_span": framing.get("anatomical_span"),
        },
        "relation_candidates": candidates,
        "candidate_count": len(candidates),
        "geometry": geometry,
        "geometry_error": error,
        "interpretation": (
            "Shadow diagnostic only. Qwen proposes semantic relations; DWPose geometry "
            "is measured independently here. No fact-sheet or caption authority is changed."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shadow-audit local hand/forearm relations against DWPose geometry. "
            "No model calls and no mutations."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--family-registry", type=Path, default=DEFAULT_FAMILY_REGISTRY)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir
        else run_dir / DEFAULT_INPUT_SUBDIR
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / DEFAULT_OUTPUT_SUBDIR
    )
    registry_path = args.family_registry.expanduser().resolve()

    if not run_dir.is_dir() or not input_dir.is_dir():
        print(
            f"Required directory missing: run={run_dir} input={input_dir}",
            file=sys.stderr,
        )
        return 2
    if not registry_path.is_file():
        print(f"Family registry not found: {registry_path}", file=sys.stderr)
        return 2

    family_lookup = _family_lookup(_read_json(registry_path))
    requested = {str(value) for value in args.only}
    paths = sorted(input_dir.glob("*.fact_sheet.json"))
    if requested:
        paths = [
            path
            for path in paths
            if path.name.removesuffix(".fact_sheet.json") in requested
        ]
        found = {
            path.name.removesuffix(".fact_sheet.json")
            for path in paths
        }
        missing = sorted(requested - found)
        if missing:
            print(
                "Requested fact sheets not found: " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
    if not paths:
        print(f"No fact sheets found in: {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".fact_sheet.json")
        out_path = output_dir / f"{key}.local_relation_geometry.json"
        if out_path.exists() and not args.overwrite:
            print(f"Output exists (use --overwrite): {out_path}", file=sys.stderr)
            return 2

        record = analyze_sheet(
            _read_json(path),
            family_meta=family_lookup.get(key),
        )
        _write_json(out_path, record)
        records.append(record)

        geometry = record.get("geometry") or {}
        hip = geometry.get("production_hand_on_hip_binding") or {}
        print(
            f"{key}: {record['status']} "
            f"framing={record['framing']['composer_text'] or '-'} "
            f"candidates={record['candidate_count']} "
            f"face_wrist={geometry.get('nearest_wrist_to_face_norm_body')} "
            f"face_limb={geometry.get('nearest_upper_limb_to_face_norm_body')} "
            f"hip={hip.get('anatomical_side') or '-'}:"
            f"{hip.get('wrist_hip_distance_norm')}"
        )

    counts = Counter(str(record.get("status") or "unknown") for record in records)
    relation_counts = Counter(
        candidate["relation_type"]
        for record in records
        for candidate in record.get("relation_candidates") or []
    )
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "family_registry": str(registry_path),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "relation_candidate_counts": dict(sorted(relation_counts.items())),
        "records": records,
    }
    index_path = output_dir / "local_relation_geometry.index.json"
    _write_json(index_path, index)
    print(f"Index: {index_path}")
    return 1 if counts.get("geometry_unavailable", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
