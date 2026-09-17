from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from . import framing_semantics_shadow_v01 as v01
from . import framing_semantics_shadow_v03 as v03

SCHEMA_VERSION = "framing-semantics-shadow-0.4"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "framing-semantics-shadow-v0.4"
DEFAULT_UNIFACE_SUBDIRS = (
    Path("face-authority-uniface-v02-all"),
    Path("face-authority-uniface-all"),
)

# These are deliberately conservative calibration bands, not universal
# photographic definitions.  On the current 87-image validation set there is
# a clean face-height gap between 0.266 and 0.302 for head->shoulders crops,
# and between 0.631 and 0.690 for head-only crops.  The gaps are retained as
# abstention bands rather than collapsed to a brittle single threshold.
HEAD_SHOULDERS_MEDIUM_CLOSE_MAX = 0.27
HEAD_SHOULDERS_CLOSE_MIN = 0.30
HEAD_ONLY_CLOSE_MAX = 0.64
HEAD_ONLY_EXTREME_MIN = 0.68


def _uniface_dir(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_dir() else None
    for rel in DEFAULT_UNIFACE_SUBDIRS:
        candidate = run_dir / rel
        if candidate.is_dir():
            return candidate
    return None


def _image_size(uniface: dict[str, Any]) -> tuple[int, int] | None:
    raw = uniface.get("image")
    if not isinstance(raw, str) or not raw.strip():
        return None
    image = cv2.imread(str(Path(raw).expanduser()))
    if image is None:
        return None
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _face_geometry(
    uniface: dict[str, Any] | None,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
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

    size = image_size or _image_size(uniface)
    if size is None:
        return {
            "status": "unavailable",
            "authority": "observed_face_detector",
            "source": "uniface_retinaface_target_bound",
            "source_status": source_status,
            "bbox_xyxy": [float(v) for v in box[:4]],
            "reason": "source_image_dimensions_unavailable",
        }

    width, height = size
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    cx1 = max(0.0, min(float(width), x1))
    cy1 = max(0.0, min(float(height), y1))
    cx2 = max(0.0, min(float(width), x2))
    cy2 = max(0.0, min(float(height), y2))
    fw = max(0.0, cx2 - cx1)
    fh = max(0.0, cy2 - cy1)
    if fw <= 0.0 or fh <= 0.0:
        return {
            "status": "unavailable",
            "authority": "observed_face_detector",
            "source": "uniface_retinaface_target_bound",
            "source_status": source_status,
            "bbox_xyxy": [x1, y1, x2, y2],
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
        "bbox_clipped_to_image": any(abs(a - b) > 1e-6 for a, b in zip((x1, y1, x2, y2), (cx1, cy1, cx2, cy2))),
        "note": (
            "RetinaFace target-bound detector bbox intersected with the source image. "
            "This is observed face-scale evidence, not reconstructed head geometry."
        ),
    }


def _candidate(label: str, face: dict[str, Any], span: dict[str, Any], band: str) -> dict[str, Any]:
    surface = {
        "extreme_close_up": "extreme close-up",
        "close_up": "close-up",
        "medium_close_up": "medium close-up",
    }[label]
    return {
        "status": "candidate",
        "label": label,
        "composer_text": surface,
        "basis": [
            "canonical_anatomical_span",
            "target_bound_uniface_retinaface_height_fraction",
            f"face_height_fraction={face['height_fraction']:.3f}",
            f"calibration_band={band}",
        ],
        "pose_family": None,
        "span_coherent": True,
        "canonical_span": {
            "upper": span.get("upper_anchor"),
            "lower": span.get("lower_anchor"),
        },
        "face_scale_authority": "observed_face_detector",
        "note": (
            "Optional photographic language calibrated from target-bound UniFace face height. "
            "Canonical crop truth remains the anatomical span."
        ),
    }


def _withheld_close_family(span: dict[str, Any], face: dict[str, Any], reason: str) -> dict[str, Any]:
    basis = [
        reason,
        f"canonical_span={span.get('upper_anchor')}->{span.get('lower_anchor')}",
    ]
    if face.get("height_fraction") is not None:
        basis.append(f"face_height_fraction={float(face['height_fraction']):.3f}")
    if face.get("reason"):
        basis.append(f"face_geometry_reason={face['reason']}")
    return {
        "status": "withheld",
        "label": None,
        "composer_text": None,
        "basis": basis,
        "pose_family": None,
        "face_scale_authority": "observed_face_detector",
        "note": (
            "Close-family photographic scale withheld. Use the canonical anatomical span when "
            "target-bound observed face geometry is unavailable or lies in a calibration abstention band."
        ),
    }


def _close_family_scale(span: dict[str, Any], face: dict[str, Any]) -> dict[str, Any] | None:
    key = (str(span.get("upper_anchor") or ""), str(span.get("lower_anchor") or ""))
    if key not in {("head", "head"), ("head", "shoulders")}:
        return None

    if face.get("status") != "available":
        return _withheld_close_family(span, face, "target_bound_uniface_face_unavailable")

    h = float(face["height_fraction"])
    if key == ("head", "shoulders"):
        if h <= HEAD_SHOULDERS_MEDIUM_CLOSE_MAX:
            return _candidate("medium_close_up", face, span, "head_shoulders_medium_close")
        if h >= HEAD_SHOULDERS_CLOSE_MIN:
            return _candidate("close_up", face, span, "head_shoulders_close")
        return _withheld_close_family(span, face, "head_shoulders_face_height_in_abstention_band")

    if h <= HEAD_ONLY_CLOSE_MAX:
        return _candidate("close_up", face, span, "head_only_close")
    if h >= HEAD_ONLY_EXTREME_MIN:
        return _candidate("extreme_close_up", face, span, "head_only_extreme")
    return _withheld_close_family(span, face, "head_only_face_height_in_abstention_band")


def _opening(span: dict[str, Any], scale: dict[str, Any]) -> str | None:
    span_text = v01._clean(span.get("composer_text"))
    scale_text = v01._clean(scale.get("composer_text"))
    if scale_text and span_text:
        return f"[[trigger]] is shown in a {scale_text}, {span_text}"
    if span_text:
        return "[[trigger]] is " + span_text if span_text.startswith("framed ") else "[[trigger]] is shown " + span_text
    if scale_text:
        return f"[[trigger]] is shown in a {scale_text}"
    return None


def evaluate(
    policy: dict[str, Any],
    fact: dict[str, Any] | None = None,
    uniface: dict[str, Any] | None = None,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    out = v03.evaluate(policy, fact)
    span = out.get("anatomical_span") if isinstance(out.get("anatomical_span"), dict) else {}
    face = _face_geometry(uniface, image_size=image_size)
    close_scale = _close_family_scale(span, face)
    if close_scale is not None:
        out["standard_shot_scale"] = close_scale
        out["proposed_opening_template"] = _opening(span, close_scale)

    out["schema_version"] = SCHEMA_VERSION
    out["face_scale_geometry"] = face
    invariants = out.get("invariants") if isinstance(out.get("invariants"), dict) else {}
    invariants.update(
        close_family_scale_uses_target_bound_face_detector_not_dwpose_body_bbox=True,
        face_scale_bbox_is_intersected_with_source_image=True,
        uniface_no_face_cannot_create_close_family_scale=True,
        close_family_threshold_gaps_are_abstention_bands=True,
        reconstructed_head_geometry_cannot_create_face_scale_authority=True,
    )
    out["invariants"] = invariants
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Shadow probe for anatomical-span framing with target-bound UniFace face-scale "
            "calibration for close-family photographic language."
        )
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--fact-dir", type=Path)
    p.add_argument("--uniface-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / v01.DEFAULT_POLICY_SUBDIR
    if not policy_dir.is_dir():
        print(f"Perception-policy directory not found: {policy_dir}", file=sys.stderr)
        return 2

    fact_dir = v01._fact_dir(run_dir, args.fact_dir)
    uniface_dir = _uniface_dir(run_dir, args.uniface_dir)
    if uniface_dir is None:
        print("UniFace directory not found; pass --uniface-dir DIR", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in requested]
    if not paths:
        print(f"No matching perception-policy records found in {policy_dir}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    for policy_path in paths:
        key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output_dir / f"{key}.framing_shadow.json"
        if out_path.is_file() and not args.overwrite:
            record = v01._read_json(out_path)
        else:
            policy = v01._read_json(policy_path)
            fact = None
            if fact_dir is not None:
                fact_path = fact_dir / f"{key}.fact_sheet.json"
                if fact_path.is_file():
                    fact = v01._read_json(fact_path)
            uniface_path = uniface_dir / f"{key}.uniface.json"
            uniface = v01._read_json(uniface_path) if uniface_path.is_file() else None
            record = evaluate(policy, fact, uniface)
            record["sources"] = {
                "perception_policy": str(policy_path),
                "fact_sheet": str(fact_dir / f"{key}.fact_sheet.json") if fact_dir is not None else None,
                "uniface": str(uniface_path) if uniface_path.is_file() else None,
            }
            v01._write_json(out_path, record)
        records.append(record)
        span = record.get("anatomical_span") if isinstance(record.get("anatomical_span"), dict) else {}
        scale = record.get("standard_shot_scale") if isinstance(record.get("standard_shot_scale"), dict) else {}
        face = record.get("face_scale_geometry") if isinstance(record.get("face_scale_geometry"), dict) else {}
        gate = record.get("pose_gate_shadow") if isinstance(record.get("pose_gate_shadow"), dict) else {}
        face_h = face.get("height_fraction")
        face_h_text = f"{float(face_h):.3f}" if isinstance(face_h, (int, float)) else "-"
        print(
            f"{key}: span={span.get('upper_anchor')}->{span.get('lower_anchor')} "
            f"face_h={face_h_text} scale={scale.get('label') or '-'} "
            f"broad={gate.get('broad_pose_supported')} mode={gate.get('proposed_mode')}"
        )

    scale_counts = Counter(
        str((r.get("standard_shot_scale") or {}).get("label") or "withheld")
        for r in records
    )
    mode_counts = Counter(
        str((r.get("pose_gate_shadow") or {}).get("proposed_mode") or "unknown")
        for r in records
    )
    face_counts = Counter(
        str((r.get("face_scale_geometry") or {}).get("status") or "unknown")
        for r in records
    )
    changed = [str(r.get("image_key")) for r in records if (r.get("pose_gate_shadow") or {}).get("changed_from_legacy")]
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "fact_dir": str(fact_dir) if fact_dir else None,
        "uniface_dir": str(uniface_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "standard_shot_scale_counts": dict(sorted(scale_counts.items())),
        "proposed_mode_counts": dict(sorted(mode_counts.items())),
        "face_scale_geometry_counts": dict(sorted(face_counts.items())),
        "pose_gate_changed_keys": changed,
        "calibration": {
            "head_shoulders_medium_close_max": HEAD_SHOULDERS_MEDIUM_CLOSE_MAX,
            "head_shoulders_close_min": HEAD_SHOULDERS_CLOSE_MIN,
            "head_only_close_max": HEAD_ONLY_CLOSE_MAX,
            "head_only_extreme_min": HEAD_ONLY_EXTREME_MIN,
            "metric": "target_bound_uniface_retinaface_visible_bbox_height_fraction",
            "note": "Bands were chosen around clean gaps in the current 87-image validation census and remain shadow-only.",
        },
        "records": records,
    }
    v01._write_json(output_dir / "framing_semantics_shadow.index.json", index)
    print(f"Index: {output_dir / 'framing_semantics_shadow.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
