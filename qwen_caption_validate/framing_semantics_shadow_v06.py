from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import framing_semantics_shadow_v05 as v05
from . import framing_semantics_shadow_v01 as v01

SCHEMA_VERSION = "framing-semantics-shadow-0.6"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "framing-semantics-shadow-v0.6"

# Validation-set calibration bands derived from visual review plus observed
# target-face/target-person geometry. These are shadow thresholds, not claimed
# universal photographic definitions.
HEAD_SHOULDERS_MEDIUM_MAX = 0.36
HEAD_SHOULDERS_MEDIUM_CLOSE_MIN = 0.38
HEAD_ONLY_CLOSE_MIN = 0.55


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_dwpose_dir(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_dir() else None
    candidates = []
    for p in run_dir.rglob("*.dwpose.json"):
        if "semantic-v3" in str(p):
            continue
        candidates.append(p.parent)
    if not candidates:
        return None
    counts: dict[Path, int] = {}
    for p in candidates:
        counts[p] = counts.get(p, 0) + 1
    return max(counts, key=counts.get)


def _load_detector_session(device: str):
    try:
        from easy_dwpose import DWposeDetector
    except ImportError as exc:
        raise RuntimeError("easy-dwpose is not installed; re-run build_workspace.sh") from exc
    detector = DWposeDetector(device=device)
    return detector.pose_estimation.session_det


def _resize_for_detector(image: np.ndarray) -> np.ndarray:
    from easy_dwpose.body_estimation import resize_image
    return resize_image(image.copy(), target_resolution=512)


def _infer_person_boxes(session_det: Any, image: np.ndarray) -> list[list[float]]:
    from easy_dwpose.body_estimation.detector import inference_detector
    resized = _resize_for_detector(image)
    h, w = resized.shape[:2]
    raw = inference_detector(session_det, resized)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 4)
    out: list[list[float]] = []
    for x1, y1, x2, y2 in arr:
        out.append([float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)])
    return out


def _area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a: list[float], b: list[float]) -> list[float]:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def _target_keypoint_bbox(dwpose: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(dwpose, dict):
        return None
    target = ((dwpose.get("derived") or {}).get("target") or {})
    kb = target.get("keypoint_bbox") or {}
    keys = ("x0", "y0", "x1", "y1")
    if not all(isinstance(kb.get(k), (int, float)) for k in keys):
        return None
    return [float(kb[k]) for k in keys]


def _choose_person_box(boxes: list[list[float]], keypoint_box: list[float] | None) -> tuple[int | None, list[float] | None, float | None]:
    if not boxes or keypoint_box is None:
        return None, None, None
    kp_area = max(_area(keypoint_box), 1e-9)
    scored = []
    for i, box in enumerate(boxes):
        coverage = _area(_intersection(box, keypoint_box)) / kp_area
        scored.append((coverage, -_area(box), i, box))
    scored.sort(reverse=True)
    coverage, _, i, box = scored[0]
    return i, box, float(coverage)


def _person_geometry(
    boxes: list[list[float]],
    keypoint_box: list[float] | None,
) -> dict[str, Any]:
    i, box, coverage = _choose_person_box(boxes, keypoint_box)
    if box is None:
        return {
            "status": "unavailable",
            "authority": "observed_person_detector",
            "source": "easy_dwpose_yolox_target_bound",
            "reason": "person_box_or_target_keypoint_bbox_unavailable",
            "detected_person_count": len(boxes),
        }
    x1, y1, x2, y2 = box
    cx1 = max(0.0, min(1.0, x1))
    cy1 = max(0.0, min(1.0, y1))
    cx2 = max(0.0, min(1.0, x2))
    cy2 = max(0.0, min(1.0, y2))
    width = max(0.0, cx2 - cx1)
    height = max(0.0, cy2 - cy1)
    return {
        "status": "available" if width > 0 and height > 0 else "unavailable",
        "authority": "observed_person_detector",
        "source": "easy_dwpose_yolox_target_bound",
        "detected_person_count": len(boxes),
        "target_box_index": i,
        "target_keypoint_coverage_fraction": round(float(coverage or 0.0), 6),
        "bbox_xyxy_normalized": [round(float(v), 6) for v in box],
        "visible_bbox_xyxy_normalized": [round(v, 6) for v in (cx1, cy1, cx2, cy2)],
        "visible_width_fraction": round(width, 6),
        "visible_height_fraction": round(height, 6),
        "visible_area_fraction": round(width * height, 6),
        "crop_edges": {
            "left": cx1 <= 0.01,
            "top": cy1 <= 0.01,
            "right": cx2 >= 0.99,
            "bottom": cy2 >= 0.99,
        },
        "note": (
            "YOLOX person detector box already used by Easy-DWPose, associated to the cached target "
            "skeleton by keypoint-bbox containment. This is observed person geometry, not SAM3D reconstruction."
        ),
    }


def _ratio(face: dict[str, Any], person: dict[str, Any]) -> float | None:
    if face.get("status") != "available" or person.get("status") != "available":
        return None
    fh = face.get("height_fraction")
    ph = person.get("visible_height_fraction")
    if not isinstance(fh, (int, float)) or not isinstance(ph, (int, float)) or ph <= 0:
        return None
    return float(fh) / float(ph)


def _candidate(label: str, span: dict[str, Any], ratio: float, face: dict[str, Any], person: dict[str, Any], band: str) -> dict[str, Any]:
    surface = {
        "medium": "medium shot",
        "medium_close_up": "medium close-up",
        "close_up": "close-up",
    }[label]
    return {
        "status": "candidate",
        "label": label,
        "composer_text": surface,
        "basis": [
            "canonical_anatomical_span",
            "target_bound_uniface_retinaface_height_fraction",
            "target_bound_easy_dwpose_yolox_person_height_fraction",
            f"face_person_height_ratio={ratio:.3f}",
            f"face_height_fraction={float(face['height_fraction']):.3f}",
            f"person_visible_height_fraction={float(person['visible_height_fraction']):.3f}",
            f"calibration_band={band}",
        ],
        "pose_family": None,
        "canonical_span": {"upper": span.get("upper_anchor"), "lower": span.get("lower_anchor")},
        "face_scale_authority": "observed_face_detector",
        "person_scale_authority": "observed_person_detector",
        "note": (
            "Optional photographic language calibrated from the ratio of target-bound observed face height "
            "to target-bound observed visible person height. Canonical crop truth remains the anatomical span."
        ),
    }


def _withheld(span: dict[str, Any], ratio: float | None, reason: str) -> dict[str, Any]:
    basis = [reason, f"canonical_span={span.get('upper_anchor')}->{span.get('lower_anchor')}"]
    if ratio is not None:
        basis.append(f"face_person_height_ratio={ratio:.3f}")
    return {
        "status": "withheld",
        "label": None,
        "composer_text": None,
        "basis": basis,
        "pose_family": None,
        "note": "Conventional photographic scale withheld; use the canonical anatomical span instead.",
    }


def _scale(span: dict[str, Any], face: dict[str, Any], person: dict[str, Any]) -> dict[str, Any] | None:
    key = (str(span.get("upper_anchor") or ""), str(span.get("lower_anchor") or ""))
    if key not in {("head", "shoulders"), ("head", "head")}:
        return None
    r = _ratio(face, person)
    if r is None:
        return _withheld(span, None, "target_bound_face_or_person_geometry_unavailable")

    if key == ("head", "shoulders"):
        if r <= HEAD_SHOULDERS_MEDIUM_MAX:
            return _candidate("medium", span, r, face, person, "head_shoulders_medium")
        if r >= HEAD_SHOULDERS_MEDIUM_CLOSE_MIN:
            return _candidate("medium_close_up", span, r, face, person, "head_shoulders_medium_close")
        return _withheld(span, r, "head_shoulders_face_person_ratio_in_abstention_band")

    # No extreme-close-up class is emitted in v0.6. The reviewed head-only
    # examples up to ~0.71 face/person ratio still read as normal close-ups.
    if r >= HEAD_ONLY_CLOSE_MIN:
        return _candidate("close_up", span, r, face, person, "head_only_close")
    return _withheld(span, r, "head_only_face_person_ratio_below_close_calibration")


def _opening(span: dict[str, Any], scale: dict[str, Any]) -> str | None:
    span_text = v01._clean(span.get("composer_text"))
    scale_text = v01._clean(scale.get("composer_text"))
    if scale_text and span_text:
        article = "an" if scale_text.startswith("extreme ") else "a"
        return f"[[trigger]] is shown in {article} {scale_text}, {span_text}"
    if span_text:
        return "[[trigger]] is " + span_text if span_text.startswith("framed ") else "[[trigger]] is shown " + span_text
    if scale_text:
        article = "an" if scale_text.startswith("extreme ") else "a"
        return f"[[trigger]] is shown in {article} {scale_text}"
    return None


def evaluate(
    policy: dict[str, Any],
    fact: dict[str, Any] | None,
    uniface: dict[str, Any] | None,
    person_geometry: dict[str, Any],
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    out = v05._BASE_EVALUATE(policy, fact, uniface, image_size=image_size)
    span = out.get("anatomical_span") if isinstance(out.get("anatomical_span"), dict) else {}
    face = out.get("face_scale_geometry") if isinstance(out.get("face_scale_geometry"), dict) else {}
    scale = _scale(span, face, person_geometry)
    if scale is not None:
        out["standard_shot_scale"] = scale
    else:
        # Keep non-close-family decisions from v0.4/v0.3, e.g. medium for
        # head->hips and full-body for coherent standing head->ankles cases.
        scale = out.get("standard_shot_scale") if isinstance(out.get("standard_shot_scale"), dict) else {}

    out["person_scale_geometry"] = person_geometry
    out["face_person_height_ratio"] = round(_ratio(face, person_geometry), 6) if _ratio(face, person_geometry) is not None else None
    out["proposed_opening_template"] = _opening(span, scale)
    out["schema_version"] = SCHEMA_VERSION

    invariants = out.get("invariants") if isinstance(out.get("invariants"), dict) else {}
    invariants.update(
        close_family_scale_uses_face_to_person_relative_geometry=True,
        person_geometry_is_observed_yolox_not_sam3d_reconstruction=True,
        person_box_is_bound_to_cached_dwpose_target=True,
        head_shoulders_scale_has_ratio_abstention_band=True,
        extreme_close_up_disabled_without_positive_visual_calibration=True,
    )
    out["invariants"] = invariants
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow framing probe using anatomical span + UniFace face geometry + Easy-DWPose YOLOX person geometry.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--fact-dir", type=Path)
    p.add_argument("--uniface-dir", type=Path)
    p.add_argument("--dwpose-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", default="cuda:0")
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
    fact_dir = v01._fact_dir(run_dir, args.fact_dir)
    uniface_dir = v05.v04._uniface_dir(run_dir, args.uniface_dir)
    dwpose_dir = _find_dwpose_dir(run_dir, args.dwpose_dir)
    if not policy_dir.is_dir() or uniface_dir is None or dwpose_dir is None:
        print(f"Required input missing: policy={policy_dir} uniface={uniface_dir} dwpose={dwpose_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in requested]
    if not paths:
        print("No matching perception-policy records found", file=sys.stderr)
        return 2

    session_det = _load_detector_session(args.device)
    records: list[dict[str, Any]] = []

    for policy_path in paths:
        key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output_dir / f"{key}.framing_shadow.json"
        if out_path.exists() and not args.overwrite:
            record = _read_json(out_path)
        else:
            policy = _read_json(policy_path)
            fact = None
            if fact_dir is not None:
                fp = fact_dir / f"{key}.fact_sheet.json"
                if fp.exists():
                    fact = _read_json(fp)
            up = uniface_dir / f"{key}.uniface.json"
            uniface = _read_json(up) if up.exists() else None
            dp = dwpose_dir / f"{key}.dwpose.json"
            dwpose = _read_json(dp) if dp.exists() else None

            image_path = Path(str((uniface or {}).get("image") or "")).expanduser()
            if not image_path.is_file():
                person = {"status": "unavailable", "authority": "observed_person_detector", "source": "easy_dwpose_yolox_target_bound", "reason": "source_image_missing"}
            else:
                image = np.asarray(Image.open(image_path).convert("RGB"))
                boxes = _infer_person_boxes(session_det, image)
                person = _person_geometry(boxes, _target_keypoint_bbox(dwpose))

            record = evaluate(policy, fact, uniface, person)
            record["sources"] = {
                "perception_policy": str(policy_path),
                "fact_sheet": str(fact_dir / f"{key}.fact_sheet.json") if fact_dir else None,
                "uniface": str(up) if up.exists() else None,
                "dwpose": str(dp) if dp.exists() else None,
            }
            v01._write_json(out_path, record)

        records.append(record)
        span = record.get("anatomical_span") or {}
        scale = record.get("standard_shot_scale") or {}
        person = record.get("person_scale_geometry") or {}
        ratio = record.get("face_person_height_ratio")
        ratio_text = f"{float(ratio):.3f}" if isinstance(ratio, (int, float)) else "-"
        print(f"{key}: span={span.get('upper_anchor')}->{span.get('lower_anchor')} ratio={ratio_text} scale={scale.get('label') or '-'} person={person.get('status')}")

    scale_counts = Counter(str((r.get("standard_shot_scale") or {}).get("label") or "withheld") for r in records)
    mode_counts = Counter(str((r.get("pose_gate_shadow") or {}).get("proposed_mode") or "unknown") for r in records)
    person_counts = Counter(str((r.get("person_scale_geometry") or {}).get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "record_count": len(records),
        "standard_shot_scale_counts": dict(sorted(scale_counts.items())),
        "person_scale_geometry_counts": dict(sorted(person_counts.items())),
        "proposed_mode_counts": dict(sorted(mode_counts.items())),
        "calibration": {
            "head_shoulders_medium_max": HEAD_SHOULDERS_MEDIUM_MAX,
            "head_shoulders_medium_close_min": HEAD_SHOULDERS_MEDIUM_CLOSE_MIN,
            "head_only_close_min": HEAD_ONLY_CLOSE_MIN,
            "extreme_close_up": "disabled_pending_positive_visual_calibration",
        },
        "records": records,
    }
    v01._write_json(output_dir / "framing_semantics_shadow.index.json", index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
