from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import framing_semantics_shadow_v06 as shadow

SCHEMA_VERSION = "person-detection-evidence-0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "person-detection-evidence-v0.1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _evaluate(policy: dict[str, Any], session_det: Any) -> dict[str, Any]:
    key = str(policy.get("image_key") or "")
    image_path = Path(str(policy.get("image") or "")).expanduser()
    sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
    dwpose_path = Path(str(sources.get("dwpose") or "")).expanduser()

    if not image_path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "image_key": key,
            "reason": "source_image_missing",
            "person_geometry": {
                "status": "unavailable",
                "authority": "observed_person_detector",
                "source": "easy_dwpose_yolox_target_bound",
                "reason": "source_image_missing",
            },
        }
    if not dwpose_path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "image_key": key,
            "reason": "dwpose_source_missing",
            "person_geometry": {
                "status": "unavailable",
                "authority": "observed_person_detector",
                "source": "easy_dwpose_yolox_target_bound",
                "reason": "dwpose_source_missing",
            },
        }

    image = np.asarray(Image.open(image_path).convert("RGB"))
    boxes = shadow._infer_person_boxes(session_det, image)
    dwpose = _read_json(dwpose_path)
    keypoint_box = shadow._target_keypoint_bbox(dwpose)
    geometry = shadow._person_geometry(boxes, keypoint_box)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if geometry.get("status") == "available" else "unavailable",
        "image_key": key,
        "image": str(image_path),
        "sources": {
            "perception_policy": None,
            "dwpose": str(dwpose_path),
            "detector": "easy_dwpose_yolox_l",
        },
        "person_geometry": geometry,
        "raw_normalized_person_boxes": [
            [round(float(v), 6) for v in box]
            for box in boxes
        ],
        "target_keypoint_bbox_normalized": keypoint_box,
        "invariants": {
            "observed_detector_geometry_not_reconstruction": True,
            "target_person_box_is_bound_to_cached_dwpose_target": True,
            "sam3d_is_not_used": True,
            "rtmpose_is_not_rerun": True,
            "only_existing_easy_dwpose_yolox_detector_is_rerun_for_legacy_cache_enrichment": True,
        },
        "note": (
            "Transitional enrichment for legacy DWPose caches that did not persist the YOLOX person box. "
            "Fresh DWPose production runs should eventually persist this detector observation during the original pass."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recover target-bound Easy-DWPose YOLOX person geometry for legacy DWPose caches."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    if not run_dir.is_dir() or not policy_dir.is_dir():
        print(f"Required directory missing: run={run_dir} policy={policy_dir}", file=sys.stderr)
        return 2

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in requested]
    if not paths:
        print("No matching production v0.2 perception-policy records found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    session_det = shadow._load_detector_session(args.device)
    records: list[dict[str, Any]] = []

    for policy_path in paths:
        key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output_dir / f"{key}.person_detection.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            policy = _read_json(policy_path)
            record = _evaluate(policy, session_det)
            sources = record.setdefault("sources", {})
            sources["perception_policy"] = str(policy_path)
            _write_json(out_path, record)
        records.append(record)

        geom = record.get("person_geometry") if isinstance(record.get("person_geometry"), dict) else {}
        print(
            f"{key}: {record.get('status')} "
            f"h={geom.get('visible_height_fraction') or '-'} "
            f"match={geom.get('target_keypoint_coverage_fraction') or '-'}"
        )

    counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "invariants": {
            "observed_detector_geometry_not_reconstruction": True,
            "sam3d_is_not_used": True,
            "rtmpose_is_not_rerun": True,
            "legacy_cache_enrichment_only": True,
        },
        "records": records,
    }
    _write_json(output_dir / "person_detection_evidence.index.json", index)
    print(f"Index: {output_dir / 'person_detection_evidence.index.json'}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
