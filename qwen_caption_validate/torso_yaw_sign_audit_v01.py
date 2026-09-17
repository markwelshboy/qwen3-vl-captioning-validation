from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import fact_sheet_specialist_normalizer_v05 as phase4b4
from . import sam3d_caption_orientation_v01 as orientation_v01

SCHEMA_VERSION = "torso-yaw-sign-audit-0.1"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.13"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "torso-yaw-sign-audit-v0.1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _body(sheet: dict[str, Any]) -> dict[str, Any]:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    return facts.get("body") if isinstance(facts.get("body"), dict) else {}


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _recompute_signed_yaw(payload: dict[str, Any]) -> float | None:
    forward = payload.get("forward_camera_xyz")
    to_camera = payload.get("to_camera_xyz")
    if not isinstance(forward, list) or len(forward) < 3:
        return None
    if not isinstance(to_camera, list) or len(to_camera) < 3:
        return None
    try:
        return orientation_v01._signed_yaw_between_deg(
            np.asarray(forward[:3], dtype=np.float64),
            np.asarray(to_camera[:3], dtype=np.float64),
        )
    except Exception:
        return None


def _segment_record(
    name: str,
    payload: dict[str, Any],
    *,
    composer_eligible: bool,
) -> dict[str, Any]:
    yaw = _finite_number(payload.get("yaw_deg"))
    recomputed = _recompute_signed_yaw(payload)
    mapped = phase4b4._turn_direction(yaw)
    stored = payload.get("turn_direction")
    stored_publishable = bool(payload.get("turn_direction_publishable"))
    expected_publishable = bool(mapped and composer_eligible)
    threshold = float(phase4b4.TURN_DIRECTION_MIN_ABS_YAW_DEG)

    if yaw is None:
        status = "yaw_unavailable"
    elif stored != mapped:
        status = "stored_direction_mismatch"
    elif stored_publishable != expected_publishable:
        status = "publishability_mismatch"
    elif recomputed is not None and abs(recomputed - yaw) > 0.2:
        status = "raw_vector_yaw_mismatch"
    else:
        status = "consistent"

    return {
        "segment": name,
        "status": status,
        "signed_yaw_deg": yaw,
        "yaw_magnitude_deg": payload.get("yaw_magnitude_deg"),
        "orientation_band": payload.get("orientation_band"),
        "approx_yaw_deg": payload.get("approx_yaw_deg"),
        "deadband_threshold_deg": threshold,
        "inside_near_frontal_deadband": bool(yaw is not None and abs(yaw) <= threshold),
        "production_mapped_direction": mapped,
        "stored_turn_direction": stored,
        "composer_eligible": composer_eligible,
        "expected_turn_direction_publishable": expected_publishable,
        "stored_turn_direction_publishable": stored_publishable,
        "direction_matches_production_helper": stored == mapped,
        "publishability_matches_production_helper": stored_publishable == expected_publishable,
        "raw_vectors": {
            "forward_camera_xyz": payload.get("forward_camera_xyz"),
            "to_camera_xyz": payload.get("to_camera_xyz"),
            "recomputed_signed_yaw_deg": round(recomputed, 3) if recomputed is not None else None,
            "recomputed_matches_stored_yaw": bool(
                yaw is not None and recomputed is not None and abs(recomputed - yaw) <= 0.2
            ),
        },
        "production_convention": (
            "positive signed yaw -> frame_left; negative signed yaw -> frame_right; "
            f"direction withheld when abs(yaw)<={threshold:g} degrees"
        ),
    }


def audit_sheet(sheet: dict[str, Any], *, source_path: str | None = None) -> dict[str, Any]:
    image_key = str(sheet.get("image_key") or "unknown")
    body = _body(sheet)
    geometry = body.get("torso_geometry") if isinstance(body.get("torso_geometry"), dict) else {}
    composer_eligible = bool(geometry.get("composer_eligible"))

    if not geometry or not geometry.get("available"):
        return {
            "schema_version": SCHEMA_VERSION,
            "image_key": image_key,
            "source_record": source_path,
            "status": "torso_geometry_unavailable",
            "production_helper": "fact_sheet_specialist_normalizer_v05._turn_direction",
            "segments": [],
        }

    segments: list[dict[str, Any]] = []
    for key in ("body_root_orientation", "upper_torso_orientation"):
        payload = geometry.get(key) if isinstance(geometry.get(key), dict) else None
        if payload:
            segments.append(_segment_record(key, payload, composer_eligible=composer_eligible))

    summary = geometry.get("caption_orientation") if isinstance(geometry.get("caption_orientation"), dict) else {}
    preferred = (
        geometry.get("upper_torso_orientation")
        if isinstance(geometry.get("upper_torso_orientation"), dict)
        and geometry.get("upper_torso_orientation", {}).get("orientation_band") is not None
        else geometry.get("body_root_orientation")
    )
    preferred = preferred if isinstance(preferred, dict) else {}
    preferred_mapped = phase4b4._turn_direction(preferred.get("yaw_deg"))

    bad = [s for s in segments if s.get("status") != "consistent"]
    status = "consistent" if segments and not bad else ("no_orientation_segments" if not segments else "mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "image_key": image_key,
        "source_record": source_path,
        "status": status,
        "production_helper": "fact_sheet_specialist_normalizer_v05._turn_direction",
        "production_orientation_math": "sam3d_caption_orientation_v01._signed_yaw_between_deg",
        "production_deadband_deg": phase4b4.TURN_DIRECTION_MIN_ABS_YAW_DEG,
        "production_convention": "positive_signed_yaw=frame_left; negative_signed_yaw=frame_right",
        "composer_eligible": composer_eligible,
        "segments": segments,
        "caption_orientation": {
            "mode": summary.get("mode"),
            "preferred_orientation_band": summary.get("preferred_orientation_band"),
            "preferred_approx_yaw_deg": summary.get("preferred_approx_yaw_deg"),
            "preferred_turn_direction_stored": summary.get("preferred_turn_direction"),
            "preferred_turn_direction_from_production_helper": preferred_mapped,
            "turn_direction_publishable_stored": summary.get("turn_direction_publishable"),
            "relative_twist_yaw_deg": summary.get("relative_twist_yaw_deg"),
            "relative_twist_magnitude_deg": summary.get("relative_twist_magnitude_deg"),
        },
        "interpretation": (
            "This audit verifies the actual production sign mapping and deadband against the stored fact-sheet geometry. "
            "It does not use reconstruction output as evidence that frame-left/frame-right is visually correct. "
            "Visual controls should be compared directly with signed yaw after this internal consistency check."
        ),
    }


def _iter_paths(input_dir: Path, only: set[str]) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.fact_sheet.json")):
        key = path.name.removesuffix(".fact_sheet.json")
        if only and key not in only:
            continue
        yield path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit signed torso yaw using the exact production sign helper and SAM3D orientation math. "
            "No model calls and no fact-sheet mutation."
        )
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--input-dir", type=Path)
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

    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    only = {str(x) for x in args.only}
    paths = list(_iter_paths(input_dir, only))
    if only:
        found = {p.name.removesuffix(".fact_sheet.json") for p in paths}
        missing = sorted(only - found)
        if missing:
            print(f"Requested fact sheets not found: {', '.join(missing)}", file=sys.stderr)
            return 2
    if not paths:
        print(f"No fact sheets found in: {input_dir}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    for path in paths:
        sheet = _read_json(path)
        record = audit_sheet(sheet, source_path=str(path))
        out_path = output_dir / f"{record['image_key']}.torso_yaw_sign_audit.json"
        if out_path.exists() and not args.overwrite:
            print(f"Output exists (use --overwrite): {out_path}", file=sys.stderr)
            return 2
        _write_json(out_path, record)
        records.append(record)

        pieces = []
        for segment in record.get("segments") or []:
            pieces.append(
                f"{segment['segment']}={segment.get('signed_yaw_deg')}->{segment.get('production_mapped_direction') or 'withheld'}"
            )
        print(f"{record['image_key']}: {record['status']} | " + "; ".join(pieces))

    counts = Counter(str(r.get("status")) for r in records)
    index = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "production_convention": "positive_signed_yaw=frame_left; negative_signed_yaw=frame_right",
        "deadband_deg": phase4b4.TURN_DIRECTION_MIN_ABS_YAW_DEG,
        "records": records,
    }
    index_path = output_dir / "torso_yaw_sign_audit.index.json"
    _write_json(index_path, index)
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
