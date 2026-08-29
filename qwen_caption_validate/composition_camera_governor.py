from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _norm_elevation(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "verylow": "very_low",
        "low_angle": "low",
        "eye": "eye_level",
        "eyelevel": "eye_level",
        "high_angle": "high",
        "veryhigh": "very_high",
    }
    text = aliases.get(text, text)
    return text if text in {"very_low", "low", "eye_level", "high", "very_high", "unknown"} else None


def govern_camera_elevation(
    composition_payload: dict[str, Any],
    camera_diagnostic: dict[str, Any] | None,
) -> dict[str, Any]:
    gestalt = composition_payload.get("gestalt")
    if not isinstance(gestalt, dict):
        gestalt = {}
    source_camera = gestalt.get("camera")
    if not isinstance(source_camera, dict):
        source_camera = {}

    source_elevation = _norm_elevation(source_camera.get("elevation"))
    source_pitch = source_camera.get("pitch")
    source_confidence = source_camera.get("confidence")

    diagnostic = camera_diagnostic if isinstance(camera_diagnostic, dict) else {}
    low = diagnostic.get("low_angle_support")
    if not isinstance(low, dict):
        low = {}

    geometry_action = str(low.get("action") or "missing").lower()
    geometry_elevation = _norm_elevation(low.get("qualified_elevation") or low.get("candidate_elevation"))
    geometry_band = str(low.get("confidence_band") or "withheld").lower()
    geometry_authority = low.get("authority")
    geometry_reasons = list(low.get("reasons") or [])
    geometry_limitations = list(low.get("limitations") or [])

    final_elevation = source_elevation
    action = "vlm_preserved"
    authority = "composition_gestalt"
    reasons: list[str] = []
    limitations: list[str] = []

    if geometry_action == "qualified" and geometry_elevation == "low":
        if source_elevation in {"very_low", "low"}:
            # Geometry establishes the low family, but does not distinguish low
            # from very-low. Preserve the VLM's finer low-family label.
            final_elevation = source_elevation
            action = "vlm_low_corroborated_by_geometry"
            authority = "composition_gestalt_plus_dwpose_sam3d_camera_geometry"
            reasons.append("qualified deterministic geometry independently supports the VLM low-angle family")
        else:
            final_elevation = "low"
            action = "vlm_elevation_overridden_by_geometry"
            authority = "dwpose_visible_torso_plus_sam3d_camera_geometry"
            reasons.append(
                f"qualified deterministic camera geometry overrides VLM elevation {source_elevation or 'unknown'} with low"
            )
        reasons.extend(geometry_reasons)

    elif geometry_action == "supporting" and geometry_elevation == "low":
        if source_elevation in {"very_low", "low"}:
            final_elevation = source_elevation
            action = "vlm_low_supported_by_non_authoritative_geometry"
            authority = "composition_gestalt_with_sam3d_supporting_geometry"
            reasons.append(
                "SAM3D low-angle geometry agrees with the VLM but lacks DWPose visibility required for independent authority"
            )
            reasons.extend(geometry_reasons)
            limitations.extend(geometry_limitations)
        else:
            action = "vlm_preserved_geometry_support_insufficient_to_override"
            authority = "composition_gestalt"
            reasons.append(
                "SAM3D proposes a low-angle family but DWPose visibility is insufficient, so it cannot override the VLM"
            )
            limitations.extend(geometry_limitations)

    elif geometry_action in {"withheld", "missing"}:
        action = "vlm_preserved_geometry_withheld"
        authority = "composition_gestalt"
        limitations.extend(geometry_limitations)

    else:
        action = "vlm_preserved_unhandled_geometry_state"
        authority = "composition_gestalt"
        limitations.append(
            f"unhandled camera geometry state action={geometry_action!r} elevation={geometry_elevation!r}"
        )

    governed_gestalt = copy.deepcopy(gestalt)
    governed_camera = governed_gestalt.get("camera")
    if not isinstance(governed_camera, dict):
        governed_camera = {}
        governed_gestalt["camera"] = governed_camera
    governed_camera["elevation"] = final_elevation

    return {
        "schema_version": "composition-camera-governance-1.0",
        "source_camera": {
            "elevation": source_elevation,
            "pitch": source_pitch,
            "confidence": source_confidence,
        },
        "geometry_camera_support": {
            "action": geometry_action,
            "candidate_elevation": geometry_elevation,
            "confidence_band": geometry_band,
            "authority": geometry_authority,
            "reasons": geometry_reasons,
            "limitations": geometry_limitations,
            "body_axis_camera_position": diagnostic.get("body_axis_camera_position"),
            "vertical_depth_ordering": diagnostic.get("vertical_depth_ordering"),
            "dwpose_visibility_gate": diagnostic.get("dwpose_visibility_gate"),
        },
        "governed_camera": {
            "elevation": final_elevation,
            # v1 only governs elevation. Pitch remains exactly the composition
            # probe's claim until a separately calibrated pitch authority exists.
            "pitch": source_pitch,
            "source_confidence": source_confidence,
        },
        "action": action,
        "authority": authority,
        "reasons": reasons,
        "limitations": limitations,
        "governed_gestalt": governed_gestalt,
        "policy": {
            "qualified_low_geometry_may_override_vlm": True,
            "supporting_low_geometry_may_override_vlm": False,
            "positive_headward_geometry_may_qualify_high": False,
            "very_low_is_never_created_by_geometry": True,
            "pitch_is_not_governed": True,
        },
    }


def _matches(key: str, only: list[str]) -> bool:
    if not only:
        return True
    return any(fnmatch.fnmatch(key, pattern) or pattern in key for pattern in only)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-composition-camera-governor",
        description=(
            "Govern Composition Gestalt camera elevation with DWPose-gated SAM3D camera geometry. "
            "Only visibility-qualified LOW geometry may override the VLM."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--composition-dir", type=Path, required=True)
    parser.add_argument("--camera-diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    composition_dir = args.composition_dir.expanduser().resolve()
    camera_path = args.camera_diagnostic.expanduser().resolve()
    output_dir = (
        args.output_dir or (run_dir / "composition-camera-governed-v1")
    ).expanduser().resolve()

    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    if not composition_dir.is_dir():
        print(f"Composition directory not found: {composition_dir}", file=sys.stderr)
        return 2
    if not camera_path.is_file():
        print(f"Camera diagnostic not found: {camera_path}", file=sys.stderr)
        return 2

    diagnostic_run = _read_json(camera_path)
    diagnostic_by_key = {
        str(item.get("image_key")): item.get("diagnostic")
        for item in diagnostic_run.get("records") or []
        if isinstance(item, dict) and item.get("image_key")
    }

    source_paths = sorted(composition_dir.glob("*.composition_gestalt.json"))
    source_paths = [path for path in source_paths if _matches(path.name.removesuffix(".composition_gestalt.json"), args.only)]
    if not source_paths:
        print("No matching composition gestalt records found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}

    for source_path in source_paths:
        key = source_path.name.removesuffix(".composition_gestalt.json")
        out_path = output_dir / f"{key}.composition_camera_governed.json"
        if out_path.exists() and not args.overwrite:
            payload = _read_json(out_path)
            gov = payload.get("camera_governance") or {}
            status = "reused"
        else:
            source = _read_json(source_path)
            gov = govern_camera_elevation(source, diagnostic_by_key.get(key))
            payload = {
                "image_key": key,
                "source_composition": str(source_path),
                "source_camera_diagnostic": str(camera_path) if key in diagnostic_by_key else None,
                "source_schema_valid": source.get("schema_valid"),
                "camera_governance": gov,
            }
            _write_json(out_path, payload)
            status = "written"

        action = str(gov.get("action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        records.append({
            "image_key": key,
            "status": status,
            "source_elevation": (gov.get("source_camera") or {}).get("elevation"),
            "governed_elevation": (gov.get("governed_camera") or {}).get("elevation"),
            "pitch": (gov.get("governed_camera") or {}).get("pitch"),
            "action": action,
            "authority": gov.get("authority"),
        })
        print(
            f"{key}: {(gov.get('source_camera') or {}).get('elevation')} -> "
            f"{(gov.get('governed_camera') or {}).get('elevation')} "
            f"[{action}]"
        )

    index = {
        "schema_version": "composition-camera-governance-run-1.0",
        "run_dir": str(run_dir),
        "composition_dir": str(composition_dir),
        "camera_diagnostic": str(camera_path),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "action_counts": action_counts,
        "records": records,
    }
    _write_json(output_dir / "composition_camera_governance.index.json", index)
    print(f"Camera governance: {output_dir}")
    print("Actions: " + ", ".join(f"{key}={value}" for key, value in sorted(action_counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
