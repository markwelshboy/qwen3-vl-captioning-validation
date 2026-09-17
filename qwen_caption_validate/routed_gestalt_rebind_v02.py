from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_SUBDIR = Path("semantic-v3") / "routed-gestalt-v0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_BODY_SUBDIR = Path("semantic-v3") / "fragment-probe-routed-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "routed-gestalt-v0.2"
SCHEMA_VERSION = "routed-gestalt-0.2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _policy_files(policy_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if only:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in only]
    return paths


def _body_reference(body_path: Path) -> dict[str, Any] | None:
    if not body_path.is_file():
        return None
    body = _read_json(body_path)
    return {
        "path": str(body_path),
        "status": body.get("status"),
        "policy_mode": body.get("policy_mode"),
        "model_call": body.get("model_call"),
    }


def _rebind_record(
    source: dict[str, Any],
    policy: dict[str, Any],
    *,
    source_path: Path,
    policy_path: Path,
    body_path: Path,
) -> dict[str, Any]:
    """Rebind route-independent Phase-3 semantics to the promoted routing metadata.

    The v0.1 gestalt prompt is identical for every policy mode and does not receive
    pose/configuration/framing/head/gaze geometry.  Therefore its model acquisition
    can be reused exactly while its route/provenance metadata is advanced to v0.2.
    """
    out = copy.deepcopy(source)
    key = str(policy.get("image_key") or source.get("image_key") or source_path.name.removesuffix(".gestalt.json"))
    new_mode = str((policy.get("policy") or {}).get("mode") or "unknown")
    old_mode = str(source.get("policy_mode") or "unknown")

    out["schema_version"] = SCHEMA_VERSION
    out["image_key"] = key
    out["policy_mode"] = new_mode
    out["policy_source"] = str(policy_path)
    out["body_acquisition"] = _body_reference(body_path)
    out["route_rebind"] = {
        "source_gestalt": str(source_path),
        "source_schema_version": source.get("schema_version"),
        "source_policy_mode": old_mode,
        "target_policy_mode": new_mode,
        "acquisition_reused_without_model_call": True,
        "raw_response_reused_without_model_call": True,
        "reason": (
            "routed_gestalt_v01_uses_the_same_image_only_prompt_for_all_policy_modes_"
            "and_pose_configuration_framing_head_gaze_laterality_are_owned_elsewhere"
        ),
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministically rebind route-independent gestalt v0.1 acquisitions to production routing v0.2."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--body-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve() if args.source_dir else run_dir / DEFAULT_SOURCE_SUBDIR
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    body_dir = args.body_dir.expanduser().resolve() if args.body_dir else run_dir / DEFAULT_BODY_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    for label, path in (("run", run_dir), ("source gestalt", source_dir), ("policy", policy_dir), ("body", body_dir)):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    policy_paths = _policy_files(policy_dir, set(args.only))
    if not policy_paths:
        print(f"No matching perception-policy records found in {policy_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for policy_path in policy_paths:
        policy = _read_json(policy_path)
        key = str(policy.get("image_key") or policy_path.name.removesuffix(".perception_policy.json"))
        source_path = source_dir / f"{key}.gestalt.json"
        body_path = body_dir / f"{key}.routed_fragments.json"
        out_path = output_dir / f"{key}.gestalt.json"

        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        elif not source_path.is_file():
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "error": "missing_source_gestalt_v0.1",
                "policy_source": str(policy_path),
            }
            _write_json(out_path, record)
        elif not body_path.is_file():
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "error": "missing_fragment_probe_routed_v0.2",
                "policy_source": str(policy_path),
                "source_gestalt": str(source_path),
            }
            _write_json(out_path, record)
        else:
            source = _read_json(source_path)
            if source.get("status") != "ok":
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "image_key": key,
                    "error": f"source_gestalt_not_ok:{source.get('status')}",
                    "source_gestalt": str(source_path),
                    "policy_source": str(policy_path),
                }
            else:
                record = _rebind_record(
                    source,
                    policy,
                    source_path=source_path,
                    policy_path=policy_path,
                    body_path=body_path,
                )
            _write_json(out_path, record)

        records.append(record)
        print(
            f"{key}: {record.get('status')} | "
            f"{((record.get('route_rebind') or {}).get('source_policy_mode') or '-')} -> "
            f"{record.get('policy_mode') or '-'}"
        )

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    counts = Counter(str(r.get("status") or "unknown") for r in records)
    mode_counts = Counter(str(r.get("policy_mode") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "source_dir": str(source_dir),
        "policy_dir": str(policy_dir),
        "body_dir": str(body_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "invariants": {
            "no_model_calls": True,
            "v01_model_acquisition_is_reused_exactly": True,
            "same_gestalt_prompt_for_all_policy_modes": True,
            "gestalt_prompt_has_no_pose_geometry_input": True,
            "pose_configuration_framing_head_gaze_laterality_owned_elsewhere": True,
            "policy_and_body_provenance_rebound_to_v0.2": True,
        },
        "records": records,
    }
    _write_json(output_dir / "routed_gestalt.index.json", index)
    print(f"Index: {output_dir / 'routed_gestalt.index.json'}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
