from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from . import caption_policy_identity_v01 as base

DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.14"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
EXPECTED_INPUT_SCHEMA = "caption-fact-sheet-0.2.14"
SCHEMA_VERSION = base.SCHEMA_VERSION
PROFILE = base.PROFILE


def main() -> int:
    args = base.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    if not input_dir.is_dir():
        print(f"Phase-4B.14 fact-sheet directory not found: {input_dir}", file=sys.stderr)
        return 2

    paths = base._input_files(input_dir, set(args.only))
    if not paths:
        print(f"No matching fact sheets found in {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for input_path in paths:
        key = input_path.name.removesuffix(".fact_sheet.json")
        out_path = output_dir / input_path.name
        if out_path.is_file() and not args.overwrite:
            record = base._read_json(out_path)
        else:
            source = base._read_json(input_path)
            if source.get("schema_version") != EXPECTED_INPUT_SCHEMA:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "image_key": key,
                    "error": f"expected {EXPECTED_INPUT_SCHEMA}, got {source.get('schema_version')}",
                }
            else:
                record = base.apply_character_identity_policy(source)
            base._write_json(out_path, record)
        records.append(record)
        print(f"{key}: {record.get('status')} | identity policy")

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "profile": PROFILE,
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "invariants": {
            "no_model_calls": True,
            "phase4b14_is_input_not_recomputed": True,
            "authoritative_broad_pose_adjudication_is_preserved": True,
            "authoritative_torso_relation_adjudication_is_preserved": True,
            "support_contact_truth_gate_is_preserved": True,
            "support_topology_truth_gate_is_preserved": True,
            "bilateral_knee_flexion_adjudication_is_preserved": True,
            "unilateral_raised_leg_adjudication_is_preserved": True,
            "crouched_stance_depth_adjudication_is_preserved": True,
            "canonical_pose_text_is_preserved": True,
            "signed_torso_turn_direction_is_preserved": True,
            "specialist_bound_anatomical_laterality_is_preserved": True,
            "appearance_observation_is_preserved": True,
            "persistent_identity_traits_are_withheld_from_composer": True,
            "transient_appearance_remains_captionable": True,
        },
        "records": records,
    }
    base._write_json(output_dir / "caption_policy.index.json", index)
    print(f"Index: {output_dir / 'caption_policy.index.json'}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
