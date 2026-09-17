from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "caption-fact-sheet-0.2.16"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.15"
DEFAULT_FRAMING_SUBDIR = Path("semantic-v3") / "framing-semantics-v1.0"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.16"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _apply_framing_authority(
    sheet: dict[str, Any],
    framing: dict[str, Any],
    *,
    framing_path: Path | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(sheet)
    key = str(out.get("image_key") or "")
    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    out["facts"] = facts

    previous = copy.deepcopy(facts.get("framing")) if isinstance(facts.get("framing"), dict) else None
    framing_ok = framing.get("status") in {"ok", "resolved_null"}
    same_key = not framing.get("image_key") or str(framing.get("image_key")) == key

    if not framing_ok or not same_key:
        audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
        violations = list(audit.get("violations") or [])
        violations.append(
            "production_framing_missing_or_mismatched"
            if same_key
            else "production_framing_image_key_mismatch"
        )
        audit["violations"] = sorted(set(str(v) for v in violations if v))
        audit["phase"] = "4B.16-authoritative"
        out["audit"] = audit
        out["schema_version"] = SCHEMA_VERSION
        out["status"] = "needs_review"
        return out

    span = copy.deepcopy(framing.get("anatomical_span") or {})
    scale = copy.deepcopy(framing.get("standard_shot_scale") or {})
    composer = copy.deepcopy(framing.get("composer_framing") or {})

    facts["framing"] = {
        "available": framing.get("status") == "ok",
        "source": "framing-semantics-v1.0",
        "authority": "deterministic_observation",
        "anatomical_span": span,
        "standard_shot_scale": scale,
        "composer_framing": composer,
        "face_scale_geometry": copy.deepcopy(framing.get("face_scale_geometry") or {}),
        "person_scale_geometry": copy.deepcopy(framing.get("person_scale_geometry") or {}),
        "face_person_height_ratio": framing.get("face_person_height_ratio"),
        "broad_pose_supported": bool(framing.get("broad_pose_supported")),
        "legacy": {
            "previous_fact": previous,
            "extent_hint": framing.get("legacy_extent_hint"),
            "caption_authoritative": False,
        },
        "note": (
            "Canonical framing is the observed pose-neutral anatomical span. "
            "Conventional shot scale is optional and caption-facing only when the "
            "validated framing specialist authorizes it."
        ),
    }

    sources = out.get("sources") if isinstance(out.get("sources"), dict) else {}
    out["sources"] = sources
    sources["framing_semantics"] = str(framing_path) if framing_path is not None else None

    reserved = out.get("reserved_domains") if isinstance(out.get("reserved_domains"), dict) else {}
    out["reserved_domains"] = reserved
    reserved["framing"] = {
        "owner": "framing-semantics-v1.0",
        "status": "resolved" if framing.get("status") == "ok" else "resolved_null",
    }

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.16-authoritative"
    audit["specialist_adjudication_phase"] = "4B.16-authoritative"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        canonical_framing_is_pose_neutral_anatomical_span=True,
        legacy_extent_bucket_is_not_caption_authoritative=True,
        qwen_cannot_create_framing_authority=True,
        sam3d_reconstruction_cannot_create_framing_authority=True,
        standard_shot_scale_is_optional=True,
        standard_shot_scale_does_not_override_anatomical_span=True,
        caption_surface_uses_scale_without_mechanical_span_repetition_when_scale_is_authorized=True,
        full_body_language_not_authorized_from_dwpose_ankles_alone=True,
        extreme_close_up_disabled_without_positive_calibration=True,
        prior_phase4b15_head_support_and_pose_specialists_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-4B.16 production framing authority promotion."
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--framing-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    framing_dir = args.framing_dir.expanduser().resolve() if args.framing_dir else run_dir / DEFAULT_FRAMING_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    if not run_dir.is_dir() or not input_dir.is_dir() or not framing_dir.is_dir():
        print(
            f"Required directory missing: run={run_dir} input={input_dir} framing={framing_dir}",
            file=sys.stderr,
        )
        return 2

    requested = set(args.only)
    paths = sorted(input_dir.glob("*.fact_sheet.json"))
    if requested:
        paths = [p for p in paths if p.name.removesuffix(".fact_sheet.json") in requested]
    if not paths:
        print("No matching Phase-4B.15 fact sheets found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for input_path in paths:
        key = input_path.name.removesuffix(".fact_sheet.json")
        framing_path = framing_dir / f"{key}.framing.json"
        out_path = output_dir / f"{key}.fact_sheet.json"

        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        elif not framing_path.is_file():
            record = copy.deepcopy(_read_json(input_path))
            record["schema_version"] = SCHEMA_VERSION
            record["status"] = "error"
            record["error"] = "missing_production_framing_semantics"
            _write_json(out_path, record)
        else:
            record = _apply_framing_authority(
                _read_json(input_path),
                _read_json(framing_path),
                framing_path=framing_path,
            )
            _write_json(out_path, record)

        records.append(record)
        framing_fact = ((record.get("facts") or {}).get("framing") or {})
        composer = framing_fact.get("composer_framing") if isinstance(framing_fact, dict) else {}
        print(
            f"{key}: {record.get('status')} "
            f"framing={composer.get('composer_text') if isinstance(composer, dict) else '-'}"
        )

    counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "framing_dir": str(framing_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "records": records,
    }
    _write_json(output_dir / "caption_fact_sheet.index.json", index)
    print(f"Index: {output_dir / 'caption_fact_sheet.index.json'}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
