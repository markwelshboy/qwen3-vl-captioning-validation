from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "caption-fact-sheet-0.2.17"
EXPECTED_INPUT_SCHEMA = "caption-fact-sheet-0.2.16"
EXPECTED_SELFIE_SCHEMA_PREFIX = "selfie-evidence-shadow-0.7"
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.16"
DEFAULT_SELFIE_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.7"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.17"


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


def _capture_fact_from_selfie(selfie: dict[str, Any]) -> dict[str, Any] | None:
    if selfie.get("status") != "ok":
        return None

    decision = selfie.get("decision") if isinstance(selfie.get("decision"), dict) else {}
    capture_style = (
        selfie.get("capture_style")
        if isinstance(selfie.get("capture_style"), dict)
        else {}
    )
    evidence = selfie.get("evidence") if isinstance(selfie.get("evidence"), dict) else {}
    mirror_policy = (
        selfie.get("mirror_caption_surface_policy")
        if isinstance(selfie.get("mirror_caption_surface_policy"), dict)
        else {}
    )

    if not decision.get("publishable_selfie"):
        return None

    subtype = _clean(capture_style.get("subtype")) or _clean(decision.get("capture_subtype"))
    if subtype not in {"direct_selfie", "mirror_selfie"}:
        return None

    promoted = [
        _clean(value)
        for value in (decision.get("promoted_fact_candidates") or [])
        if _clean(value)
    ]

    # Mirror v0.7 deliberately owns only the high-level capture semantic.
    # Direct-selfie facts may additionally expose validated camera/composition
    # facts already selected by the selfie specialist.
    if subtype == "mirror_selfie":
        composer_text = "mirror selfie"
        promoted = ["mirror selfie"]
        spatial_surface_mode = "depicted_frame"
    else:
        composer_text = "selfie-style capture"
        spatial_surface_mode = "direct_camera_frame"

    neutral = evidence.get("neutral_semantic") if isinstance(evidence.get("neutral_semantic"), dict) else {}
    mirror = (
        evidence.get("mirror_selfie_semantic")
        if isinstance(evidence.get("mirror_selfie_semantic"), dict)
        else {}
    )
    camera = evidence.get("camera_viewpoint") if isinstance(evidence.get("camera_viewpoint"), dict) else {}
    arm = evidence.get("foreground_arm") if isinstance(evidence.get("foreground_arm"), dict) else {}
    shoulder = evidence.get("nearer_shoulder") if isinstance(evidence.get("nearer_shoulder"), dict) else {}

    fact: dict[str, Any] = {
        "available": True,
        "family": "selfie",
        "subtype": subtype,
        "composer_text": composer_text,
        "spatial_surface_mode": spatial_surface_mode,
        "authority": _clean(capture_style.get("authority"))
        or _clean(decision.get("selfie_semantic_authority"))
        or _clean(decision.get("decision_rule_version"))
        or "selfie_evidence_shadow_v07",
        "promotion_status": "accepted_specialist_capture_semantic",
        "promoted_fact_candidates": promoted,
        "evidence_summary": {
            "neutral_semantic_grade": neutral.get("grade"),
            "mirror_semantic_grade": mirror.get("grade"),
            "camera_viewpoint_grade": camera.get("grade"),
            "foreground_arm_grade": arm.get("grade"),
            "nearer_shoulder_internal": shoulder.get("anatomical_side_nearer"),
        },
        "note": (
            "Capture style is a specialist-owned semantic. Mirror selfies retain "
            "anatomical geometry internally but use depicted-frame coordinates at "
            "the composer surface."
        ),
    }

    if subtype == "mirror_selfie":
        fact["mirror_surface_policy"] = copy.deepcopy(mirror_policy)
        fact["composer_spatial_contract"] = {
            "directional_reference_system": "frame_relative_only",
            "anatomical_laterality": "withhold",
            "physical_reflection_laterality": "withhold",
        }
    else:
        # Keep the specialist's already-promoted direct-selfie facts intact.
        # These are not re-derived here.
        fact["direct_capture_facts"] = promoted

    return fact


def _apply_selfie_capture_authority(
    sheet: dict[str, Any],
    selfie: dict[str, Any],
    *,
    selfie_path: Path | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(sheet)
    key = str(out.get("image_key") or "")

    if out.get("schema_version") != EXPECTED_INPUT_SCHEMA:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "image_key": key,
            "error": (
                f"expected {EXPECTED_INPUT_SCHEMA}, got {out.get('schema_version')}"
            ),
        }

    selfie_key = str(selfie.get("image_key") or "")
    schema = str(selfie.get("schema_version") or "")
    if selfie_key and selfie_key != key:
        out["schema_version"] = SCHEMA_VERSION
        out["status"] = "needs_review"
        out["error"] = "selfie_evidence_image_key_mismatch"
        return out
    if not schema.startswith(EXPECTED_SELFIE_SCHEMA_PREFIX):
        out["schema_version"] = SCHEMA_VERSION
        out["status"] = "needs_review"
        out["error"] = (
            f"expected {EXPECTED_SELFIE_SCHEMA_PREFIX}*, got {schema or 'missing'}"
        )
        return out

    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    out["facts"] = facts
    capture = _capture_fact_from_selfie(selfie)
    if capture:
        facts["capture"] = capture
    else:
        facts.pop("capture", None)

    sources = out.get("sources") if isinstance(out.get("sources"), dict) else {}
    out["sources"] = sources
    sources["selfie_evidence"] = str(selfie_path) if selfie_path is not None else None

    reserved = (
        out.get("reserved_domains")
        if isinstance(out.get("reserved_domains"), dict)
        else {}
    )
    out["reserved_domains"] = reserved
    reserved["capture_style"] = {
        "owner": "selfie-evidence-shadow-v0.7",
        "status": "resolved" if capture else "resolved_null",
    }

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    audit["phase"] = "4B.17-authoritative"
    audit["specialist_adjudication_phase"] = "4B.17-authoritative"
    invariants = (
        audit.get("invariants")
        if isinstance(audit.get("invariants"), dict)
        else {}
    )
    invariants.update(
        selfie_capture_is_specialist_owned=True,
        direct_selfie_rule_is_not_recomputed=True,
        mirror_selfie_rule_is_not_recomputed=True,
        mirror_selfie_uses_depicted_frame_surface=True,
        mirror_selfie_anatomical_laterality_remains_internal=True,
        mirror_selfie_phone_presence_alone_cannot_create_capture_authority=True,
        prior_phase4b16_framing_and_body_specialists_remain_authoritative=True,
    )
    audit["invariants"] = invariants
    audit["capture_style_promoted"] = bool(capture)
    audit["capture_subtype"] = capture.get("subtype") if capture else None
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase-4B.17 promotion of validated selfie capture semantics into the "
            "caption fact sheet."
        )
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--selfie-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir
        else run_dir / DEFAULT_INPUT_SUBDIR
    )
    selfie_dir = (
        args.selfie_dir.expanduser().resolve()
        if args.selfie_dir
        else run_dir / DEFAULT_SELFIE_SUBDIR
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / DEFAULT_OUTPUT_SUBDIR
    )

    if not run_dir.is_dir() or not input_dir.is_dir() or not selfie_dir.is_dir():
        print(
            f"Required directory missing: run={run_dir} input={input_dir} selfie={selfie_dir}",
            file=sys.stderr,
        )
        return 2

    requested = set(args.only)
    paths = sorted(input_dir.glob("*.fact_sheet.json"))
    if requested:
        paths = [
            p for p in paths
            if p.name.removesuffix(".fact_sheet.json") in requested
        ]
    if not paths:
        print("No matching Phase-4B.16 fact sheets found.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for input_path in paths:
        key = input_path.name.removesuffix(".fact_sheet.json")
        selfie_path = selfie_dir / f"{key}.selfie_evidence.json"
        out_path = output_dir / f"{key}.fact_sheet.json"

        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        elif not selfie_path.is_file():
            record = copy.deepcopy(_read_json(input_path))
            record["schema_version"] = SCHEMA_VERSION
            record["status"] = "error"
            record["error"] = "missing_selfie_evidence_v07"
            _write_json(out_path, record)
        else:
            record = _apply_selfie_capture_authority(
                _read_json(input_path),
                _read_json(selfie_path),
                selfie_path=selfie_path,
            )
            _write_json(out_path, record)

        records.append(record)
        capture = ((record.get("facts") or {}).get("capture") or {})
        print(
            f"{key}: {record.get('status')} "
            f"capture={capture.get('subtype') if isinstance(capture, dict) else '-'}"
        )

    counts = Counter(str(r.get("status") or "unknown") for r in records)
    subtype_counts = Counter(
        str((((r.get("facts") or {}).get("capture") or {}).get("subtype")) or "none")
        for r in records
        if r.get("status") in {"ok", "needs_review"}
    )
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "selfie_dir": str(selfie_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "capture_subtype_counts": dict(sorted(subtype_counts.items())),
        "records": records,
    }
    _write_json(output_dir / "caption_fact_sheet.index.json", index)
    print(f"Index: {output_dir / 'caption_fact_sheet.index.json'}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
