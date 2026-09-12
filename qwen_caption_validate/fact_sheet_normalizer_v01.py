from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.1"
DEFAULT_BODY_SUBDIR = Path("semantic-v3") / "fragment-probe-routed-v0.1"
DEFAULT_GESTALT_SUBDIR = Path("semantic-v3") / "routed-gestalt-v0.1"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.1"
SCHEMA_VERSION = "caption-fact-sheet-0.1"
VALID_MODES = {"framing_only", "configuration", "pose_allowed", "pose_guided"}
VISUAL_FIELDS = ("appearance", "objects", "scene", "secondary_people")
LATERALITY_RE = re.compile(r"\b(left|right)\b", re.IGNORECASE)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _candidate(text: str, *, source: str, domain: str, authority: str, promotion_status: str, note: str | None = None) -> dict[str, Any]:
    item = {
        "text": text,
        "source": source,
        "domain": domain,
        "authority": authority,
        "promotion_status": promotion_status,
    }
    if note:
        item["note"] = note
    return item


def _visual_candidate(text: str, domain: str) -> dict[str, Any]:
    if LATERALITY_RE.search(text):
        return _candidate(
            text,
            source="gestalt_qwen",
            domain=domain,
            authority="qwen_semantic_candidate",
            promotion_status="held_for_laterality_normalization",
            note="Contains left/right language; anatomical laterality is owned by DWPose in Phase 4B.",
        )
    return _candidate(
        text,
        source="gestalt_qwen",
        domain=domain,
        authority="qwen_semantic_candidate",
        promotion_status="accepted_candidate",
    )


def _normalize_visual(acquisition: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for field in VISUAL_FIELDS:
        values = acquisition.get(field)
        items: list[dict[str, Any]] = []
        if isinstance(values, list):
            for value in values:
                text = _clean_text(value)
                if text:
                    items.append(_visual_candidate(text, field))
        out[field] = items
    return out


def _normalize_context(acquisition: dict[str, Any]) -> dict[str, Any]:
    expression_action: list[dict[str, Any]] = []
    values = acquisition.get("expression_action")
    if isinstance(values, list):
        for value in values:
            text = _clean_text(value)
            if text:
                expression_action.append(_candidate(
                    text,
                    source="gestalt_qwen",
                    domain="expression_action",
                    authority="context_only",
                    promotion_status="held_for_domain_normalization",
                    note="Mixed field may contain expression, action, pose, gaze, or capture mechanics; no restricted-domain authority is granted in Phase 4A.",
                ))

    uncertainties: list[str] = []
    raw_uncertainties = acquisition.get("uncertainties")
    if isinstance(raw_uncertainties, list):
        for value in raw_uncertainties:
            text = _clean_text(value)
            if text:
                uncertainties.append(text)

    gestalt_text = _clean_text(acquisition.get("gestalt"))
    gestalt = None
    if gestalt_text:
        gestalt = _candidate(
            gestalt_text,
            source="gestalt_qwen",
            domain="gestalt",
            authority="context_only",
            promotion_status="not_authoritative",
            note="Preserved as holistic context only; wording here cannot create pose, gaze, framing, head-pose, or laterality facts.",
        )
    return {
        "expression_action": expression_action,
        "gestalt": gestalt,
        "uncertainties": uncertainties,
    }


def _normalize_body(body: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[str], list[str]]:
    violations: list[str] = []
    warnings: list[str] = []
    extraction = body.get("extraction") if isinstance(body.get("extraction"), dict) else {}
    pose_raw = extraction.get("pose_candidate") if isinstance(extraction, dict) else None
    rel_raw = extraction.get("body_relationships") if isinstance(extraction, dict) else None

    pose_candidate = None
    if isinstance(pose_raw, dict):
        text = _clean_text(pose_raw.get("text"))
        if text:
            if mode in {"pose_allowed", "pose_guided"}:
                pose_candidate = _candidate(
                    text,
                    source="routed_body_qwen",
                    domain="broad_pose",
                    authority="qwen_pose_hypothesis",
                    promotion_status="candidate",
                    note="Allowed by Phase-1 crop policy but remains a hypothesis until Phase 4B normalization.",
                )
            else:
                violations.append("broad_pose_candidate_present_in_non_pose_route")

    relationships: list[dict[str, Any]] = []
    if isinstance(rel_raw, list):
        for item in rel_raw:
            if not isinstance(item, dict):
                continue
            text = _clean_text(item.get("text"))
            if not text:
                continue
            if mode == "framing_only":
                violations.append("body_relationship_present_in_framing_only_route")
                continue
            relationships.append(_candidate(
                text,
                source="routed_body_qwen",
                domain="configuration",
                authority="route_scoped_candidate",
                promotion_status="pending_phase_4b",
                note="Preserved exactly, but laterality/head/geometry leakage is not authoritative until specialist normalization.",
            ))

    parse = body.get("parse") if isinstance(body.get("parse"), dict) else {}
    route_violations = parse.get("route_violations")
    if isinstance(route_violations, list):
        for value in route_violations:
            text = _clean_text(value)
            if text:
                warnings.append(f"phase2:{text}")

    return {
        "pose_candidate": pose_candidate,
        "configuration": relationships,
    }, violations, warnings


def _reserved_domains() -> dict[str, dict[str, str]]:
    return {
        "framing": {"owner": "deterministic_crop_geometry", "status": "pending_phase_4b"},
        "anatomical_laterality": {"owner": "dwpose", "status": "pending_phase_4b"},
        "torso_geometry": {"owner": "sam3d_plus_visibility_gate", "status": "pending_phase_4b"},
        "head_pose": {"owner": "pyfeat", "status": "pending_phase_4b"},
        "gaze": {"owner": "l2cs_when_uniface_observable", "status": "pending_phase_4b"},
    }


def _build_fact_sheet(policy: dict[str, Any], body: dict[str, Any], gestalt: dict[str, Any], *, policy_path: Path, body_path: Path, gestalt_path: Path) -> dict[str, Any]:
    key = str(policy.get("image_key") or body.get("image_key") or gestalt.get("image_key") or "")
    mode = str((policy.get("policy") or {}).get("mode") or "unknown")
    pose_relevance = policy.get("pose_relevance")
    violations: list[str] = []
    warnings: list[str] = []

    if mode not in VALID_MODES:
        violations.append("unknown_policy_mode")
    body_mode = body.get("policy_mode")
    gestalt_mode = gestalt.get("policy_mode")
    if body_mode and body_mode != mode:
        violations.append("body_policy_mode_mismatch")
    if gestalt_mode and gestalt_mode != mode:
        violations.append("gestalt_policy_mode_mismatch")
    if body.get("status") not in {"ok", "skipped_by_policy"}:
        warnings.append(f"body_status:{body.get('status')}")
    if gestalt.get("status") != "ok":
        warnings.append(f"gestalt_status:{gestalt.get('status')}")

    body_facts, body_violations, body_warnings = _normalize_body(body, mode)
    violations.extend(body_violations)
    warnings.extend(body_warnings)

    acquisition = gestalt.get("acquisition") if isinstance(gestalt.get("acquisition"), dict) else {}
    visual = _normalize_visual(acquisition)
    context_only = _normalize_context(acquisition)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if not violations else "needs_review",
        "image_key": key,
        "policy": {
            "mode": mode,
            "pose_relevance": pose_relevance,
        },
        "sources": {
            "perception_policy": str(policy_path),
            "routed_body": str(body_path),
            "routed_gestalt": str(gestalt_path),
        },
        "facts": {
            "body": body_facts,
            "visual": visual,
        },
        "context_only": context_only,
        "reserved_domains": _reserved_domains(),
        "audit": {
            "violations": sorted(set(violations)),
            "warnings": sorted(set(warnings)),
            "caption_ready": False,
            "phase": "4A",
        },
    }


def _policy_files(policy_dir: Path, only: set[str]) -> list[Path]:
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if only:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in only]
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-4A deterministic typed fact-sheet normalizer.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--policy-dir", type=Path)
    parser.add_argument("--body-dir", type=Path)
    parser.add_argument("--gestalt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    body_dir = args.body_dir.expanduser().resolve() if args.body_dir else run_dir / DEFAULT_BODY_SUBDIR
    gestalt_dir = args.gestalt_dir.expanduser().resolve() if args.gestalt_dir else run_dir / DEFAULT_GESTALT_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    for path in (policy_dir, body_dir, gestalt_dir):
        if not path.is_dir():
            print(f"Required directory not found: {path}", file=sys.stderr)
            return 2

    requested = set(args.only)
    policy_paths = _policy_files(policy_dir, requested)
    if not policy_paths:
        print(f"No matching perception-policy records found in {policy_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for policy_path in policy_paths:
        policy = _read_json(policy_path)
        key = str(policy.get("image_key") or policy_path.name.removesuffix(".perception_policy.json"))
        body_path = body_dir / f"{key}.routed_fragments.json"
        gestalt_path = gestalt_dir / f"{key}.gestalt.json"
        out_path = output_dir / f"{key}.fact_sheet.json"
        if out_path.is_file() and not args.overwrite:
            records.append(_read_json(out_path))
            continue
        if not body_path.is_file() or not gestalt_path.is_file():
            missing = []
            if not body_path.is_file():
                missing.append("routed_body")
            if not gestalt_path.is_file():
                missing.append("routed_gestalt")
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "error": "missing_inputs:" + ",".join(missing),
            }
        else:
            body = _read_json(body_path)
            gestalt = _read_json(gestalt_path)
            record = _build_fact_sheet(
                policy,
                body,
                gestalt,
                policy_path=policy_path,
                body_path=body_path,
                gestalt_path=gestalt_path,
            )
        _write_json(out_path, record)
        records.append(record)
        print(f"{key}: {record.get('status')}")

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "body_dir": str(body_dir),
        "gestalt_dir": str(gestalt_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "invariants": {
            "no_model_calls": True,
            "gestalt_wording_cannot_create_restricted_domain_authority": True,
            "framing_only_cannot_acquire_pose_from_gestalt": True,
            "configuration_cannot_acquire_broad_pose_from_gestalt": True,
            "expression_action_is_context_only_until_domain_normalization": True,
            "left_right_visual_semantics_are_held_for_dwpose_normalization": True,
            "caption_ready": False,
        },
        "records": records,
    }
    index_path = output_dir / "caption_fact_sheet.index.json"
    _write_json(index_path, index)
    print(f"Index: {index_path}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
