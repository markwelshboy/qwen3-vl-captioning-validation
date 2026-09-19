from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "crop-consistency-diagnostic-0.1"
REGISTRY_SCHEMA_VERSION = "crop-family-registry-0.1"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = PACKAGE_ROOT / "notes" / "semantic_v3_blind_crop_family_registry_v01.json"
DEFAULT_COMPOSER_SUBDIR = Path("semantic-v3") / "text-composer-v0.14"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "crop-consistency-v0.1"

SHOT_SCALE_SCORE = {
    "close_up": 1.0,
    "medium_close_up": 2.0,
    "medium": 3.0,
    "medium_wide": 4.0,
    "wide": 5.0,
    "full": 5.0,
    "full_body": 5.0,
}
BODY_PART_SCORE = {
    "head": 0.0,
    "shoulders": 1.0,
    "chest": 1.5,
    "waist": 2.0,
    "hips": 2.5,
    "knees": 3.5,
    "ankles": 4.5,
    "feet": 5.0,
}
POSE_PATTERNS = {
    "standing": re.compile(r"\b(?:stand|stands|standing)\b", re.I),
    "seated": re.compile(r"\b(?:seated|sit|sits|sitting)\b", re.I),
    "lying": re.compile(r"\b(?:lie|lies|lying|reclined|reclining)\b", re.I),
    "crouching": re.compile(r"\b(?:crouch|crouches|crouched|crouching)\b", re.I),
    "kneeling": re.compile(r"\b(?:kneel|kneels|kneeled|kneeling)\b", re.I),
    "squatting": re.compile(r"\b(?:squat|squats|squatted|squatting)\b", re.I),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _pose_group(value: Any) -> str | None:
    text = _clean(value)
    if not text:
        return None
    for name, pattern in POSE_PATTERNS.items():
        if pattern.search(text):
            return name
    return text.lower()


def _framing_score(framing: dict[str, Any]) -> float | None:
    if not isinstance(framing, dict):
        return None
    label = _clean(framing.get("shot_scale_label"))
    if label in SHOT_SCALE_SCORE:
        return SHOT_SCALE_SCORE[label]
    span = framing.get("anatomical_span") if isinstance(framing.get("anatomical_span"), dict) else {}
    upper = _clean(span.get("upper"))
    lower = _clean(span.get("lower"))
    if lower in BODY_PART_SCORE:
        lower_score = BODY_PART_SCORE[lower]
        upper_penalty = 0.0 if upper == "head" else 0.25 if upper == "shoulders" else 0.15
        return max(0.5, lower_score - upper_penalty)
    return None


def _torso_summary(body: dict[str, Any]) -> dict[str, Any]:
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    preferred = torso.get("preferred") if isinstance(torso.get("preferred"), dict) else torso
    return {
        "camera_orientation": _clean(preferred.get("camera_orientation")),
        "approx_yaw_deg": preferred.get("approx_yaw_deg"),
        "turn_direction": _clean(preferred.get("turn_direction")),
    }


def snapshot_record(record: dict[str, Any]) -> dict[str, Any]:
    projection = record.get("evidence_projection") if isinstance(record.get("evidence_projection"), dict) else {}
    auth = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = auth.get("body") if isinstance(auth.get("body"), dict) else {}
    framing = auth.get("framing") if isinstance(auth.get("framing"), dict) else {}
    head = auth.get("head") if isinstance(auth.get("head"), dict) else {}
    gaze = auth.get("gaze") if isinstance(auth.get("gaze"), dict) else {}
    capture = auth.get("capture") if isinstance(auth.get("capture"), dict) else {}
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    return {
        "image_key": str(record.get("image_key") or ""),
        "status": record.get("status"),
        "framing": {
            "composer_text": _clean(framing.get("composer_text")),
            "surface_source": _clean(framing.get("surface_source")),
            "shot_scale_label": _clean(framing.get("shot_scale_label")),
            "anatomical_span": framing.get("anatomical_span") if isinstance(framing.get("anatomical_span"), dict) else None,
            "coverage_score": _framing_score(framing),
        },
        "body": {
            "broad_pose": _clean(body.get("broad_pose")),
            "broad_pose_group": _pose_group(body.get("broad_pose")),
            "canonical_broad_pose": _clean(body.get("canonical_broad_pose")),
            "canonical_broad_pose_group": _pose_group(body.get("canonical_broad_pose")),
            "configuration": [str(x) for x in configuration if isinstance(x, str)],
            "torso": _torso_summary(body),
        },
        "head": {
            "horizontal": _clean(head.get("horizontal")),
            "vertical": _clean(head.get("vertical")),
            "turn_strength": _clean(head.get("turn_strength")),
            "composer_text": _clean(head.get("composer_text")),
        },
        "gaze": {
            "horizontal": _clean(gaze.get("horizontal")),
            "vertical": _clean(gaze.get("vertical")),
            "camera_relationship": _clean(gaze.get("camera_relationship")),
        },
        "capture": {
            "subtype": _clean(capture.get("subtype")),
            "composer_text": _clean(capture.get("composer_text")),
        },
        "appearance": [str(x) for x in auth.get("appearance", []) if isinstance(x, str)],
        "objects": [str(x) for x in auth.get("objects", []) if isinstance(x, str)],
        "scene": [str(x) for x in auth.get("scene", []) if isinstance(x, str)],
        "caption": _clean(record.get("caption")),
        "audit_violations": list((record.get("caption_audit") or {}).get("violations") or []),
    }


def _value_transition(name: str, wider: Any, tighter: Any, *, contradiction: bool = True) -> dict[str, Any]:
    if wider == tighter:
        state = "stable" if wider is not None else "absent"
    elif wider is not None and tighter is None:
        state = "collapsed"
    elif wider is None and tighter is not None:
        state = "emerged_in_tighter_crop"
    else:
        state = "changed"
    return {
        "domain": name,
        "wider": wider,
        "tighter": tighter,
        "state": state,
        "contradiction": bool(contradiction and state == "changed" and wider is not None and tighter is not None),
    }


def _list_transition(name: str, wider: list[str], tighter: list[str]) -> dict[str, Any]:
    w = set(wider)
    t = set(tighter)
    return {
        "domain": name,
        "retained": sorted(w & t),
        "dropped": sorted(w - t),
        "emerged_in_tighter_crop": sorted(t - w),
    }


def compare_snapshots(wider: dict[str, Any], tighter: dict[str, Any]) -> dict[str, Any]:
    transitions = [
        _value_transition("broad_pose_group", wider["body"]["broad_pose_group"], tighter["body"]["broad_pose_group"]),
        _value_transition("canonical_broad_pose_group", wider["body"]["canonical_broad_pose_group"], tighter["body"]["canonical_broad_pose_group"]),
        _value_transition("torso.turn_direction", wider["body"]["torso"]["turn_direction"], tighter["body"]["torso"]["turn_direction"]),
        _value_transition("head.horizontal", wider["head"]["horizontal"], tighter["head"]["horizontal"]),
        _value_transition("head.vertical", wider["head"]["vertical"], tighter["head"]["vertical"]),
        _value_transition("gaze.horizontal", wider["gaze"]["horizontal"], tighter["gaze"]["horizontal"]),
        _value_transition("gaze.vertical", wider["gaze"]["vertical"], tighter["gaze"]["vertical"]),
        _value_transition("gaze.camera_relationship", wider["gaze"]["camera_relationship"], tighter["gaze"]["camera_relationship"]),
        _value_transition("capture.subtype", wider["capture"]["subtype"], tighter["capture"]["subtype"]),
    ]
    contradictions = [t for t in transitions if t["contradiction"]]
    expansions = [
        t for t in transitions
        if t["state"] == "emerged_in_tighter_crop" and t["domain"] in {"broad_pose_group", "canonical_broad_pose_group"}
    ]
    return {
        "wider": wider["image_key"],
        "tighter": tighter["image_key"],
        "wider_framing": wider["framing"],
        "tighter_framing": tighter["framing"],
        "semantic_transitions": transitions,
        "list_transitions": [
            _list_transition("body.configuration", wider["body"]["configuration"], tighter["body"]["configuration"]),
            _list_transition("appearance", wider["appearance"], tighter["appearance"]),
            _list_transition("objects", wider["objects"], tighter["objects"]),
            _list_transition("scene", wider["scene"], tighter["scene"]),
        ],
        "contradictions": contradictions,
        "authority_expansions_in_tighter_crop": expansions,
    }


def _sort_members(snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        snapshots,
        key=lambda s: (
            -(s["framing"].get("coverage_score") if s["framing"].get("coverage_score") is not None else -1),
            s["image_key"],
        ),
    )


def audit_family(family: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    members = [str(x) for x in family.get("members") or []]
    missing = [x for x in members if x not in records]
    snapshots = [snapshot_record(records[x]) for x in members if x in records]
    by_key = {s["image_key"]: s for s in snapshots}
    declared_order = [str(x) for x in family.get("wide_to_tight") or []]
    if declared_order:
        ordered = [by_key[x] for x in declared_order if x in by_key]
        ordered += [s for s in _sort_members(snapshots) if s["image_key"] not in declared_order]
    else:
        ordered = _sort_members(snapshots)

    relative_fov = family.get("relative_fov_area") if isinstance(family.get("relative_fov_area"), dict) else {}
    for snapshot in ordered:
        value = relative_fov.get(snapshot["image_key"])
        snapshot["source_relative_fov_area"] = value if isinstance(value, (int, float)) else None

    exact_groups = [set(map(str, group)) for group in (family.get("exact_duplicate_groups") or [])]
    pairs: list[dict[str, Any]] = []
    for wider, tighter in zip(ordered, ordered[1:]):
        pair = compare_snapshots(wider, tighter)
        key_pair = {wider["image_key"], tighter["image_key"]}
        pair["source_crop_relation"] = (
            "exact_duplicate" if any(key_pair.issubset(group) for group in exact_groups)
            else "wider_to_tighter"
        )
        wfov = wider.get("source_relative_fov_area")
        tfov = tighter.get("source_relative_fov_area")
        pair["source_fov_ratio"] = (
            round(float(wfov) / float(tfov), 3)
            if isinstance(wfov, (int, float)) and isinstance(tfov, (int, float)) and tfov
            else None
        )
        pair["framing_changed"] = wider["framing"].get("composer_text") != tighter["framing"].get("composer_text")
        pairs.append(pair)

    exact_duplicate_mismatches: list[dict[str, Any]] = []
    for group in family.get("exact_duplicate_groups") or []:
        present = [str(x) for x in group if str(x) in records]
        if len(present) < 2:
            continue
        base_key = present[0]
        base = (records[base_key].get("evidence_projection") or {}).get("authoritative_facts") or {}
        for other_key in present[1:]:
            other = (records[other_key].get("evidence_projection") or {}).get("authoritative_facts") or {}
            if base != other:
                exact_duplicate_mismatches.append({"a": base_key, "b": other_key})

    contradictions = [
        {"pair": [p["wider"], p["tighter"]], **t}
        for p in pairs for t in p["contradictions"]
    ]
    expansions = [
        {"pair": [p["wider"], p["tighter"]], **t}
        for p in pairs for t in p["authority_expansions_in_tighter_crop"]
    ]
    if missing:
        status = "missing_records"
    elif exact_duplicate_mismatches or contradictions:
        status = "review"
    elif expansions:
        status = "review"
    else:
        status = "consistent"
    return {
        "family_id": family.get("family_id"),
        "kind": family.get("kind"),
        "members": members,
        "missing_records": missing,
        "ordered_wide_to_tight": [s["image_key"] for s in ordered],
        "member_snapshots": ordered,
        "pairwise_crop_transitions": pairs,
        "contradictions": contradictions,
        "authority_expansions_in_tighter_crop": expansions,
        "exact_duplicate_projection_mismatches": exact_duplicate_mismatches,
        "status": status,
    }


def _markdown(index: dict[str, Any]) -> str:
    lines = [
        "# Crop Consistency Diagnostic v0.1",
        "",
        "Crop defines the truth envelope: wider crops may authorize global pose/support/scene facts; tighter crops should preserve surviving visible facts while allowing out-of-frame semantics to collapse. A tighter crop should not invent a conflicting fact simply because the wider source photograph proves it exists.",
        "",
        f"- families: {index['family_count']}",
        f"- multi-member crop families: {index['multi_member_family_count']}",
        f"- records represented: {index['record_count']}",
        f"- status counts: {json.dumps(index['status_counts'], sort_keys=True)}",
        "",
    ]
    for family in index["families"]:
        if len(family["members"]) < 2:
            continue
        lines += [
            f"## {family['family_id']} — {family['status']}",
            "",
            "Wide → tight: " + " → ".join(family["ordered_wide_to_tight"]),
            "",
        ]
        for pair in family["pairwise_crop_transitions"]:
            interesting = [
                t for t in pair["semantic_transitions"]
                if t["state"] not in {"stable", "absent"}
            ]
            lines.append(f"### {pair['wider']} → {pair['tighter']}")
            lines.append("")
            for t in interesting:
                mark = " **CONTRADICTION**" if t["contradiction"] else ""
                lines.append(f"- {t['domain']}: {t['state']} — '{t['wider']}' → '{t['tighter']}'{mark}")
            for lt in pair["list_transitions"]:
                if lt["dropped"] or lt["emerged_in_tighter_crop"]:
                    lines.append(
                        f"- {lt['domain']}: dropped={lt['dropped'] or []}; emerged={lt['emerged_in_tighter_crop'] or []}"
                    )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run(
    run_dir: Path,
    registry_path: Path,
    composer_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    registry = _read_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema: {registry.get('schema_version')}")
    families = registry.get("families") or []
    wanted_keys = {str(k) for fam in families for k in (fam.get("members") or [])}
    records: dict[str, dict[str, Any]] = {}
    for key in sorted(wanted_keys):
        path = composer_dir / f"{key}.composed.json"
        if path.is_file():
            records[key] = _read_json(path)

    out_json = output_dir / "crop_consistency.index.json"
    out_md = output_dir / "crop_consistency.report.md"
    if not overwrite and (out_json.exists() or out_md.exists()):
        raise FileExistsError(f"output exists; use --overwrite: {output_dir}")

    audited = [audit_family(fam, records) for fam in families]
    counts = Counter(x["status"] for x in audited)
    index = {
        "schema_version": SCHEMA_VERSION,
        "registry": str(registry_path),
        "run_dir": str(run_dir),
        "composer_dir": str(composer_dir),
        "record_count": len(wanted_keys),
        "loaded_record_count": len(records),
        "family_count": len(audited),
        "multi_member_family_count": sum(len(x["members"]) > 1 for x in audited),
        "status_counts": dict(sorted(counts.items())),
        "principle": (
            "crop defines the truth envelope; semantics may collapse as evidence leaves frame "
            "but surviving facts should not contradict the same source photograph"
        ),
        "families": audited,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_json, index)
    out_md.write_text(_markdown(index), encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare same-source crop families across the final composer projection. "
            "No model calls and no fact-sheet mutation."
        )
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--composer-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    registry = args.registry.expanduser().resolve()
    if not registry.is_file():
        print(f"Registry not found: {registry}", file=sys.stderr)
        return 2
    composer_dir = (
        args.composer_dir.expanduser().resolve()
        if args.composer_dir
        else run_dir / DEFAULT_COMPOSER_SUBDIR
    )
    if not composer_dir.is_dir():
        print(f"Composer directory not found: {composer_dir}", file=sys.stderr)
        return 2
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / DEFAULT_OUTPUT_SUBDIR
    )
    try:
        index = run(run_dir, registry, composer_dir, output_dir, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "Crop consistency: "
        f"families={index['family_count']} "
        f"multi={index['multi_member_family_count']} "
        f"records={index['loaded_record_count']}/{index['record_count']}"
    )
    print(f"Status counts: {json.dumps(index['status_counts'], sort_keys=True)}")
    print(f"JSON: {output_dir / 'crop_consistency.index.json'}")
    print(f"Report: {output_dir / 'crop_consistency.report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
