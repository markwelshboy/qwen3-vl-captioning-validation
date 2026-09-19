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
DEFAULT_FACT_SHEET_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3.2"
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
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    label = _clean(framing.get("shot_scale_label")) if isinstance(framing, dict) else None
    if label in SHOT_SCALE_SCORE:
        return SHOT_SCALE_SCORE[label]
    span = (
        framing.get("anatomical_span")
        if isinstance(framing, dict)
        and isinstance(framing.get("anatomical_span"), dict)
        else {}
    )
    lower = _clean(span.get("lower"))
    upper = _clean(span.get("upper"))
    if lower in BODY_PART_SCORE:
        penalty = (
            0.0
            if upper == "head"
            else 0.25
            if upper == "shoulders"
            else 0.15
        )
        return max(0.5, BODY_PART_SCORE[lower] - penalty)
    return None


def _empty_snapshot(image_key: str) -> dict[str, Any]:
    return {
        "image_key": image_key,
        "status": None,
        "framing": {
            "composer_text": None,
            "surface_source": None,
            "shot_scale_label": None,
            "anatomical_span": None,
            "coverage_score": None,
        },
        "body": {
            "broad_pose": None,
            "broad_pose_group": None,
            "canonical_broad_pose": None,
            "canonical_broad_pose_group": None,
            "configuration": [],
            "torso": {
                "camera_orientation": None,
                "approx_yaw_deg": None,
                "turn_direction": None,
            },
        },
        "head": {
            "horizontal": None,
            "vertical": None,
            "turn_strength": None,
            "composer_text": None,
        },
        "gaze": {
            "horizontal": None,
            "vertical": None,
            "camera_relationship": None,
        },
        "capture": {
            "subtype": None,
            "composer_text": None,
        },
        "appearance": [],
        "objects": [],
        "scene": [],
    }


def _visual_texts(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        text
        for item in values
        if isinstance(item, dict)
        and (text := _clean(item.get("composer_text")))
    ]


def _fact_sheet_broad_pose(
    body: dict[str, Any],
    framing: dict[str, Any],
) -> tuple[str | None, str | None]:
    candidate = (
        body.get("pose_candidate")
        if isinstance(body.get("pose_candidate"), dict)
        else {}
    )
    adjud = (
        body.get("broad_pose_adjudication")
        if isinstance(body.get("broad_pose_adjudication"), dict)
        else {}
    )
    status = _clean(adjud.get("status"))
    supported = bool(framing.get("broad_pose_supported"))
    broad = None
    if (
        status not in {"abstain", "rejected", "withheld"}
        and (
            supported
            or status in {"protected", "confirmed", "repaired"}
        )
    ):
        broad = (
            _clean(candidate.get("composer_text"))
            or _clean(adjud.get("canonical_pose_text"))
            or _clean(candidate.get("text"))
            or _clean(adjud.get("qwen_pose_text"))
        )
    canonical = _clean(adjud.get("canonical_pose_text"))
    return broad, canonical


def snapshot_fact_sheet(sheet: dict[str, Any]) -> dict[str, Any]:
    key = str(sheet.get("image_key") or "")
    out = _empty_snapshot(key)
    out["status"] = sheet.get("status")

    facts = (
        sheet.get("facts")
        if isinstance(sheet.get("facts"), dict)
        else {}
    )
    framing = (
        facts.get("framing")
        if isinstance(facts.get("framing"), dict)
        else {}
    )
    composer_framing = (
        framing.get("composer_framing")
        if isinstance(framing.get("composer_framing"), dict)
        else {}
    )
    anatomical = (
        framing.get("anatomical_span")
        if isinstance(framing.get("anatomical_span"), dict)
        else {}
    )
    standard = (
        framing.get("standard_shot_scale")
        if isinstance(framing.get("standard_shot_scale"), dict)
        else {}
    )
    out["framing"] = {
        "composer_text": _clean(composer_framing.get("composer_text")),
        "surface_source": _clean(composer_framing.get("source")),
        "shot_scale_label": (
            _clean(standard.get("label"))
            if standard.get("status") == "available"
            else None
        ),
        "anatomical_span": (
            {
                "upper": _clean(anatomical.get("upper_anchor")),
                "lower": _clean(anatomical.get("lower_anchor")),
            }
            if anatomical.get("status") == "available"
            else None
        ),
    }
    out["framing"]["coverage_score"] = _framing_score(out["framing"])

    body = (
        facts.get("body")
        if isinstance(facts.get("body"), dict)
        else {}
    )
    broad, canonical = _fact_sheet_broad_pose(body, framing)
    config = [
        text
        for item in body.get("configuration", [])
        if isinstance(item, dict)
        and (text := _clean(item.get("composer_text")))
    ]
    torso = (
        body.get("torso_geometry")
        if isinstance(body.get("torso_geometry"), dict)
        else {}
    )
    orient = (
        torso.get("caption_orientation")
        if isinstance(torso.get("caption_orientation"), dict)
        else {}
    )
    torso_ok = bool(torso.get("composer_eligible"))
    out["body"] = {
        "broad_pose": broad,
        "broad_pose_group": _pose_group(broad),
        "canonical_broad_pose": canonical,
        "canonical_broad_pose_group": _pose_group(canonical),
        "configuration": config,
        "torso": {
            "camera_orientation": (
                _clean(orient.get("preferred_orientation_band"))
                if torso_ok
                else None
            ),
            "approx_yaw_deg": (
                orient.get("preferred_approx_yaw_deg")
                if torso_ok
                else None
            ),
            "turn_direction": (
                _clean(orient.get("preferred_turn_direction"))
                if torso_ok
                and orient.get("turn_direction_publishable")
                else None
            ),
        },
    }

    head = (
        facts.get("head_pose")
        if isinstance(facts.get("head_pose"), dict)
        else {}
    )
    horizontal = (
        head.get("horizontal")
        if isinstance(head.get("horizontal"), dict)
        else {}
    )
    vertical = (
        head.get("vertical")
        if isinstance(head.get("vertical"), dict)
        else {}
    )
    head_h = (
        _clean(horizontal.get("value"))
        if horizontal.get("publishable")
        else None
    )
    head_v = (
        _clean(vertical.get("value"))
        if vertical.get("publishable")
        else None
    )
    strength = (
        _clean(head.get("yaw_strength"))
        if head.get("yaw_strength_publishable")
        else None
    )
    parts: list[str] = []
    if head_h:
        parts.append(f"head {strength or 'turned'} toward {head_h}")
    if head_v:
        parts.append(f"tilted {head_v}")
    out["head"] = {
        "horizontal": head_h,
        "vertical": head_v,
        "turn_strength": strength,
        "composer_text": " and ".join(parts) or None,
    }

    gaze = (
        facts.get("gaze")
        if isinstance(facts.get("gaze"), dict)
        else {}
    )
    semantics = (
        gaze.get("caption_semantics")
        if isinstance(gaze.get("caption_semantics"), dict)
        else {}
    )

    def gaze_value(name: str) -> str | None:
        item = (
            semantics.get(name)
            if isinstance(semantics.get(name), dict)
            else {}
        )
        return (
            _clean(item.get("composer_value"))
            if item.get("publishable")
            else None
        )

    out["gaze"] = {
        "horizontal": gaze_value("horizontal"),
        "vertical": gaze_value("vertical"),
        "camera_relationship": gaze_value("camera_relationship"),
    }

    capture = (
        facts.get("capture")
        if isinstance(facts.get("capture"), dict)
        else {}
    )
    if capture.get("available"):
        out["capture"] = {
            "subtype": _clean(capture.get("subtype")),
            "composer_text": _clean(capture.get("composer_text")),
        }

    visual = (
        facts.get("visual")
        if isinstance(facts.get("visual"), dict)
        else {}
    )
    out["appearance"] = _visual_texts(visual.get("appearance"))
    out["objects"] = _visual_texts(visual.get("objects"))
    out["scene"] = _visual_texts(visual.get("scene"))
    return out


def snapshot_projection(record: dict[str, Any]) -> dict[str, Any]:
    key = str(record.get("image_key") or "")
    out = _empty_snapshot(key)
    out["status"] = record.get("status")

    projection = (
        record.get("evidence_projection")
        if isinstance(record.get("evidence_projection"), dict)
        else {}
    )
    auth = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    body = (
        auth.get("body")
        if isinstance(auth.get("body"), dict)
        else {}
    )
    framing = (
        auth.get("framing")
        if isinstance(auth.get("framing"), dict)
        else {}
    )
    head = (
        auth.get("head")
        if isinstance(auth.get("head"), dict)
        else {}
    )
    gaze = (
        auth.get("gaze")
        if isinstance(auth.get("gaze"), dict)
        else {}
    )
    capture = (
        auth.get("capture")
        if isinstance(auth.get("capture"), dict)
        else {}
    )
    torso = (
        body.get("torso_orientation")
        if isinstance(body.get("torso_orientation"), dict)
        else {}
    )
    preferred = (
        torso.get("preferred")
        if isinstance(torso.get("preferred"), dict)
        else torso
    )

    out["framing"] = {
        "composer_text": _clean(framing.get("composer_text")),
        "surface_source": _clean(framing.get("surface_source")),
        "shot_scale_label": _clean(framing.get("shot_scale_label")),
        "anatomical_span": (
            framing.get("anatomical_span")
            if isinstance(framing.get("anatomical_span"), dict)
            else None
        ),
    }
    out["framing"]["coverage_score"] = _framing_score(out["framing"])

    broad = _clean(body.get("broad_pose"))
    canonical = _clean(body.get("canonical_broad_pose"))
    out["body"] = {
        "broad_pose": broad,
        "broad_pose_group": _pose_group(broad),
        "canonical_broad_pose": canonical,
        "canonical_broad_pose_group": _pose_group(canonical),
        "configuration": [
            str(x)
            for x in body.get("configuration", [])
            if isinstance(x, str)
        ],
        "torso": {
            "camera_orientation": _clean(preferred.get("camera_orientation")),
            "approx_yaw_deg": preferred.get("approx_yaw_deg"),
            "turn_direction": _clean(preferred.get("turn_direction")),
        },
    }
    out["head"] = {
        "horizontal": _clean(head.get("horizontal")),
        "vertical": _clean(head.get("vertical")),
        "turn_strength": _clean(head.get("turn_strength")),
        "composer_text": _clean(head.get("composer_text")),
    }
    out["gaze"] = {
        "horizontal": _clean(gaze.get("horizontal")),
        "vertical": _clean(gaze.get("vertical")),
        "camera_relationship": _clean(gaze.get("camera_relationship")),
    }
    out["capture"] = {
        "subtype": _clean(capture.get("subtype")),
        "composer_text": _clean(capture.get("composer_text")),
    }
    out["appearance"] = [
        str(x)
        for x in auth.get("appearance", [])
        if isinstance(x, str)
    ]
    out["objects"] = [
        str(x)
        for x in auth.get("objects", [])
        if isinstance(x, str)
    ]
    out["scene"] = [
        str(x)
        for x in auth.get("scene", [])
        if isinstance(x, str)
    ]
    out["caption"] = _clean(record.get("caption"))
    out["audit_violations"] = list(
        (record.get("caption_audit") or {}).get("violations") or []
    )
    return out


# Backward-compatible name for callers that only care about the composer projection.
snapshot_record = snapshot_projection


def _value_transition(
    name: str,
    wider: Any,
    tighter: Any,
    *,
    contradiction: bool = True,
) -> dict[str, Any]:
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
        "contradiction": bool(
            contradiction
            and state == "changed"
            and wider is not None
            and tighter is not None
        ),
    }


def _list_transition(
    name: str,
    wider: list[str],
    tighter: list[str],
) -> dict[str, Any]:
    wider_set = set(wider)
    tighter_set = set(tighter)
    return {
        "domain": name,
        "retained": sorted(wider_set & tighter_set),
        "dropped": sorted(wider_set - tighter_set),
        "emerged_in_tighter_crop": sorted(tighter_set - wider_set),
    }


def compare_snapshots(
    wider: dict[str, Any],
    tighter: dict[str, Any],
) -> dict[str, Any]:
    transitions = [
        _value_transition(
            "broad_pose_group",
            wider["body"]["broad_pose_group"],
            tighter["body"]["broad_pose_group"],
        ),
        _value_transition(
            "canonical_broad_pose_group",
            wider["body"]["canonical_broad_pose_group"],
            tighter["body"]["canonical_broad_pose_group"],
        ),
        _value_transition(
            "torso.turn_direction",
            wider["body"]["torso"]["turn_direction"],
            tighter["body"]["torso"]["turn_direction"],
        ),
        _value_transition(
            "head.horizontal",
            wider["head"]["horizontal"],
            tighter["head"]["horizontal"],
        ),
        _value_transition(
            "head.vertical",
            wider["head"]["vertical"],
            tighter["head"]["vertical"],
        ),
        _value_transition(
            "gaze.horizontal",
            wider["gaze"]["horizontal"],
            tighter["gaze"]["horizontal"],
        ),
        _value_transition(
            "gaze.vertical",
            wider["gaze"]["vertical"],
            tighter["gaze"]["vertical"],
        ),
        _value_transition(
            "gaze.camera_relationship",
            wider["gaze"]["camera_relationship"],
            tighter["gaze"]["camera_relationship"],
        ),
        _value_transition(
            "capture.subtype",
            wider["capture"]["subtype"],
            tighter["capture"]["subtype"],
        ),
    ]
    contradictions = [
        transition
        for transition in transitions
        if transition["contradiction"]
    ]
    expansions = [
        transition
        for transition in transitions
        if transition["state"] == "emerged_in_tighter_crop"
        and transition["domain"]
        in {"broad_pose_group", "canonical_broad_pose_group"}
    ]
    return {
        "wider": wider["image_key"],
        "tighter": tighter["image_key"],
        "wider_framing": wider["framing"],
        "tighter_framing": tighter["framing"],
        "semantic_transitions": transitions,
        "list_transitions": [
            _list_transition(
                "body.configuration",
                wider["body"]["configuration"],
                tighter["body"]["configuration"],
            ),
            _list_transition(
                "appearance",
                wider["appearance"],
                tighter["appearance"],
            ),
            _list_transition(
                "objects",
                wider["objects"],
                tighter["objects"],
            ),
            _list_transition(
                "scene",
                wider["scene"],
                tighter["scene"],
            ),
        ],
        "contradictions": contradictions,
        "authority_expansions_in_tighter_crop": expansions,
    }


def _semantic_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: snapshot[key]
        for key in (
            "framing",
            "body",
            "head",
            "gaze",
            "capture",
            "appearance",
            "objects",
            "scene",
        )
    }


def _audit_layer(
    family: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    members = [str(x) for x in family.get("members") or []]
    missing = [key for key in members if key not in snapshots]
    declared_order = [
        str(x)
        for x in family.get("wide_to_tight") or members
    ]
    ordered = [
        snapshots[key]
        for key in declared_order
        if key in snapshots
    ]

    relative_fov = (
        family.get("relative_fov_area")
        if isinstance(family.get("relative_fov_area"), dict)
        else {}
    )
    for snapshot in ordered:
        snapshot["source_relative_fov_area"] = relative_fov.get(
            snapshot["image_key"]
        )

    exact_groups = [
        set(map(str, group))
        for group in family.get("exact_duplicate_groups") or []
    ]
    pairs: list[dict[str, Any]] = []
    for wider, tighter in zip(ordered, ordered[1:]):
        pair = compare_snapshots(wider, tighter)
        key_pair = {wider["image_key"], tighter["image_key"]}
        pair["source_crop_relation"] = (
            "exact_duplicate"
            if any(key_pair.issubset(group) for group in exact_groups)
            else "wider_to_tighter"
        )
        wider_fov = wider.get("source_relative_fov_area")
        tighter_fov = tighter.get("source_relative_fov_area")
        pair["source_fov_ratio"] = (
            round(float(wider_fov) / float(tighter_fov), 3)
            if isinstance(wider_fov, (int, float))
            and isinstance(tighter_fov, (int, float))
            and tighter_fov
            else None
        )
        pair["framing_changed"] = (
            wider["framing"].get("composer_text")
            != tighter["framing"].get("composer_text")
        )
        pairs.append(pair)

    exact_mismatches: list[dict[str, Any]] = []
    for group in family.get("exact_duplicate_groups") or []:
        present = [
            str(key)
            for key in group
            if str(key) in snapshots
        ]
        if len(present) < 2:
            continue
        base = _semantic_payload(snapshots[present[0]])
        for other in present[1:]:
            if base != _semantic_payload(snapshots[other]):
                exact_mismatches.append(
                    {"a": present[0], "b": other}
                )

    contradictions = [
        {"pair": [pair["wider"], pair["tighter"]], **transition}
        for pair in pairs
        for transition in pair["contradictions"]
    ]
    expansions = [
        {"pair": [pair["wider"], pair["tighter"]], **transition}
        for pair in pairs
        for transition in pair["authority_expansions_in_tighter_crop"]
    ]
    if missing:
        status = "missing_records"
    elif exact_mismatches or contradictions or expansions:
        status = "review"
    else:
        status = "consistent"

    return {
        "missing_records": missing,
        "ordered_wide_to_tight": [
            snapshot["image_key"]
            for snapshot in ordered
        ],
        "member_snapshots": ordered,
        "pairwise_crop_transitions": pairs,
        "contradictions": contradictions,
        "authority_expansions_in_tighter_crop": expansions,
        "exact_duplicate_semantic_mismatches": exact_mismatches,
        "status": status,
    }


def audit_family(
    family: dict[str, Any],
    fact_sheets: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    members = set(family.get("members") or [])
    source = {
        key: snapshot_fact_sheet(value)
        for key, value in fact_sheets.items()
        if key in members
    }
    projection = {
        key: snapshot_projection(value)
        for key, value in records.items()
        if key in members
    }

    source_audit = _audit_layer(family, source)
    projection_audit = _audit_layer(family, projection)
    statuses = {
        source_audit["status"],
        projection_audit["status"],
    }
    if "missing_records" in statuses:
        status = "missing_records"
    elif "review" in statuses:
        status = "review"
    else:
        status = "consistent"

    return {
        "family_id": family.get("family_id"),
        "kind": family.get("kind"),
        "members": [str(x) for x in family.get("members") or []],
        "fact_sheet": source_audit,
        "composer_projection": projection_audit,
        "status": status,
    }


def _markdown_layer(
    lines: list[str],
    title: str,
    layer: dict[str, Any],
) -> None:
    lines += [
        f"### {title} — {layer['status']}",
        "",
    ]
    for pair in layer["pairwise_crop_transitions"]:
        changes = [
            transition
            for transition in pair["semantic_transitions"]
            if transition["state"] not in {"stable", "absent"}
        ]
        list_changed = any(
            item["dropped"] or item["emerged_in_tighter_crop"]
            for item in pair["list_transitions"]
        )
        if not changes and not list_changed:
            continue
        lines += [
            f"#### {pair['wider']} → {pair['tighter']}",
            "",
        ]
        for transition in changes:
            mark = (
                " **CONTRADICTION**"
                if transition["contradiction"]
                else ""
            )
            lines.append(
                f"- {transition['domain']}: {transition['state']} — "
                f"'{transition['wider']}' → '{transition['tighter']}'"
                f"{mark}"
            )
        for item in pair["list_transitions"]:
            if item["dropped"] or item["emerged_in_tighter_crop"]:
                lines.append(
                    f"- {item['domain']}: "
                    f"dropped={item['dropped'] or []}; "
                    f"emerged={item['emerged_in_tighter_crop'] or []}"
                )
        lines.append("")


def _markdown(index: dict[str, Any]) -> str:
    lines = [
        "# Crop Consistency Diagnostic v0.1",
        "",
        (
            "Crop defines the truth envelope: wider crops may authorize global "
            "pose/support/scene facts; tighter crops should preserve surviving "
            "visible facts while allowing out-of-frame semantics to collapse. "
            "Same-source family information is evaluation-only and never becomes "
            "evidence for the tighter crop."
        ),
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
        order = family["fact_sheet"]["ordered_wide_to_tight"]
        lines += [
            f"## {family['family_id']} — {family['status']}",
            "",
            "Wide → tight: " + " → ".join(order),
            "",
        ]
        _markdown_layer(
            lines,
            "Fact sheet (pre-composer)",
            family["fact_sheet"],
        )
        _markdown_layer(
            lines,
            "Composer evidence projection (pre-generation)",
            family["composer_projection"],
        )
    return "\n".join(lines).rstrip() + "\n"


def run(
    run_dir: Path,
    registry_path: Path,
    fact_sheet_dir: Path,
    composer_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    registry = _read_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported registry schema: {registry.get('schema_version')}"
        )

    families = registry.get("families") or []
    wanted_keys = {
        str(key)
        for family in families
        for key in (family.get("members") or [])
    }
    fact_sheets: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for key in sorted(wanted_keys):
        path = fact_sheet_dir / f"{key}.fact_sheet.json"
        if path.is_file():
            fact_sheets[key] = _read_json(path)
        path = composer_dir / f"{key}.composed.json"
        if path.is_file():
            records[key] = _read_json(path)

    out_json = output_dir / "crop_consistency.index.json"
    out_md = output_dir / "crop_consistency.report.md"
    if not overwrite and (out_json.exists() or out_md.exists()):
        raise FileExistsError(
            f"output exists; use --overwrite: {output_dir}"
        )

    audited = [
        audit_family(family, fact_sheets, records)
        for family in families
    ]
    counts = Counter(record["status"] for record in audited)
    index = {
        "schema_version": SCHEMA_VERSION,
        "registry": str(registry_path),
        "run_dir": str(run_dir),
        "fact_sheet_dir": str(fact_sheet_dir),
        "composer_dir": str(composer_dir),
        "record_count": len(wanted_keys),
        "loaded_fact_sheet_count": len(fact_sheets),
        "loaded_composer_record_count": len(records),
        "family_count": len(audited),
        "multi_member_family_count": sum(
            len(record["members"]) > 1
            for record in audited
        ),
        "status_counts": dict(sorted(counts.items())),
        "principle": (
            "crop defines the truth envelope; facts may collapse as evidence "
            "leaves frame but surviving facts should not contradict the same "
            "source photograph"
        ),
        "families": audited,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_json, index)
    out_md.write_text(_markdown(index), encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-source crop families first at the fact-sheet layer, "
            "then at the composer evidence-projection layer. "
            "No model calls and no mutations."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--fact-sheet-dir", type=Path)
    parser.add_argument("--composer-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(
            f"Run directory not found: {run_dir}",
            file=sys.stderr,
        )
        return 2

    registry = args.registry.expanduser().resolve()
    fact_sheet_dir = (
        args.fact_sheet_dir.expanduser().resolve()
        if args.fact_sheet_dir
        else run_dir / DEFAULT_FACT_SHEET_SUBDIR
    )
    composer_dir = (
        args.composer_dir.expanduser().resolve()
        if args.composer_dir
        else run_dir / DEFAULT_COMPOSER_SUBDIR
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / DEFAULT_OUTPUT_SUBDIR
    )

    for label, path in (
        ("Registry", registry),
        ("Fact-sheet directory", fact_sheet_dir),
        ("Composer directory", composer_dir),
    ):
        if not path.exists():
            print(
                f"{label} not found: {path}",
                file=sys.stderr,
            )
            return 2

    try:
        index = run(
            run_dir,
            registry,
            fact_sheet_dir,
            composer_dir,
            output_dir,
            overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        "Crop consistency: "
        f"families={index['family_count']} "
        f"multi={index['multi_member_family_count']} "
        f"fact_sheets={index['loaded_fact_sheet_count']}/{index['record_count']} "
        f"projections={index['loaded_composer_record_count']}/{index['record_count']}"
    )
    print(
        f"Status counts: "
        f"{json.dumps(index['status_counts'], sort_keys=True)}"
    )
    print(f"JSON: {output_dir / 'crop_consistency.index.json'}")
    print(f"Report: {output_dir / 'crop_consistency.report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
