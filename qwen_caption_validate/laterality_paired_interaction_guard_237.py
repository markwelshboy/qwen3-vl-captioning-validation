from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

from .laterality_match import _side_name
from .laterality_paired_interaction_guard_236 import (
    _complete_bilateral_arms,
    _hand_interactions,
    _qualified_side,
    _read,
    _same_target,
    _source_arm_record_for_target,
    _source_side,
    _write,
)
from .runner import model_slug, resolve_model_id


_GUARD_AUTHORITY = "paired_distinct_interaction_cross_field_conflict_withheld"


def _notes_restate_side(item: dict[str, Any], side: str) -> bool:
    """Require Analyze to repeat the anatomical side outside actor_part/source_actor_part.

    The structured source-side label alone is advisory because Analyze can use frame-left/
    frame-right language for limbs. A free-text note such as "left hand gripping fabric"
    provides a separate cross-field consistency signal before that advisory label is allowed
    to veto deterministic laterality. This is not treated as independent visual evidence; it
    merely raises the source semantic topology from a single-field claim to a repeated claim.
    """
    text = str(item.get("notes") or "")
    return bool(
        re.search(
            rf"\b{re.escape(side)}\s+(?:hand|wrist|finger|fingers)\b",
            text,
            re.I,
        )
    )


def _withhold_body_side(item: dict[str, Any], reason: str) -> None:
    state = item.setdefault("fusion_v2", {})
    state["qualified_anatomical_side"] = "unknown"
    state["laterality_selection_usable"] = False
    state["laterality_authority"] = _GUARD_AUTHORITY
    state.setdefault("laterality_reasons", []).append(
        f"Fusion-v2.3.7 withholds anatomical side: {reason}"
    )
    source_part = str(item.get("source_part") or item.get("part") or "")
    item["part"] = _side_name(source_part, None)


def _withhold_interaction_side(item: dict[str, Any], reason: str) -> None:
    state = item.setdefault("fusion_v2", {})
    state["qualified_actor_anatomical_side"] = "unknown"
    state["laterality_selection_usable"] = False
    state["laterality_authority"] = _GUARD_AUTHORITY
    state.setdefault("laterality_reasons", []).append(
        f"Fusion-v2.3.7 withholds actor anatomical side: {reason}"
    )
    source_actor = str(item.get("source_actor_part") or item.get("actor_part") or "")
    item["actor_part"] = _side_name(source_actor, None)


def guard_cross_field_paired_distinct_interactions(
    payload: dict[str, Any],
    dwpose: dict[str, Any],
) -> dict[str, Any]:
    """Withhold a conflicting paired-hand side only when source laterality is cross-field stable.

    Fusion-v2.3.6 established that a distinct-target two-hand configuration can expose a
    greedy semantic-to-wrist association error, but it was intentionally too conservative:
    a single structured Analyze source-side label could veto a deterministic DWPose/SAM
    correction. That also withholds useful correct corrections when Analyze used frame-side
    rather than anatomical-side labels.

    v2.3.7 therefore keeps the paired topology guard but requires BOTH interactions to repeat
    their source anatomical side in free-text interaction evidence (for example, notes saying
    "left hand gripping fabric" and "right hand resting on table"). The repeated source claim
    is still not promoted to anatomical truth; it only permits a conflict to become ambiguity,
    so the result remains side-withholding rather than source-side restoration.
    """
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    if not isinstance(fusion, dict):
        return out

    audit: dict[str, Any] = {
        "schema_version": "paired-distinct-interaction-laterality-audit-1.1",
        "bilateral_arm_chains_complete": _complete_bilateral_arms(dwpose),
        "pairs_considered": [],
        "pairs_applied": [],
        "policy": {
            "opposite_source_sides_required": True,
            "distinct_targets_required": True,
            "source_side_body_relation_required_for_each_target": True,
            "source_side_restatement_in_interaction_notes_required_for_both": True,
            "complete_bilateral_dwpose_arm_chains_required": True,
            "source_laterality_never_blindly_restored": True,
            "conflicting_refined_side_is_withheld": True,
        },
    }

    if not audit["bilateral_arm_chains_complete"]:
        fusion["schema_version"] = "analysis-fusion-2.3.7"
        fusion["paired_distinct_interaction_laterality_audit"] = audit
        return out

    hands = _hand_interactions(fusion)
    used_interactions: set[int] = set()
    used_parts: set[int] = set()

    for i in range(len(hands)):
        first_index, first = hands[i]
        first_side = _source_side(first, interaction=True)
        for j in range(i + 1, len(hands)):
            second_index, second = hands[j]
            second_side = _source_side(second, interaction=True)
            if {first_side, second_side} != {"left", "right"}:
                continue
            if _same_target(first.get("target"), second.get("target")):
                continue

            by_side = {
                str(first_side): (first_index, first),
                str(second_side): (second_index, second),
            }
            left_i, left_item = by_side["left"]
            right_i, right_item = by_side["right"]
            left_body = _source_arm_record_for_target(fusion, "left", left_item.get("target"))
            right_body = _source_arm_record_for_target(fusion, "right", right_item.get("target"))
            left_restate = _notes_restate_side(left_item, "left")
            right_restate = _notes_restate_side(right_item, "right")

            record: dict[str, Any] = {
                "interaction_indices": [left_i, right_i],
                "source_sides": ["left", "right"],
                "targets": [left_item.get("target"), right_item.get("target")],
                "qualified_sides_before": [
                    _qualified_side(left_item, interaction=True) or "unknown",
                    _qualified_side(right_item, interaction=True) or "unknown",
                ],
                "source_body_matches": {
                    "left": left_body[0] if left_body else None,
                    "right": right_body[0] if right_body else None,
                },
                "source_side_restatement_in_notes": {
                    "left": left_restate,
                    "right": right_restate,
                },
            }
            audit["pairs_considered"].append(copy.deepcopy(record))

            if left_body is None or right_body is None:
                record["action"] = "unchanged"
                record["reason"] = "paired_source_side_body_target_relations_not_uniquely_available"
                continue

            if not (left_restate and right_restate):
                record["action"] = "unchanged"
                record["reason"] = "source_side_not_cross_field_repeated_for_both_interactions"
                continue

            left_q = _qualified_side(left_item, interaction=True)
            right_q = _qualified_side(right_item, interaction=True)
            left_body_q = _qualified_side(left_body[1])
            right_body_q = _qualified_side(right_body[1])
            conflict = any(
                q is not None and q != source
                for q, source in (
                    (left_q, "left"),
                    (right_q, "right"),
                    (left_body_q, "left"),
                    (right_body_q, "right"),
                )
            )
            if not conflict:
                record["action"] = "unchanged"
                record["reason"] = "no_refined_side_conflicts_with_cross_field_paired_topology"
                continue

            reason = (
                "opposite-source hand interactions have distinct targets, each target is echoed "
                "by its source-side arm record, and both anatomical sides are independently "
                "repeated in interaction evidence text; refined frame-location laterality "
                "conflicts with that cross-field topology, so preserve the interaction facts "
                "and withhold anatomical side"
            )

            for interaction_index, interaction in ((left_i, left_item), (right_i, right_item)):
                if interaction_index not in used_interactions:
                    _withhold_interaction_side(interaction, reason)
                    used_interactions.add(interaction_index)
            for part_index, part in (left_body, right_body):
                if part_index not in used_parts:
                    _withhold_body_side(part, reason)
                    used_parts.add(part_index)

            record.update(
                action="withheld_pair_laterality",
                authority=_GUARD_AUTHORITY,
                reason=reason,
                qualified_sides_after=["unknown", "unknown"],
            )
            audit["pairs_applied"].append(copy.deepcopy(record))

    fusion["schema_version"] = "analysis-fusion-2.3.7"
    fusion["paired_distinct_interaction_laterality_audit"] = audit
    fusion.setdefault("selection_policy", {})["paired_distinct_interaction_laterality"] = (
        "Distinct-target opposite-source hand pairs may veto a conflicting refined side only "
        "when each target is echoed by its source-side arm record and both source anatomical "
        "sides are repeated in free-text interaction evidence; source laterality is never "
        "restored blindly, and conflicts are resolved by withholding side."
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-laterality-paired-interaction-guard-237",
        description=(
            "Fusion v2.3.7: cross-field-qualified paired distinct-target interaction "
            "laterality guard."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    # Replay from 2.3.5, not 2.3.6: v2.3.7 supersedes the experimental
    # over-conservative guard rather than trying to restore sides it already withheld.
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "fusion-v2.3.7" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion-v2.3.5"), (dwpose_dir, "DWPose")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    written = reused = missing = applied = 0
    records: list[dict[str, Any]] = []

    for fusion_path in sorted(fusion_dir.glob("*.fused_v2_3.json")):
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_path = output_dir / fusion_path.name
        if out_path.exists() and not args.overwrite:
            reused += 1
            continue
        dw_path = dwpose_dir / f"{key}.dwpose.json"
        if not dw_path.is_file():
            missing += 1
            records.append({"image_key": key, "status": "missing_source"})
            continue

        refined = guard_cross_field_paired_distinct_interactions(_read(fusion_path), _read(dw_path))
        _write(out_path, refined)
        audit = ((refined.get("fusion") or {}).get("paired_distinct_interaction_laterality_audit") or {})
        count = len(audit.get("pairs_applied") or [])
        applied += count
        written += 1
        records.append(
            {
                "image_key": key,
                "status": "written",
                "pairs_considered": len(audit.get("pairs_considered") or []),
                "pairs_applied": count,
            }
        )

    index = {
        "schema_version": "analysis-fusion-2.3.7-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "source_fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "output_dir": str(output_dir),
        "written": written,
        "reused": reused,
        "missing_sources": missing,
        "paired_distinct_interaction_guards_applied": applied,
        "records": records,
    }
    _write(output_dir / "laterality_paired_interaction_guard_237.index.json", index)
    print(f"Fusion-v2.3.7 output: {output_dir}")
    print(f"Written: {written}; reused: {reused}; missing: {missing}; guards applied: {applied}")
    return 0 if written or reused else 2


if __name__ == "__main__":
    raise SystemExit(main())
