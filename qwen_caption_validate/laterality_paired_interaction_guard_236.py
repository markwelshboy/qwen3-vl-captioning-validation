from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

from .laterality_geometry import _connectivity
from .laterality_match import _raw_side, _side_name
from .runner import model_slug, resolve_model_id


_HAND_RE = re.compile(r"\b(?:hand|wrist|finger|fingers)\b", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GUARD_AUTHORITY = "paired_distinct_interaction_conflict_withheld"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _source_side(item: dict[str, Any], *, interaction: bool = False) -> str | None:
    state = item.get("fusion_v2") or {}
    fields = (
        state.get("source_actor_anatomical_side"),
        _raw_side(item.get("source_actor_part")),
    ) if interaction else (
        state.get("source_anatomical_side"),
        _raw_side(item.get("source_part")),
        item.get("anatomical_side"),
    )
    for value in fields:
        side = str(value or "").lower()
        if side in {"left", "right"}:
            return side
    return None


def _qualified_side(item: dict[str, Any], *, interaction: bool = False) -> str | None:
    state = item.get("fusion_v2") or {}
    key = "qualified_actor_anatomical_side" if interaction else "qualified_anatomical_side"
    side = str(state.get(key) or "").lower()
    return side if side in {"left", "right"} else None


def _target_owned_usable(item: dict[str, Any], *, interaction: bool = False) -> bool:
    state = item.get("fusion_v2") or {}
    ownership_key = "qualified_actor_ownership" if interaction else "qualified_ownership"
    fallback_key = "actor_ownership" if interaction else "ownership"
    owner = state.get(ownership_key) or item.get(fallback_key)
    return bool(state.get("selection_usable")) and owner == "target"


def _norm_target(value: Any) -> str:
    tokens = _TOKEN_RE.findall(str(value or "").lower())
    stop = {"the", "a", "an", "some", "edge", "surface"}
    return " ".join(token for token in tokens if token not in stop)


def _target_tokens(value: Any) -> set[str]:
    text = _norm_target(value)
    return set(text.split()) if text else set()


def _same_target(a: Any, b: Any) -> bool:
    ta = _target_tokens(a)
    tb = _target_tokens(b)
    if not ta or not tb:
        return False
    return bool(ta & tb)


def _hand_interactions(fusion: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(fusion.get("qualified_interactions") or []):
        if not isinstance(item, dict) or not _target_owned_usable(item, interaction=True):
            continue
        if str(item.get("evidence_status") or "observed").lower() not in {"", "observed"}:
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.70:
            continue
        source_actor = str(item.get("source_actor_part") or item.get("actor_part") or "")
        if not _HAND_RE.search(source_actor):
            continue
        if _source_side(item, interaction=True) not in {"left", "right"}:
            continue
        if not _target_tokens(item.get("target")):
            continue
        out.append((index, item))
    return out


def _source_arm_record_for_target(
    fusion: dict[str, Any],
    side: str,
    target: Any,
) -> tuple[int, dict[str, Any]] | None:
    matches: list[tuple[int, dict[str, Any]]] = []
    for index, item in enumerate(fusion.get("qualified_body_parts") or []):
        if not isinstance(item, dict) or not _target_owned_usable(item):
            continue
        if _source_side(item) != side:
            continue
        source_part = str(item.get("source_part") or item.get("part") or "")
        semantic_text = " ".join(
            str(item.get(field) or "")
            for field in ("part", "source_part", "visible_subparts", "geometry", "contact", "support")
        )
        if not re.search(r"\b(?:arm|hand|wrist|forearm|elbow|finger)\b", source_part + " " + semantic_text, re.I):
            continue
        if not _same_target(target, semantic_text):
            continue
        matches.append((index, item))
    return matches[0] if len(matches) == 1 else None


def _complete_bilateral_arms(dwpose: dict[str, Any]) -> bool:
    connectivity = _connectivity(dwpose)
    return all(bool((connectivity.get(f"{side}_arm") or {}).get("complete")) for side in ("left", "right"))


def _withhold_body_side(item: dict[str, Any], reason: str) -> None:
    state = item.setdefault("fusion_v2", {})
    state["qualified_anatomical_side"] = "unknown"
    state["laterality_selection_usable"] = False
    state["laterality_authority"] = _GUARD_AUTHORITY
    state.setdefault("laterality_reasons", []).append(
        f"Fusion-v2.3.6 withholds anatomical side: {reason}"
    )
    source_part = str(item.get("source_part") or item.get("part") or "")
    item["part"] = _side_name(source_part, None)


def _withhold_interaction_side(item: dict[str, Any], reason: str) -> None:
    state = item.setdefault("fusion_v2", {})
    state["qualified_actor_anatomical_side"] = "unknown"
    state["laterality_selection_usable"] = False
    state["laterality_authority"] = _GUARD_AUTHORITY
    state.setdefault("laterality_reasons", []).append(
        f"Fusion-v2.3.6 withholds actor anatomical side: {reason}"
    )
    source_actor = str(item.get("source_actor_part") or item.get("actor_part") or "")
    item["actor_part"] = _side_name(source_actor, None)


def guard_paired_distinct_interactions(
    payload: dict[str, Any],
    dwpose: dict[str, Any],
) -> dict[str, Any]:
    """Withhold greedy hand laterality when a distinct-target bilateral pair conflicts.

    This guard is deliberately narrow. It does not restore Analyze's source-side label.
    Instead it detects a paired semantic topology such as left-hand→fabric and
    right-hand→table where each source-side arm record independently names the same
    target relation. If downstream greedy frame-location matching flips/collapses that
    pair, anatomical side is withheld for the affected pair while the interaction facts
    themselves remain usable.

    The rule requires complete bilateral DWPose arm chains, two opposite source sides,
    distinct targets, and source-side arm corroboration for both targets. Single-hand
    corrections and bilateral same-target interactions are therefore untouched.
    """
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    if not isinstance(fusion, dict):
        return out

    audit: dict[str, Any] = {
        "schema_version": "paired-distinct-interaction-laterality-audit-1.0",
        "bilateral_arm_chains_complete": _complete_bilateral_arms(dwpose),
        "pairs_considered": [],
        "pairs_applied": [],
        "policy": {
            "opposite_source_sides_required": True,
            "distinct_targets_required": True,
            "source_side_body_relation_required_for_each_target": True,
            "complete_bilateral_dwpose_arm_chains_required": True,
            "source_laterality_never_blindly_restored": True,
            "conflicting_refined_side_is_withheld": True,
        },
    }

    if not audit["bilateral_arm_chains_complete"]:
        fusion["schema_version"] = "analysis-fusion-2.3.6"
        fusion["paired_distinct_interaction_laterality_audit"] = audit
        return out

    hands = _hand_interactions(fusion)
    used_interactions: set[int] = set()
    used_parts: set[int] = set()

    for i in range(len(hands)):
        left_index, first = hands[i]
        first_side = _source_side(first, interaction=True)
        for j in range(i + 1, len(hands)):
            right_index, second = hands[j]
            second_side = _source_side(second, interaction=True)
            if {first_side, second_side} != {"left", "right"}:
                continue
            if _same_target(first.get("target"), second.get("target")):
                continue

            by_side = {
                str(first_side): (left_index, first),
                str(second_side): (right_index, second),
            }
            left_i, left_item = by_side["left"]
            right_i, right_item = by_side["right"]
            left_body = _source_arm_record_for_target(fusion, "left", left_item.get("target"))
            right_body = _source_arm_record_for_target(fusion, "right", right_item.get("target"))

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
            }
            audit["pairs_considered"].append(copy.deepcopy(record))

            if left_body is None or right_body is None:
                record["action"] = "unchanged"
                record["reason"] = "paired_source_side_body_target_relations_not_uniquely_available"
                continue

            left_q = _qualified_side(left_item, interaction=True)
            right_q = _qualified_side(right_item, interaction=True)
            left_body_q = _qualified_side(left_body[1])
            right_body_q = _qualified_side(right_body[1])
            conflict = any((q is not None and q != source) for q, source in (
                (left_q, "left"),
                (right_q, "right"),
                (left_body_q, "left"),
                (right_body_q, "right"),
            ))
            if not conflict:
                record["action"] = "unchanged"
                record["reason"] = "no_refined_side_conflicts_with_paired_semantic_topology"
                continue

            reason = (
                "opposite-source hand interactions have distinct targets independently echoed by "
                "their source-side arm records, but refined frame-location laterality conflicts "
                "with that paired topology; preserve interaction semantics and withhold side"
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

    fusion["schema_version"] = "analysis-fusion-2.3.6"
    fusion["paired_distinct_interaction_laterality_audit"] = audit
    fusion.setdefault("selection_policy", {})["paired_distinct_interaction_laterality"] = (
        "When opposite-source target-hand interactions have distinct targets and each target is "
        "independently echoed by its source-side arm record, a conflicting greedy refined side is "
        "withheld rather than allowing coarse frame-location matching to publish false anatomy."
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-laterality-paired-interaction-guard-236",
        description="Fusion v2.3.6: withhold conflicting greedy hand laterality in paired distinct-target interactions.",
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
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "fusion-v2.3.6" / slug)).expanduser().resolve()

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

        refined = guard_paired_distinct_interactions(_read(fusion_path), _read(dw_path))
        _write(out_path, refined)
        audit = ((refined.get("fusion") or {}).get("paired_distinct_interaction_laterality_audit") or {})
        count = len(audit.get("pairs_applied") or [])
        applied += count
        written += 1
        records.append({
            "image_key": key,
            "status": "written",
            "pairs_considered": len(audit.get("pairs_considered") or []),
            "pairs_applied": count,
        })

    index = {
        "schema_version": "analysis-fusion-2.3.6-run",
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
    _write(output_dir / "laterality_paired_interaction_guard_236.index.json", index)
    print(f"Fusion-v2.3.6 output: {output_dir}")
    print(f"Written: {written}; reused: {reused}; missing: {missing}; guards applied: {applied}")
    return 0 if written or reused else 2


if __name__ == "__main__":
    raise SystemExit(main())
