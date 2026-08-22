from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

from .laterality_geometry import (
    _hand_entities,
    _load_sam2d,
    _mirror_sensitive,
    _read,
    _target_points,
    _write,
)
from .laterality_match import (
    _distal_arm,
    _family,
    _match_chain,
    _match_hand,
    _raw_side,
    _side_name,
)
from .runner import model_slug, resolve_model_id


def _normalized_part(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").lower()).strip()


def _source_key(item: dict[str, Any]) -> tuple[str | None, str | None]:
    family = _family(item)
    side = _raw_side(item.get("part")) or str(item.get("anatomical_side") or "").lower()
    if side not in {"left", "right"}:
        side = None
    return side, family


def _set_part_side(
    item: dict[str, Any],
    *,
    source_part: str,
    source_side: str | None,
    qualified_side: str | None,
    authority: str,
    reason: str,
) -> None:
    state = item.setdefault("fusion_v2", {})
    state["source_anatomical_side"] = source_side or "unknown"
    state["qualified_anatomical_side"] = qualified_side or "unknown"
    state["laterality_selection_usable"] = qualified_side in {"left", "right"}
    state["laterality_authority"] = authority
    state.setdefault("laterality_reasons", []).append(
        (
            f"Fusion-v2.3.1 qualifies side={qualified_side}: {reason}"
            if qualified_side
            else f"Fusion-v2.3.1 withholds Analyze laterality: {reason}"
        )
    )
    item.setdefault("source_part", source_part)
    item["part"] = _side_name(source_part, qualified_side)


def _set_interaction_side(
    item: dict[str, Any],
    *,
    source_actor: str,
    source_side: str | None,
    qualified_side: str | None,
    authority: str,
    reason: str,
) -> None:
    state = item.setdefault("fusion_v2", {})
    state["source_actor_anatomical_side"] = source_side or "unknown"
    state["qualified_actor_anatomical_side"] = qualified_side or "unknown"
    state["laterality_selection_usable"] = qualified_side in {"left", "right"}
    state["laterality_authority"] = authority
    state.setdefault("laterality_reasons", []).append(
        (
            f"Fusion-v2.3.1 qualifies actor side={qualified_side}: {reason}"
            if qualified_side
            else f"Fusion-v2.3.1 withholds interaction laterality: {reason}"
        )
    )
    item.setdefault("source_actor_part", source_actor)
    item["actor_part"] = _side_name(source_actor, qualified_side)


def refine_laterality(
    payload: dict[str, Any],
    analysis: dict[str, Any],
    dw: dict[str, Any],
    sam: dict[str, Any],
    sam_path: Path,
) -> dict[str, Any]:
    """Refine Fusion-v2.3 laterality using observed DWPose geometry plus SAM3D corroboration.

    DWPose is the primary observed laterality/chain authority. SAM3D may corroborate or
    veto a DWPose anatomical label only for joints that DWPose actually observed.
    SAM3D reconstruction by itself never establishes visibility. Analyze laterality is
    advisory and may be corrected when the physical entity is deterministically matched.
    """
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    if not isinstance(fusion, dict):
        return out

    points = _target_points(dw)
    sam2d = _load_sam2d(sam_path, sam)
    hand_entities = _hand_entities(dw, points, sam2d)
    mirror_sensitive = _mirror_sensitive(analysis)

    audit: dict[str, Any] = {
        "schema_version": "laterality-authority-audit-1.0",
        "mirror_sensitive": mirror_sensitive,
        "sam3d_used": sam2d is not None,
        "policy": {
            "dwpose_observed_geometry_primary": True,
            "sam3d_requires_dwpose_observation": True,
            "sam3d_reconstruction_never_establishes_visibility": True,
            "analyze_laterality_is_advisory": True,
        },
        "hand_entities": copy.deepcopy(hand_entities),
        "body_part_decisions": [],
        "interaction_decisions": [],
        "duplicate_entity_downgrades": [],
    }

    source_anchors: dict[tuple[str, str], list[dict[str, Any]]] = {}
    physical_anchors: dict[tuple[str, str], list[tuple[int, dict[str, Any], int]]] = {}

    body_parts = fusion.get("qualified_body_parts") or []
    for index, item in enumerate(body_parts):
        if not isinstance(item, dict):
            continue

        state = item.get("fusion_v2") or {}
        source_part = str(item.get("source_part") or item.get("part") or "")
        source_side = _raw_side(source_part)
        if source_side is None:
            raw_side = str(item.get("anatomical_side") or "").lower()
            source_side = raw_side if raw_side in {"left", "right"} else None
        family = _family(item)
        decision: dict[str, Any] = {
            "index": index,
            "source_part": source_part,
            "source_side": source_side or "unknown",
            "family": family,
        }

        owner = state.get("qualified_ownership") or item.get("ownership")
        if not state.get("selection_usable") or owner != "target" or family not in {"arm", "leg"}:
            decision["action"] = "unchanged"
            audit["body_part_decisions"].append(decision)
            continue

        if mirror_sensitive:
            _set_part_side(
                item,
                source_part=source_part,
                source_side=source_side,
                qualified_side=None,
                authority="mirror_sensitive_withheld",
                reason="mirror/reflection geometry prevents real-person anatomical-side correction",
            )
            decision.update(
                action="withheld",
                authority="mirror_sensitive_withheld",
                reason="mirror_sensitive",
            )
            audit["body_part_decisions"].append(decision)
            continue

        qualified_side: str | None = None
        authority = "unresolved_entity_association"
        reason = ""
        rank = 0

        if family == "arm" and _distal_arm(item):
            entity, match_reason = _match_hand(item, hand_entities)
            if entity is not None and entity.get("qualified_side") in {"left", "right"}:
                qualified_side = str(entity["qualified_side"])
                authority = str(entity.get("authority") or "dwpose_observed_wrist")
                reason = f"{match_reason}; {entity.get('resolution_reason') or 'observed_hand_entity'}"
                rank = 3 if authority == "dwpose_sam_correlated" else 2
                decision["hand_entity"] = copy.deepcopy(entity)
            else:
                reason = match_reason
        else:
            qualified_side, match_info = _match_chain(item, family, dw, points, sam2d)
            authority = str(match_info.get("authority") or "unresolved_entity_association")
            reason = str(match_info.get("reason") or "unresolved_entity_association")
            rank = 2 if qualified_side else 0
            decision["chain_match"] = match_info

        _set_part_side(
            item,
            source_part=source_part,
            source_side=source_side,
            qualified_side=qualified_side,
            authority=authority,
            reason=reason,
        )

        if qualified_side:
            action = "qualified" if source_side == qualified_side else "corrected"
            decision.update(
                action=action,
                qualified_side=qualified_side,
                authority=authority,
                reason=reason,
                anchor_rank=rank,
            )
            physical_anchors.setdefault((qualified_side, family), []).append((index, item, rank))
            if source_side:
                source_anchors.setdefault((source_side, family), []).append(item)
        else:
            decision.update(action="withheld", authority=authority, reason=reason)

        audit["body_part_decisions"].append(decision)

    for (side, family), records in physical_anchors.items():
        if len(records) < 2:
            continue
        ordered = sorted(records, key=lambda value: value[2], reverse=True)
        best_rank = ordered[0][2]
        for index, item, rank in ordered[1:]:
            if rank >= best_rank:
                continue
            state = item.setdefault("fusion_v2", {})
            old_part = str(item.get("source_part") or item.get("part") or "")
            state["qualified_anatomical_side"] = "unknown"
            state["laterality_selection_usable"] = False
            state["laterality_authority"] = "duplicate_physical_entity_withheld"
            state.setdefault("laterality_reasons", []).append(
                "Fusion-v2.3.1 withholds duplicate semantic side assignment to the same physical entity"
            )
            item["part"] = _side_name(old_part, None)
            audit["duplicate_entity_downgrades"].append(
                {
                    "index": index,
                    "side": side,
                    "family": family,
                    "reason": "weaker_duplicate_semantic_record_for_same_physical_entity",
                }
            )

    interactions = fusion.get("qualified_interactions") or []
    for index, item in enumerate(interactions):
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        source_actor = str(item.get("source_actor_part") or item.get("actor_part") or "")
        source_side = _raw_side(source_actor)
        decision: dict[str, Any] = {
            "index": index,
            "source_actor_part": source_actor,
            "source_side": source_side or "unknown",
        }

        owner = state.get("qualified_actor_ownership") or item.get("actor_ownership")
        if not state.get("selection_usable") or owner != "target":
            decision["action"] = "unchanged"
            audit["interaction_decisions"].append(decision)
            continue

        if mirror_sensitive:
            _set_interaction_side(
                item,
                source_actor=source_actor,
                source_side=source_side,
                qualified_side=None,
                authority="mirror_sensitive_withheld",
                reason="mirror/reflection geometry prevents real-person anatomical-side correction",
            )
            decision.update(action="withheld", authority="mirror_sensitive_withheld")
            audit["interaction_decisions"].append(decision)
            continue

        actor_family = _family({"part": source_actor, "visible_subparts": []})
        qualified_side: str | None = None
        authority = "unresolved_interaction_entity"
        reason = ""

        if source_side and actor_family:
            candidates = source_anchors.get((source_side, actor_family)) or []
            candidates = [
                candidate
                for candidate in candidates
                if (candidate.get("fusion_v2") or {}).get("laterality_selection_usable")
            ]
            corrected = {
                str((candidate.get("fusion_v2") or {}).get("qualified_anatomical_side"))
                for candidate in candidates
                if (candidate.get("fusion_v2") or {}).get("qualified_anatomical_side") in {"left", "right"}
            }
            if len(corrected) == 1:
                qualified_side = next(iter(corrected))
                candidate = candidates[0]
                authority = str((candidate.get("fusion_v2") or {}).get("laterality_authority") or "refined_body_entity")
                reason = "interaction_inherits_corrected_source_body_entity"

        if qualified_side is None and actor_family == "arm":
            handish = bool(re.search(r"\b(?:hand|finger|wrist)\b", source_actor, re.I))
            qualified_hands = [
                entity for entity in hand_entities if entity.get("qualified_side") in {"left", "right"}
            ]
            if handish and len(qualified_hands) == 1:
                entity = qualified_hands[0]
                qualified_side = str(entity["qualified_side"])
                authority = str(entity.get("authority") or "dwpose_observed_wrist")
                reason = "single_qualified_observed_hand_entity_supports_interaction"

        _set_interaction_side(
            item,
            source_actor=source_actor,
            source_side=source_side,
            qualified_side=qualified_side,
            authority=authority,
            reason=reason or "interaction entity could not be independently matched",
        )
        decision.update(
            action="qualified" if qualified_side else "withheld",
            qualified_side=qualified_side or "unknown",
            authority=authority,
            reason=reason or "unresolved_interaction_entity",
        )
        audit["interaction_decisions"].append(decision)

    fusion["schema_version"] = "analysis-fusion-2.3.1"
    fusion["laterality_authority_audit"] = audit
    fusion.setdefault("selection_policy", {})["laterality_authority"] = (
        "DWPose observed target-chain geometry is primary; SAM3D may corroborate/veto only "
        "corresponding DWPose-observed joints; Analyze laterality is advisory."
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-laterality-refine",
        description=(
            "Refine cached Fusion-v2.3 anatomical laterality using observed DWPose target "
            "chains and visibility-gated SAM3D joint-label corroboration."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--sam3d-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)

    fusion_dir = (
        args.fusion_dir.expanduser().resolve()
        if args.fusion_dir
        else run_dir / "fusion-v2.3" / slug
    )
    dwpose_dir = (
        args.dwpose_dir.expanduser().resolve()
        if args.dwpose_dir
        else run_dir / "dwpose"
    )
    sam3d_dir = (
        args.sam3d_dir.expanduser().resolve()
        if args.sam3d_dir
        else run_dir / "sam3d"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "fusion-v2.3.1" / slug
    )

    for path, label in (
        (fusion_dir, "Fusion-v2.3"),
        (dwpose_dir, "DWPose"),
        (sam3d_dir, "SAM3D"),
    ):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    missing = 0
    records: list[dict[str, Any]] = []

    for fusion_path in sorted(fusion_dir.glob("*.fused_v2_3.json")):
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_path = output_dir / fusion_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        dw_path = dwpose_dir / f"{key}.dwpose.json"
        sam_path = sam3d_dir / f"{key}.sam3d.json"
        payload = _read(fusion_path)

        analysis_path = Path(str(payload.get("analysis_path") or ""))
        if not analysis_path.is_file():
            analysis_path = run_dir / slug / f"{key}.analysis.json"

        if not dw_path.is_file() or not sam_path.is_file() or not analysis_path.is_file():
            missing += 1
            records.append(
                {
                    "image_key": key,
                    "status": "missing_source",
                    "analysis_path": str(analysis_path),
                    "dwpose_path": str(dw_path),
                    "sam3d_path": str(sam_path),
                }
            )
            continue

        analysis_record = _read(analysis_path)
        analysis = analysis_record.get("analysis")
        if not isinstance(analysis, dict):
            missing += 1
            records.append({"image_key": key, "status": "missing_analysis"})
            continue

        refined = refine_laterality(
            payload,
            analysis,
            _read(dw_path),
            _read(sam_path),
            sam_path,
        )
        refined["source_fusion_path"] = str(fusion_path)
        refined["laterality_refinement"] = {
            "schema_version": "fusion-laterality-refinement-2.3.1",
            "source_fusion_schema": ((payload.get("fusion") or {}).get("schema_version")),
            "analysis_path": str(analysis_path),
            "dwpose_path": str(dw_path),
            "sam3d_path": str(sam_path),
        }
        _write(out_path, refined)
        written += 1

        audit = ((refined.get("fusion") or {}).get("laterality_authority_audit") or {})
        decisions = audit.get("body_part_decisions") or []
        records.append(
            {
                "image_key": key,
                "status": "written",
                "corrected": sum(1 for item in decisions if item.get("action") == "corrected"),
                "qualified": sum(1 for item in decisions if item.get("action") == "qualified"),
                "withheld": sum(1 for item in decisions if item.get("action") == "withheld"),
                "sam3d_used": bool(audit.get("sam3d_used")),
                "mirror_sensitive": bool(audit.get("mirror_sensitive")),
            }
        )

    index = {
        "schema_version": "analysis-fusion-2.3.1-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "source_fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "sam3d_dir": str(sam3d_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_sources": missing,
        "records": records,
    }
    _write(output_dir / "laterality_refine.index.json", index)
    print(f"Fusion-v2.3.1 output: {output_dir}")
    print(f"Written: {written}; reused: {skipped}; missing source records: {missing}")
    return 0 if written or skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
