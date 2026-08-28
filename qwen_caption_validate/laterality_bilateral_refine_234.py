from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

from .laterality_bilateral_guard import (
    _bilateral_hand_support,
    _complete_bilateral_chains,
    _eligible_target_item,
    _norm_text,
    _source_side,
)
from .laterality_geometry import _read, _write
from .laterality_match import _family, _side_name
from .runner import model_slug, resolve_model_id


def _semantic_signature_without_frame(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _family(item),
        str(item.get("visibility") or ""),
        tuple(sorted(_norm_text(value) for value in (item.get("visible_subparts") or []))),
        _norm_text(item.get("geometry")),
        _norm_text(item.get("contact")),
        _norm_text(item.get("support")),
        _norm_text(item.get("foreshortening")),
    )


def _frame_signature(value: Any) -> tuple[str | None, str]:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    has_left = bool(re.search(r"\bleft\b", text))
    has_right = bool(re.search(r"\bright\b", text))
    side = "left" if has_left and not has_right else "right" if has_right and not has_left else None
    neutral = re.sub(r"\b(?:left|right)\b", "", text)
    neutral = re.sub(r"\s+", " ", neutral).strip(" ,.;:_-")
    return side, neutral


def _frame_relation(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    left_raw = str(left.get("image_location") or "").lower().replace("_", " ").strip()
    right_raw = str(right.get("image_location") or "").lower().replace("_", " ").strip()
    if left_raw == right_raw:
        return "same"
    left_side, left_neutral = _frame_signature(left_raw)
    right_side, right_neutral = _frame_signature(right_raw)
    if {left_side, right_side} == {"left", "right"} and left_neutral == right_neutral:
        return "complementary"
    return None


def _candidate_pairs(parts: list[dict[str, Any]]) -> list[tuple[str, int, int, str]]:
    out: list[tuple[str, int, int, str]] = []
    for family in ("arm", "leg"):
        candidates = [
            (index, item, _source_side(item))
            for index, item in enumerate(parts)
            if isinstance(item, dict) and _eligible_target_item(item) and _family(item) == family
        ]
        if len(candidates) != 2:
            continue
        by_side = {side: (index, item) for index, item, side in candidates if side in {"left", "right"}}
        if set(by_side) != {"left", "right"}:
            continue
        left_index, left = by_side["left"]
        right_index, right = by_side["right"]
        if _semantic_signature_without_frame(left) != _semantic_signature_without_frame(right):
            continue
        relation = _frame_relation(left, right)
        if relation is None:
            continue
        out.append((family, left_index, right_index, relation))
    return out


def _apply_side(item: dict[str, Any], side: str, authority: str, reason: str, *, clear_frame: bool) -> None:
    state = item.setdefault("fusion_v2", {})
    state.setdefault("source_anatomical_side", _source_side(item) or "unknown")
    source_part = str(item.get("source_part") or item.get("part") or "")
    item.setdefault("source_part", source_part)
    state["qualified_anatomical_side"] = side
    state["laterality_selection_usable"] = True
    state["laterality_authority"] = authority
    state.setdefault("laterality_reasons", []).append(f"Fusion-v2.3.4 qualifies side={side}: {reason}")
    item["anatomical_side"] = side
    item["part"] = _side_name(source_part, side)
    if clear_frame:
        item["image_location"] = None
        state["frame_location_selection_usable"] = False


def refine_complementary_bilateral_sets(
    payload: dict[str, Any],
    dw: dict[str, Any],
    sam: dict[str, Any],
    sam_path: Path,
) -> dict[str, Any]:
    """Recover bilateral sets whose only semantic difference is complementary frame location.

    Fusion 2.3.2 intentionally required identical image locations. That is too strict when
    Analyze emits two otherwise identical left/right limb records as ``left center`` and
    ``right center`` while DWPose/SAM independently establish both physical chains/hands.
    In that case the records are semantically interchangeable for captioning because target
    frame location is not caption authority. The frame locations are cleared before the
    records are re-anchored one-left/one-right.
    """
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    if not isinstance(fusion, dict):
        return out

    parts = [item for item in (fusion.get("qualified_body_parts") or []) if isinstance(item, dict)]
    pairs = _candidate_pairs(parts)
    audit: dict[str, Any] = {
        "schema_version": "bilateral-complementary-frame-audit-1.0",
        "source_fusion_schema": fusion.get("schema_version"),
        "pairs_considered": [],
        "pairs_applied": [],
        "policy": {
            "semantic_equivalence_excluding_frame_location_required": True,
            "frame_locations_must_be_identical_or_complementary": True,
            "complementary_frame_location_is_cleared_before_reanchoring": True,
            "bilateral_complete_dwpose_chains_required": True,
            "distal_arm_sets_require_two_qualified_observed_hands": True,
            "sam3d_reconstruction_alone_never_establishes_visibility": True,
        },
    }

    hand_support: tuple[bool, list[dict[str, Any]]] | None = None
    for family, left_index, right_index, frame_relation in pairs:
        left = parts[left_index]
        right = parts[right_index]
        record: dict[str, Any] = {
            "family": family,
            "left_index": left_index,
            "right_index": right_index,
            "frame_relation": frame_relation,
            "source_frame_locations": [left.get("image_location"), right.get("image_location")],
        }
        audit["pairs_considered"].append(record)

        if not _complete_bilateral_chains(dw, family):
            record["action"] = "unchanged"
            record["reason"] = "bilateral_complete_dwpose_chains_not_available"
            continue

        if family == "arm":
            distal = any(
                re.search(
                    r"\b(?:hand|wrist|finger|fingers)\b",
                    " ".join(
                        [
                            str(item.get("part") or ""),
                            *[str(value) for value in (item.get("visible_subparts") or [])],
                            str(item.get("geometry") or ""),
                            str(item.get("contact") or ""),
                        ]
                    ),
                    re.I,
                )
                for item in (left, right)
            )
            if distal:
                if hand_support is None:
                    hand_support = _bilateral_hand_support(dw, sam_path, sam)
                both_hands, entities = hand_support
                record["hand_entities"] = copy.deepcopy(entities)
                if not both_hands:
                    record["action"] = "unchanged"
                    record["reason"] = "two_qualified_observed_hand_entities_not_available"
                    continue
                authority = "dwpose_bilateral_complete_chains_and_hands_complementary_frame"
            else:
                authority = "dwpose_bilateral_complete_chains_complementary_frame"
        else:
            authority = "dwpose_bilateral_complete_chains_complementary_frame"

        reason = (
            "opposite-source semantic records are identical apart from same/complementary frame location; "
            "deterministic evidence establishes both physical chains, so the records form an unordered bilateral set"
        )
        clear_frame = frame_relation == "complementary"
        _apply_side(left, "left", authority, reason, clear_frame=clear_frame)
        _apply_side(right, "right", authority, reason, clear_frame=clear_frame)
        record.update(
            action="applied",
            authority=authority,
            qualified_sides=["left", "right"],
            frame_locations_cleared=clear_frame,
        )
        audit["pairs_applied"].append(copy.deepcopy(record))

    fusion["schema_version"] = "analysis-fusion-2.3.4"
    fusion["bilateral_complementary_frame_audit"] = audit
    fusion.setdefault("selection_policy", {})["bilateral_complementary_frame_sets"] = (
        "Otherwise-equivalent opposite-side limb records may differ only by complementary frame location; "
        "when both physical chains are deterministically observed, frame location is discarded and the pair is "
        "re-anchored one-left/one-right."
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-laterality-bilateral-refine-234",
        description="Fusion 2.3.4: recover deterministic bilateral sets missed only because Analyze used complementary frame locations.",
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
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.3" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    sam3d_dir = (args.sam3d_dir or (run_dir / "sam3d")).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "fusion-v2.3.4" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion-v2.3.3"), (dwpose_dir, "DWPose"), (sam3d_dir, "SAM3D")):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = missing = applied = 0
    records: list[dict[str, Any]] = []
    for fusion_path in sorted(fusion_dir.glob("*.fused_v2_3.json")):
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_path = output_dir / fusion_path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        dw_path = dwpose_dir / f"{key}.dwpose.json"
        sam_path = sam3d_dir / f"{key}.sam3d.json"
        if not dw_path.is_file() or not sam_path.is_file():
            missing += 1
            records.append({"image_key": key, "status": "missing_source"})
            continue

        refined = refine_complementary_bilateral_sets(_read(fusion_path), _read(dw_path), _read(sam_path), sam_path)
        _write(out_path, refined)
        written += 1
        audit = ((refined.get("fusion") or {}).get("bilateral_complementary_frame_audit") or {})
        count = len(audit.get("pairs_applied") or [])
        applied += count
        records.append(
            {
                "image_key": key,
                "status": "written",
                "pairs_considered": len(audit.get("pairs_considered") or []),
                "pairs_applied": count,
            }
        )

    index = {
        "schema_version": "analysis-fusion-2.3.4-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "source_fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "sam3d_dir": str(sam3d_dir),
        "output_dir": str(output_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_sources": missing,
        "bilateral_pairs_applied": applied,
        "records": records,
    }
    _write(output_dir / "laterality_bilateral_refine_234.index.json", index)
    print(f"Fusion-v2.3.4 output: {output_dir}")
    print(f"Written: {written}; reused: {skipped}; missing: {missing}; bilateral pairs applied: {applied}")
    return 0 if written or skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
