from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

from .laterality_geometry import _connectivity, _hand_entities, _load_sam2d, _read, _target_points, _write
from .laterality_match import _family, _raw_side
from .runner import model_slug, resolve_model_id


def _norm_text(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ")
    text = re.sub(r"\b(?:left|right)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:_-")


def _source_side(item: dict[str, Any]) -> str | None:
    state = item.get("fusion_v2") or {}
    for value in (
        state.get("source_anatomical_side"),
        _raw_side(item.get("source_part")),
        item.get("anatomical_side"),
    ):
        side = str(value or "").lower()
        if side in {"left", "right"}:
            return side
    return None


def _semantic_signature(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _family(item),
        str(item.get("visibility") or ""),
        tuple(sorted(_norm_text(value) for value in (item.get("visible_subparts") or []))),
        _norm_text(item.get("geometry")),
        _norm_text(item.get("contact")),
        _norm_text(item.get("support")),
        _norm_text(item.get("foreshortening")),
        # Deliberately keep frame location side-sensitive. Bilateral equivalence is
        # only safe when the semantic records do not themselves distinguish two
        # different image regions.
        str(item.get("image_location") or "").lower().replace("_", " ").strip(),
    )


def _eligible_target_item(item: dict[str, Any]) -> bool:
    state = item.get("fusion_v2") or {}
    owner = state.get("qualified_ownership") or item.get("ownership")
    return bool(state.get("selection_usable")) and owner == "target" and _family(item) in {"arm", "leg"}


def _bilateral_equivalent_pairs(parts: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
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
        if _semantic_signature(left) != _semantic_signature(right):
            continue
        out.append((family, left_index, right_index))
    return out


def _complete_bilateral_chains(dw: dict[str, Any], family: str) -> bool:
    connectivity = _connectivity(dw)
    return all(bool((connectivity.get(f"{side}_{family}") or {}).get("complete")) for side in ("left", "right"))


def _bilateral_hand_support(dw: dict[str, Any], sam_path: Path, sam: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    points = _target_points(dw)
    sam2d = _load_sam2d(sam_path, sam)
    entities = _hand_entities(dw, points, sam2d)
    sides = {
        str(entity.get("qualified_side"))
        for entity in entities
        if entity.get("qualified_side") in {"left", "right"}
    }
    return sides == {"left", "right"}, entities


def _set_side_from_bilateral_set(item: dict[str, Any], side: str, authority: str, reason: str) -> None:
    state = item.setdefault("fusion_v2", {})
    state.setdefault("source_anatomical_side", _source_side(item) or "unknown")
    state["qualified_anatomical_side"] = side
    state["laterality_selection_usable"] = True
    state["laterality_authority"] = authority
    state.setdefault("laterality_reasons", []).append(
        f"Fusion-v2.3.2 qualifies side={side}: {reason}"
    )


def guard_bilateral_sets(
    payload: dict[str, Any],
    dw: dict[str, Any],
    sam: dict[str, Any],
    sam_path: Path,
) -> dict[str, Any]:
    """Repair greedy laterality collisions for semantically equivalent bilateral records.

    If Analyze emitted two opposite-side records with identical semantics and no
    distinguishing frame location, the records are treated as an unordered bilateral
    set. When deterministic evidence establishes both physical chains, one record is
    anchored to anatomical left and the other to anatomical right. This avoids mapping
    both semantic records onto whichever chain happens to lie closer to image center.
    """
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    if not isinstance(fusion, dict):
        return out

    parts = [item for item in (fusion.get("qualified_body_parts") or []) if isinstance(item, dict)]
    pairs = _bilateral_equivalent_pairs(parts)
    audit: dict[str, Any] = {
        "schema_version": "bilateral-equivalence-audit-1.0",
        "pairs_considered": [],
        "pairs_applied": [],
        "policy": {
            "semantic_equivalence_required": True,
            "opposite_source_sides_required": True,
            "bilateral_complete_dwpose_chains_required": True,
            "distal_arm_sets_require_two_qualified_observed_hands": True,
            "sam3d_reconstruction_alone_never_establishes_visibility": True,
        },
    }

    hand_support: tuple[bool, list[dict[str, Any]]] | None = None
    for family, left_index, right_index in pairs:
        left = parts[left_index]
        right = parts[right_index]
        record = {
            "family": family,
            "left_index": left_index,
            "right_index": right_index,
            "signature": list(_semantic_signature(left)),
        }
        audit["pairs_considered"].append(copy.deepcopy(record))

        if not _complete_bilateral_chains(dw, family):
            record["action"] = "unchanged"
            record["reason"] = "bilateral_complete_dwpose_chains_not_available"
            continue

        if family == "arm":
            distal = any(
                re.search(r"\b(?:hand|wrist|finger|fingers)\b", " ".join([
                    str(item.get("part") or ""),
                    *[str(value) for value in (item.get("visible_subparts") or [])],
                    str(item.get("geometry") or ""),
                    str(item.get("contact") or ""),
                ]), re.I)
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
                authority = "dwpose_bilateral_complete_chains_and_hands"
            else:
                authority = "dwpose_bilateral_complete_chains"
        else:
            authority = "dwpose_bilateral_complete_chains"

        reason = (
            "equivalent opposite-side semantic records are an unordered bilateral set; "
            "deterministic evidence establishes both physical chains"
        )
        _set_side_from_bilateral_set(left, "left", authority, reason)
        _set_side_from_bilateral_set(right, "right", authority, reason)
        record.update(
            action="applied",
            authority=authority,
            reason=reason,
            qualified_sides=["left", "right"],
        )
        audit["pairs_applied"].append(copy.deepcopy(record))

    fusion["schema_version"] = "analysis-fusion-2.3.2"
    fusion["bilateral_equivalence_audit"] = audit
    fusion.setdefault("selection_policy", {})["bilateral_equivalent_sets"] = (
        "Equivalent left/right semantic records are treated as an unordered bilateral set; "
        "complete DWPose chains, and two observed hand entities for distal arm sets, prevent "
        "greedy same-side collisions."
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-laterality-bilateral-guard",
        description="Guard Fusion-v2.3.1 against greedy same-side collisions in equivalent bilateral semantic records.",
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
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.1" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    sam3d_dir = (args.sam3d_dir or (run_dir / "sam3d")).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "fusion-v2.3.2" / slug)).expanduser().resolve()

    for path, label in ((fusion_dir, "Fusion-v2.3.1"), (dwpose_dir, "DWPose"), (sam3d_dir, "SAM3D")):
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

        refined = guard_bilateral_sets(_read(fusion_path), _read(dw_path), _read(sam_path), sam_path)
        _write(out_path, refined)
        written += 1
        audit = ((refined.get("fusion") or {}).get("bilateral_equivalence_audit") or {})
        count = len(audit.get("pairs_applied") or [])
        applied += count
        records.append({
            "image_key": key,
            "status": "written",
            "pairs_considered": len(audit.get("pairs_considered") or []),
            "pairs_applied": count,
        })

    index = {
        "schema_version": "analysis-fusion-2.3.2-run",
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
    _write(output_dir / "laterality_bilateral_guard.index.json", index)
    print(f"Fusion-v2.3.2 output: {output_dir}")
    print(f"Written: {written}; reused: {skipped}; missing: {missing}; bilateral pairs applied: {applied}")
    return 0 if written or skipped else 2


if __name__ == "__main__":
    raise SystemExit(main())
