from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_134 import build_caption_projection as _build_134
from .caption_projection_134 import lint_caption as _lint_134

# These relations are bound to a particular member of a bilateral anatomical
# pair. If Fusion has corrected the semantic record from one anatomical side to
# the other, the raw relation cannot safely migrate with that correction unless
# another deterministic stage independently re-qualifies it (for example the
# synthetic signed-depth shoulder record added by Fusion 2.3.3).
_SIDE_BOUND_DEPTH_RE = re.compile(
    r"\b(?:closer|nearer|farther|further|forward|retracted|behind|in\s+front|depth[- ]stagger(?:ed|ing)?)\b",
    re.IGNORECASE,
)
_PROXIMAL_PAIR_RE = re.compile(r"\b(?:shoulder|hip|pelvis)\b", re.IGNORECASE)


def _fusion_root(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("fusion") if isinstance(payload.get("fusion"), dict) else payload
    return value if isinstance(value, dict) else {}


def _strip_side_bound_geometry(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    # Preserve neutral clauses such as "shoulders level" while dropping only
    # clauses whose meaning depends on which member of the bilateral pair the
    # original semantic record referred to.
    clauses = [clause.strip() for clause in re.split(r"[;,]", value) if clause.strip()]
    kept = [clause for clause in clauses if not _SIDE_BOUND_DEPTH_RE.search(clause)]
    return "; ".join(kept) or None


def _withhold_migrated_side_geometry(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out = copy.deepcopy(payload)
    fusion = _fusion_root(out)
    blocked: list[dict[str, Any]] = []

    for index, item in enumerate(fusion.get("qualified_body_parts") or []):
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        source_side = str(state.get("source_anatomical_side") or "unknown").lower()
        qualified_side = str(state.get("qualified_anatomical_side") or "unknown").lower()
        if source_side not in {"left", "right"} or qualified_side not in {"left", "right"}:
            continue
        if source_side == qualified_side:
            continue
        label = " ".join(
            [
                str(item.get("part") or ""),
                str(item.get("source_part") or ""),
                *[str(value) for value in (item.get("visible_subparts") or [])],
            ]
        )
        if not _PROXIMAL_PAIR_RE.search(label):
            continue
        before = item.get("geometry")
        after = _strip_side_bound_geometry(before)
        if before == after:
            continue
        item["geometry"] = after
        state["side_bound_geometry_selection_usable"] = False
        state.setdefault("laterality_reasons", []).append(
            "Projection 1.3.5 withholds side-bound depth geometry after anatomical-side correction; relation requires independent re-qualification"
        )
        blocked.append(
            {
                "path": f"fusion.qualified_body_parts[{index}].geometry",
                "reason": "side_bound_geometry_cannot_migrate_across_laterality_correction",
                "source_anatomical_side": source_side,
                "qualified_anatomical_side": qualified_side,
                "source_geometry": before,
                "projected_geometry": after,
            }
        )
    return out, blocked


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sanitized, blocked = _withhold_migrated_side_geometry(fused_payload)
    evidence, audit = _build_134(sanitized, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.3.5"
    projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    if isinstance(projection, dict):
        projection["schema_version"] = "caption-projection-audit-1.3.5"
        projection.setdefault("blocked", []).extend(blocked)
        if blocked:
            projection.setdefault("notes", []).append(
                "Corrected anatomical laterality does not transfer source-side depth relations; only independently re-qualified signed geometry may restore them."
            )
    return evidence, audit


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_134(caption, evidence))
    result["schema_version"] = "caption-authority-lint-1.3.5"
    return result
