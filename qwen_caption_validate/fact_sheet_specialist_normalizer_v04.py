from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import fact_sheet_specialist_normalizer_v03 as base
from .gaze_caption_semantics_v01 import build_gaze_caption_semantics

SCHEMA_VERSION = "caption-fact-sheet-0.2.3"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.3"


def _apply_gaze_caption_semantics(sheet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(sheet)
    warnings: list[str] = []
    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    gaze = facts.get("gaze") if isinstance(facts.get("gaze"), dict) else {}
    if not gaze:
        return out, warnings

    if not gaze.get("available") or not gaze.get("publishable"):
        gaze["caption_semantics"] = {
            "available": bool(gaze.get("available")),
            "publishable": False,
            "reason": "raw_gaze_not_publishable",
        }
        return out, warnings

    semantics = build_gaze_caption_semantics(gaze)
    gaze["caption_semantics"] = semantics
    if not semantics.get("publishable"):
        warnings.append("gaze_measurement_retained_but_caption_semantics_suppressed")
    horizontal = semantics.get("horizontal") if isinstance(semantics.get("horizontal"), dict) else {}
    if horizontal.get("semantic_class") in {"near_center", "mild_lateral_uncorroborated"}:
        warnings.append("gaze_horizontal_below_caption_salience")
    return out, warnings


def _apply_phase4b3(sheet: dict[str, Any]) -> dict[str, Any]:
    out, warnings = _apply_gaze_caption_semantics(sheet)
    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    existing = [str(x) for x in (audit.get("warnings") or []) if x]
    audit["warnings"] = sorted(set(existing + warnings))
    audit["phase"] = "4B.3"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        gaze_observability_is_separate_from_caption_salience=True,
        raw_gaze_measurement_is_preserved=True,
        near_center_gaze_is_not_forced_into_lateral_caption_language=True,
        mild_single_model_lateral_gaze_is_diagnostic_not_caption_authority=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


_BASE_BUILD_FACT_SHEET = base._build_fact_sheet


def _build_fact_sheet(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _apply_phase4b3(_BASE_BUILD_FACT_SHEET(*args, **kwargs))


def main() -> int:
    # Reuse the stable 4B.2 discovery/build path and add only a caption-facing
    # gaze semantic projection. Raw head/gaze specialist evidence remains intact.
    base._build_fact_sheet = _build_fact_sheet
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
