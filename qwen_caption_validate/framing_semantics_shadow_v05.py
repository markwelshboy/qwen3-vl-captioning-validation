from __future__ import annotations

from pathlib import Path
from typing import Any

from . import framing_semantics_shadow_v04 as v04

SCHEMA_VERSION = "framing-semantics-shadow-0.5"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "framing-semantics-shadow-v0.5"

_BASE_EVALUATE = v04.evaluate
_CLOSE_FAMILY = {"medium_close_up", "close_up", "extreme_close_up"}


def _withhold_ambiguous_scale(
    scale: dict[str, Any],
    span: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "status": "withheld",
        "label": None,
        "composer_text": None,
        "basis": [
            "standard_shot_scale_withheld_for_crop_ambiguity",
            *reasons,
            f"canonical_span={span.get('upper_anchor')}->{span.get('lower_anchor')}",
        ],
        "pose_family": scale.get("pose_family"),
        "rejected_candidate": scale,
        "note": (
            "Conventional shot scale is optional. It is withheld when adjacent or non-contiguous "
            "lower-body observations make a close-family portrait label less trustworthy than the "
            "canonical anatomical span."
        ),
    }


def _scale_ambiguity_reasons(span: dict[str, Any], scale: dict[str, Any]) -> list[str]:
    if scale.get("status") != "candidate" or scale.get("label") not in _CLOSE_FAMILY:
        return []

    reasons: list[str] = []
    noncontiguous = span.get("noncontiguous_observations")
    if isinstance(noncontiguous, list) and noncontiguous:
        reasons.append("noncontiguous_anatomical_observations=" + ",".join(str(x) for x in noncontiguous))

    # For a head->shoulders core span, even one observed hip means the crop may
    # extend materially below the conventional close-family region. Prefer the
    # literal anatomical span over forcing MCU/CU terminology in that case.
    if (
        span.get("upper_anchor") == "head"
        and span.get("lower_anchor") == "shoulders"
        and span.get("lower_partial") == "hips"
    ):
        reasons.append("partial_hip_observation_below_head_shoulders_core")

    return reasons


def _opening(span: dict[str, Any], scale: dict[str, Any]) -> str | None:
    span_text = v04.v01._clean(span.get("composer_text"))
    scale_text = v04.v01._clean(scale.get("composer_text"))
    if scale_text:
        article = "an" if scale_text.startswith("extreme ") else "a"
        if span_text:
            return f"[[trigger]] is shown in {article} {scale_text}, {span_text}"
        return f"[[trigger]] is shown in {article} {scale_text}"
    if span_text:
        return "[[trigger]] is " + span_text if span_text.startswith("framed ") else "[[trigger]] is shown " + span_text
    return None


def evaluate(
    policy: dict[str, Any],
    fact: dict[str, Any] | None = None,
    uniface: dict[str, Any] | None = None,
    *,
    image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    out = _BASE_EVALUATE(policy, fact, uniface, image_size=image_size)
    span = out.get("anatomical_span") if isinstance(out.get("anatomical_span"), dict) else {}
    scale = out.get("standard_shot_scale") if isinstance(out.get("standard_shot_scale"), dict) else {}

    reasons = _scale_ambiguity_reasons(span, scale)
    if reasons:
        scale = _withhold_ambiguous_scale(scale, span, reasons)
        out["standard_shot_scale"] = scale

    out["proposed_opening_template"] = _opening(span, scale)
    out["schema_version"] = SCHEMA_VERSION

    invariants = out.get("invariants") if isinstance(out.get("invariants"), dict) else {}
    invariants.update(
        close_family_scale_abstains_on_noncontiguous_crop_evidence=True,
        head_shoulders_close_family_abstains_when_hip_is_partially_observed=True,
        standard_scale_abstention_preserves_anatomical_span=True,
        opening_uses_correct_indefinite_article=True,
    )
    out["invariants"] = invariants
    return out


def main() -> int:
    # Reuse v0.4 discovery, UniFace loading, writer and census machinery. Only
    # replace evaluation/output identity for this conservative successor.
    old_evaluate = v04.evaluate
    old_schema = v04.SCHEMA_VERSION
    old_output = v04.DEFAULT_OUTPUT_SUBDIR
    v04.evaluate = evaluate
    v04.SCHEMA_VERSION = SCHEMA_VERSION
    v04.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    try:
        return v04.main()
    finally:
        v04.evaluate = old_evaluate
        v04.SCHEMA_VERSION = old_schema
        v04.DEFAULT_OUTPUT_SUBDIR = old_output


if __name__ == "__main__":
    raise SystemExit(main())
