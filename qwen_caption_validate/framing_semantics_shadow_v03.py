from __future__ import annotations

from pathlib import Path
from typing import Any

from . import framing_semantics_shadow_v01 as v01
from . import framing_semantics_shadow_v02 as v02

SCHEMA_VERSION = "framing-semantics-shadow-0.3"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "framing-semantics-shadow-v0.3"

_BASE_SHOT_SCALE_CANDIDATE = v01._shot_scale_candidate

_ALLOWED_BY_SPAN = {
    ("head", "head"): {"extreme_close_up", "close_up"},
    ("head", "shoulders"): {"close_up", "medium_close_up"},
    ("head", "hips"): {"medium"},
    ("head", "knees"): {"medium_wide"},
    ("head", "ankles"): {"full_body"},
}


def _withheld_for_span(raw: dict[str, Any], span: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "withheld",
        "label": None,
        "composer_text": None,
        "basis": [
            "photographic_scale_candidate_not_coherent_with_canonical_anatomical_span",
            f"canonical_span={span.get('upper_anchor')}->{span.get('lower_anchor')}",
            f"rejected_candidate={raw.get('label')}",
        ],
        "pose_family": raw.get("pose_family"),
        "rejected_candidate": raw,
        "note": (
            "Conventional shot scale withheld because it was triggered by non-contiguous/outlier tier evidence. "
            "The canonical anatomical span is the crop authority."
        ),
    }


def _shot_scale_candidate(
    tiers: dict[str, dict[str, Any]],
    span: dict[str, Any],
    bbox: dict[str, float | None],
    *,
    authoritative_pose: str | None,
) -> dict[str, Any]:
    raw = _BASE_SHOT_SCALE_CANDIDATE(
        tiers,
        span,
        bbox,
        authoritative_pose=authoritative_pose,
    )
    if raw.get("status") != "candidate":
        return raw

    key = (str(span.get("upper_anchor") or ""), str(span.get("lower_anchor") or ""))
    allowed = _ALLOWED_BY_SPAN.get(key, set())
    label = str(raw.get("label") or "")
    if label not in allowed:
        return _withheld_for_span(raw, span)

    out = dict(raw)
    out["span_coherent"] = True
    out["canonical_span"] = {"upper": key[0], "lower": key[1]}
    return out


def evaluate(policy: dict[str, Any], fact: dict[str, Any] | None = None) -> dict[str, Any]:
    # v02.evaluate ultimately calls v01.evaluate, whose global shot-scale
    # function is intentionally replaceable for this shadow successor.
    original = v01._shot_scale_candidate
    v01._shot_scale_candidate = _shot_scale_candidate
    try:
        out = v02.evaluate(policy, fact)
    finally:
        v01._shot_scale_candidate = original

    out["schema_version"] = SCHEMA_VERSION
    invariants = out.get("invariants") if isinstance(out.get("invariants"), dict) else {}
    invariants.update(
        standard_shot_scale_must_be_coherent_with_canonical_anatomical_span=True,
        noncontiguous_joint_tiers_cannot_create_standard_shot_scale=True,
        standard_shot_scale_remains_optional=True,
    )
    out["invariants"] = invariants
    return out


def main() -> int:
    # Reuse the v0.1 writer/discovery path while retaining the v0.2 corrected
    # broad-pose/local-configuration routing gate and adding only scale/span
    # coherence.
    old_evaluate = v01.evaluate
    old_schema = v01.SCHEMA_VERSION
    old_output = v01.DEFAULT_OUTPUT_SUBDIR
    v01.evaluate = evaluate
    v01.SCHEMA_VERSION = SCHEMA_VERSION
    v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    try:
        return v01.main()
    finally:
        v01.evaluate = old_evaluate
        v01.SCHEMA_VERSION = old_schema
        v01.DEFAULT_OUTPUT_SUBDIR = old_output


if __name__ == "__main__":
    raise SystemExit(main())
