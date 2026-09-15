from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v04 as phase53

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v04.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.5"
SCHEMA_VERSION = "fact-sheet-text-composer-0.5"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

_BASE_PROJECTION = phase53._projection

# Re-export validated Phase-5.3 helpers for tests and downstream retry tooling.
_torso_fact = phase53._torso_fact
_gaze_fact = phase53._gaze_fact
_caption_audit = phase53._caption_audit

# The base composer already sanitizes obvious stand/sit/lean wording from the
# holistic gestalt string.  However, a phrase such as "mostly upright over her
# left leg" can survive after the word "standing" is removed and then compete
# with specialist-authoritative pose/support facts.  Holistic context is only a
# non-authoritative fluency hint, so default-deny the whole hint if any
# specialist-owned body mechanics remain after the legacy sanitizer.
_RESIDUAL_BODY_MECHANICS_RE = re.compile(
    r"\b(?:upright|weight[- ]?bearing|weight\s+on|support(?:ing)?\s+leg|"
    r"planted|lifted|raised|lowered|foot|feet|ankle|knee|knees|leg|legs|hip|hips|"
    r"torso|upper\s+body|shoulder|shoulders|arm|arms|hand|hands|head|neck)\b",
    re.I,
)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _canonical_broad_pose(body: dict[str, Any]) -> str | None:
    """Return the post-specialist broad pose that the composer may publish."""
    adjudication = body.get("broad_pose_adjudication") if isinstance(body.get("broad_pose_adjudication"), dict) else {}
    if adjudication.get("composer_authoritative"):
        canonical = _clean(adjudication.get("canonical_pose_text"))
        if canonical:
            return canonical

    pose = body.get("pose_candidate") if isinstance(body.get("pose_candidate"), dict) else {}
    status = str(pose.get("promotion_status") or "").strip()
    accepted = status in {"candidate", "accepted", "resolved"} or status.startswith("accepted_")
    if not accepted:
        return None

    for key in ("composer_text", "normalized_text", "text"):
        text = _clean(pose.get(key))
        if text:
            return text
    return None


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # v04 ultimately delegates to the v01 projection engine. Its legacy
    # broad-pose helper read raw `text` and rejected `accepted_*` specialist
    # statuses. Patch only that helper for this composer generation so the
    # canonical specialist pose actually reaches the text-only model.
    phase53.engine._broad_pose = _canonical_broad_pose

    projection, audit = _BASE_PROJECTION(
        sheet,
        trigger_token=trigger_token,
        subject_class=subject_class,
    )

    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    source_body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    projected_body = (
        projection.get("authoritative_facts", {}).get("body", {})
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )

    # The legacy DWPose `support_geometry.overall_shape` is useful diagnostic
    # context for local relation work, but it is not a reliable broad posture or
    # ground-contact classifier. Keep it in the source fact sheet only.
    support_geometry = source_body.get("support_geometry")
    audit["support_geometry_present_in_source"] = isinstance(support_geometry, dict)
    audit["support_geometry_withheld_from_composer"] = isinstance(support_geometry, dict)

    # Prevent non-authoritative gestalt prose from re-introducing body mechanics
    # that specialists deliberately removed or replaced.  Scene/appearance/
    # object facts remain available through their typed authoritative fields.
    holistic = _clean(projection.get("holistic_context_non_authoritative"))
    residual_body_mechanics = bool(holistic and _RESIDUAL_BODY_MECHANICS_RE.search(holistic))
    if residual_body_mechanics:
        projection.pop("holistic_context_non_authoritative", None)
    audit["holistic_context_withheld_for_body_mechanics"] = residual_body_mechanics

    audit["canonical_broad_pose_projected"] = (
        _clean(projected_body.get("broad_pose")) if isinstance(projected_body, dict) else None
    )
    return projection, audit


def main() -> int:
    # Keep every validated Phase-5.3 behavior (trigger/pronouns, gaze semantics,
    # signed torso direction, specialist laterality) while ensuring canonical
    # specialist pose reaches the text-only model and non-authoritative gestalt
    # cannot re-introduce rejected body mechanics.
    phase53.engine._broad_pose = _canonical_broad_pose
    phase53._projection = _projection
    phase53.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase53.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase53.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase53.SCHEMA_VERSION = SCHEMA_VERSION
    phase53.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase53.main()


if __name__ == "__main__":
    raise SystemExit(main())
