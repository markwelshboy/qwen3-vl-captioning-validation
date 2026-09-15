from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v07 as phase56

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v06.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.8"
SCHEMA_VERSION = "fact-sheet-text-composer-0.8"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

# Scope signed torso-direction detection to the torso clause itself.  Do not let
# a later head/gaze clause in the same sentence make an unsigned torso look
# signed (e.g. "her torso is oblique ... and her head turns toward frame left").
_TORSO_SIGNED_FRAME_RE = re.compile(
    r"\b(?:torso|upper\s+torso)\b"
    r"(?:(?!\b(?:head|face|neck|gaze|eyes?)\b)[^.!?]){0,140}"
    r"\b(?:toward|to)\s+frame\s+(left|right)\b",
    re.I,
)


def _restore_validated_projection_helpers() -> None:
    """Restore the Phase-5.5 projection stack before using it directly.

    v06 normally reaches the v03 torso/gaze projection helpers through its
    chained main() calls.  v07 introduced a custom main loop and called the
    projection function directly, bypassing those runtime helper assignments.
    That silently regressed articulated torso projection to the legacy v01
    torso helper.  Bind the validated helpers explicitly so direct projection
    is equivalent to the v06 execution path.
    """
    phase56.engine._torso_fact = phase56.phase55._torso_fact
    phase56.engine._gaze_fact = phase56.phase55._gaze_fact


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _restore_validated_projection_helpers()
    projection, audit = phase56.phase55._projection(
        sheet,
        trigger_token=trigger_token,
        subject_class=subject_class,
    )

    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    relationship = phase56._articulated_relative_orientation(torso)
    if relationship:
        torso["articulated_relative_orientation"] = relationship
        body["torso_orientation"] = torso
        authoritative["body"] = body
        projection["authoritative_facts"] = authoritative

    audit["validated_torso_projection_helpers_restored"] = True
    audit["articulated_relative_orientation_projected"] = relationship is not None
    audit["articulated_relative_orientation"] = relationship
    return projection, audit


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = phase56.phase55._caption_audit(caption, projection)
    violations = list(audit.get("violations") or [])

    allowed_directions = phase56._torso_authorized_frame_directions(projection)
    used_directions = {f"frame_{m.group(1).lower()}" for m in _TORSO_SIGNED_FRAME_RE.finditer(caption)}
    unauthorized = sorted(used_directions - allowed_directions)
    if unauthorized:
        violations.append("unauthorized_torso_turn_direction:" + ",".join(unauthorized))

    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    relation = torso.get("articulated_relative_orientation") if isinstance(torso.get("articulated_relative_orientation"), dict) else {}
    relationship = relation.get("relationship")
    if relationship == "upper_torso_closer_to_frontal" and phase56._UPPER_MORE_TURNED_RE.search(caption):
        violations.append("articulated_torso_relationship_reversed")
    if relationship == "upper_torso_more_turned" and phase56._UPPER_LESS_TURNED_RE.search(caption):
        violations.append("articulated_torso_relationship_reversed")

    audit["violations"] = sorted(set(violations))
    audit["torso_authorized_frame_directions"] = sorted(allowed_directions)
    audit["torso_used_frame_directions"] = sorted(used_directions)
    audit["articulated_relationship"] = relationship
    audit["torso_direction_audit_scoped_to_torso_clause"] = True
    return audit


def main() -> int:
    _restore_validated_projection_helpers()
    phase56._projection = _projection
    phase56._caption_audit = _caption_audit
    phase56.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase56.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase56.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase56.SCHEMA_VERSION = SCHEMA_VERSION
    phase56.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase56.main()


if __name__ == "__main__":
    raise SystemExit(main())
