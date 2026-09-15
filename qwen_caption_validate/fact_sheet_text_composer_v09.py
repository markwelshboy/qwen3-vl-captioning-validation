from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v08 as phase57

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v06.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.9"
SCHEMA_VERSION = "fact-sheet-text-composer-0.9"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

_SUPPORT_CONTACT_LANGUAGE_RE = re.compile(
    r"\b(?:support(?:ed|ing)?|weight[- ]?bearing|weight\s+(?:on|over)|"
    r"contact\s+with|in\s+contact\s+with|planted|feet?\s+flat)\b",
    re.I,
)
_SUPPORT_CONTACT_EVIDENCE_RE = re.compile(
    r"\b(?:support(?:ed|ing)?|weight[- ]?bearing|weight\s+(?:on|over)|"
    r"contact\s+with|in\s+contact\s+with|planted|feet?\s+flat|"
    r"resting\s+on|rests?\s+on|back\s+against)\b",
    re.I,
)
_COUNTER_ROTATION_RE = re.compile(
    r"\bcounter[- ]?rotat(?:e|es|ed|ing|ion|ional)\b",
    re.I,
)


def _support_contact_authorized(projection: dict[str, Any]) -> bool:
    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    if body.get("global_support_shape"):
        return True
    configuration = body.get("configuration") if isinstance(body.get("configuration"), list) else []
    return any(
        isinstance(text, str) and _SUPPORT_CONTACT_EVIDENCE_RE.search(text)
        for text in configuration
    )


def _articulated_relationship(projection: dict[str, Any]) -> str | None:
    authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    torso = body.get("torso_orientation") if isinstance(body.get("torso_orientation"), dict) else {}
    relation = torso.get("articulated_relative_orientation") if isinstance(torso.get("articulated_relative_orientation"), dict) else {}
    value = relation.get("relationship")
    return str(value) if value else None


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = phase57._caption_audit(caption, projection)
    violations = list(audit.get("violations") or [])

    support_language_used = bool(_SUPPORT_CONTACT_LANGUAGE_RE.search(caption))
    support_authorized = _support_contact_authorized(projection)
    if support_language_used and not support_authorized:
        violations.append("support_contact_language_without_authority")

    relationship = _articulated_relationship(projection)
    counter_rotation_used = bool(_COUNTER_ROTATION_RE.search(caption))
    if counter_rotation_used and relationship != "counter_rotated":
        violations.append("counter_rotation_language_without_counter_rotated_geometry")

    audit["violations"] = sorted(set(violations))
    audit["support_contact_language_used"] = support_language_used
    audit["support_contact_authorized"] = support_authorized
    audit["counter_rotation_language_used"] = counter_rotation_used
    audit["articulated_relationship"] = relationship or audit.get("articulated_relationship")
    return audit


def _retry_prompt(original_prompt: str, caption: str, violations: list[str]) -> str:
    # Reuse the surgical Phase-5.6 repair instructions, then add a universal
    # no-new-facts guard. A retry is allowed to fix wording, not to compensate
    # for a removed clause by inventing biomechanics or another restricted fact.
    prompt = phase57.phase56._retry_prompt(original_prompt, caption, violations)
    extra: list[str] = [
        "Do not introduce any new support/contact, weight-bearing, planted-foot, anatomical-side, gaze, head-direction, or torso-direction claim while revising.",
        "A statement that support/contact is absent or not visible is still a support/contact claim and is forbidden unless explicitly supplied by AUTHORITATIVE FACTS.",
    ]
    if "support_contact_language_without_authority" in violations:
        extra.append(
            "Remove support/contact/weight-bearing/planted-state language entirely; do not replace it with a negative disclaimer such as 'without visible support' or 'no contact is visible'."
        )
    if "counter_rotation_language_without_counter_rotated_geometry" in violations:
        extra.append(
            "Remove 'counter-rotation' wording. Use the supplied articulated relationship literally; same-direction segments with the upper torso closer to frontal are not counter-rotated."
        )
    return prompt + "\nREVISION SAFETY:\n- " + "\n- ".join(extra)


def main() -> int:
    # v08 owns the validated direct projection path. Extend only the caption
    # audit/retry boundary; geometry and evidence projection remain unchanged.
    phase57._caption_audit = _caption_audit
    phase57.phase56._retry_prompt = _retry_prompt
    phase57.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase57.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase57.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase57.SCHEMA_VERSION = SCHEMA_VERSION
    phase57.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase57.main()


if __name__ == "__main__":
    raise SystemExit(main())
