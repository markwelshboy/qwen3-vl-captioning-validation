from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import semantic_v3_compact_renderer_v01 as v01


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_compact_renderer_v02.txt"
ARTIFACT_VERSION = "semantic-v3-compact-renderer-0.2"
RUN_VERSION = "semantic-v3-compact-renderer-0.2-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-v0.2"
DEFAULT_MAX_TOKENS = 220

PROFILE_BUDGETS = v01.PROFILE_BUDGETS
GRAMMAR_PROFILES = v01.GRAMMAR_PROFILES

_PROFILE_RULES = {
    "compact": (
        "- Aim for about 70–80 words and NEVER exceed 90.\n"
        "- Prefer 2–3 sentences; use a fourth only if needed to keep ownership unambiguous.\n"
        "- Treat this as an identity-LoRA caption: one sentence may combine trigger + pose + clothing/action; "
        "use the remaining sentence(s) for framing and only the strongest scene anchor(s).\n"
        "- Do not attempt to preserve every accessory, object, material, or lighting detail."
    ),
    "medium": (
        "- Aim for about 115–125 words and NEVER exceed 140.\n"
        "- Prefer 3–5 sentences.\n"
        "- Preserve the full governed pose, main clothing/accessories, framing, main action/expression, and a small number "
        "of strong scene anchors; omit minor decorative and surface detail.\n"
        "- This is still selective, not a shortened copy of every sentence in the source."
    ),
}

_BASE_BUILD_RENDERER_INPUT = v01.build_renderer_input


def build_renderer_input(
    *,
    semantic_caption: str,
    pose: dict[str, Any],
    trigger: str,
    grammar_profile: str,
    profile: str,
    prompt_template: str,
) -> dict[str, Any]:
    value = _BASE_BUILD_RENDERER_INPUT(
        semantic_caption=semantic_caption,
        pose=pose,
        trigger=trigger,
        grammar_profile=grammar_profile,
        profile=profile,
        prompt_template=prompt_template,
    )
    value["renderer_prompt"] = value["renderer_prompt"].replace(
        "{{PROFILE_RULES}}",
        _PROFILE_RULES[profile],
    )
    value["profile_rules"] = _PROFILE_RULES[profile]
    return value


def _trigger_audit(text: str, trigger: str) -> dict[str, Any]:
    """v0.1 trigger checks, but allow a natural comma appositive after the name.

    `V3SUBJ, in a close-up portrait, faces ...` is grammatically bound to the
    trigger and should not fail merely because the name is followed by a comma.
    Detached generic replacements such as `V3SUBJ, a woman ...` remain invalid.
    """
    escaped = re.escape(trigger)
    exact_matches = list(re.finditer(rf"(?<!\w){escaped}(?!\w)", text))
    ci_matches = list(re.finditer(rf"(?<!\w){escaped}(?!\w)", text, re.IGNORECASE))

    starts_exact = bool(re.match(rf"^{escaped}(?=\s|,)", text))
    detached_trailing = bool(re.search(rf"[,;:]\s*{escaped}[.!?]?\s*$", text, re.IGNORECASE))
    generic_after_trigger = bool(
        re.match(
            rf"^{escaped}\s*[,;:\-]\s*(?:a|the)\s+(?:woman|man|person|subject)\b",
            text,
            re.IGNORECASE,
        )
        or re.match(
            rf"^{escaped}\s+(?:is|appears|looks)\s+(?:like\s+)?(?:a|the)\s+(?:woman|man|person|subject)\b",
            text,
            re.IGNORECASE,
        )
    )

    warnings: list[str] = []
    if not exact_matches:
        warnings.append("trigger_missing")
    if len(ci_matches) != len(exact_matches):
        warnings.append("trigger_case_changed")
    if not starts_exact:
        warnings.append("trigger_not_direct_first_subject")
    if detached_trailing:
        warnings.append("detached_trailing_trigger")
    if generic_after_trigger:
        warnings.append("generic_primary_subject_after_trigger")

    return {
        "exact_trigger_count": len(exact_matches),
        "case_insensitive_trigger_count": len(ci_matches),
        "starts_with_exact_trigger_as_direct_token": starts_exact,
        "detached_trailing_trigger": detached_trailing,
        "generic_primary_subject_after_trigger": generic_after_trigger,
        "warnings": warnings,
    }


def main() -> int:
    # Reuse v0.1 mechanics and audits. v0.2 changes only the rendering contract,
    # natural trigger-appositive validation, output provenance, and a smaller
    # default generation ceiling. The frozen semantic source remains untouched.
    v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    v01.RUN_VERSION = RUN_VERSION
    v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    v01.DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS
    v01.build_renderer_input = build_renderer_input
    v01._trigger_audit = _trigger_audit
    return v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
