from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v09 as phase58

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v06.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.10"
SCHEMA_VERSION = "fact-sheet-text-composer-0.10"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

BASE_CROUCHED_STANCE = "holds a crouched stance"
QUALIFIED_CROUCHED_STANCE = "holds a crouched stance with her hips slightly lowered"

# Freeze the validated Phase-5.8 delegates before main() patches the inherited
# execution stack.
_BASE_PROJECTION = phase58.phase57._projection
_BASE_CAPTION_AUDIT = phase58._caption_audit
_BASE_RETRY_PROMPT = phase58._retry_prompt

_CROUCHED_STANCE_RE = re.compile(r"\bcrouched\s+stance\b", re.I)
_HIPS_SLIGHTLY_LOWERED_RE = re.compile(
    r"\bhips?\s+(?:are\s+)?slightly\s+lowered\b|"
    r"\bslightly\s+lowered\s+hips?\b",
    re.I,
)
_HIPS_LOWERED_ANY_RE = re.compile(
    r"\bhips?\s+(?:are\s+)?(?:slightly\s+)?lowered\b|"
    r"\b(?:slightly\s+)?lowered\s+hips?\b",
    re.I,
)
# Match actual depth-strengthening language, but do not confuse an unrelated
# later phrase such as "low-angle view" with crouch depth.
_DEEP_OR_LOW_CROUCH_RE = re.compile(
    r"\b(?:deep|low)\s+(?:standing\s+)?crouch(?:ed|ing)?(?:\s+stance)?\b|"
    r"\bcrouch(?:es|ed|ing)?\s+(?:very\s+)?(?:deeply|low)\b",
    re.I,
)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _source_depth_fact(sheet: dict[str, Any]) -> dict[str, Any] | None:
    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    depth = body.get("crouched_stance_depth") if isinstance(body.get("crouched_stance_depth"), dict) else {}
    adjudication = (
        body.get("crouched_stance_depth_adjudication")
        if isinstance(body.get("crouched_stance_depth_adjudication"), dict)
        else {}
    )

    status = str(depth.get("promotion_status") or "")
    if not status.startswith("accepted_specialist_"):
        return None
    if not adjudication.get("composer_authoritative"):
        return None
    if depth.get("classification") != "moderate_lowering":
        return None
    if depth.get("semantic_relation") != "hips_slightly_lowered":
        return None

    return {
        "classification": "moderate_lowering",
        "semantic_relation": "hips_slightly_lowered",
        "composer_text": "hips slightly lowered",
    }


def _surface_crouched_pose(*, depth_authorized: bool) -> str:
    return QUALIFIED_CROUCHED_STANCE if depth_authorized else BASE_CROUCHED_STANCE


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projection, audit = _BASE_PROJECTION(
        sheet,
        trigger_token=trigger_token,
        subject_class=subject_class,
    )

    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
    canonical_pose = _clean(body.get("broad_pose"))
    depth = _source_depth_fact(sheet)

    if canonical_pose and canonical_pose.casefold() == "crouching":
        surface = _surface_crouched_pose(depth_authorized=depth is not None)
        # The fact sheet remains semantic (`crouching`).  Only the text-facing
        # projection is lexicalized, while retaining the canonical class for
        # audit/debugging.
        body["canonical_broad_pose"] = "crouching"
        body["broad_pose"] = surface
        body["broad_pose_lexicalization"] = surface
        if depth is not None:
            body["crouched_stance_depth"] = depth
        authoritative["body"] = body
        projection["authoritative_facts"] = authoritative

        audit["crouching_surface_lexicalization"] = surface
        audit["crouched_stance_depth_projected"] = depth is not None
    else:
        audit["crouching_surface_lexicalization"] = None
        audit["crouched_stance_depth_projected"] = False

    return projection, audit


def _projected_body(projection: dict[str, Any]) -> dict[str, Any]:
    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    return authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}


def _depth_authorized(projection: dict[str, Any]) -> bool:
    body = _projected_body(projection)
    depth = body.get("crouched_stance_depth") if isinstance(body.get("crouched_stance_depth"), dict) else {}
    return bool(
        depth.get("classification") == "moderate_lowering"
        and depth.get("semantic_relation") == "hips_slightly_lowered"
    )


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = list(audit.get("violations") or [])

    body = _projected_body(projection)
    canonical_pose = _clean(body.get("canonical_broad_pose"))
    depth_authorized = _depth_authorized(projection)
    crouched_stance_used = bool(_CROUCHED_STANCE_RE.search(caption))
    slight_lowering_used = bool(_HIPS_SLIGHTLY_LOWERED_RE.search(caption))
    any_hips_lowering_used = bool(_HIPS_LOWERED_ANY_RE.search(caption))
    deep_or_low_crouch_used = bool(_DEEP_OR_LOW_CROUCH_RE.search(caption))

    if canonical_pose == "crouching" and not crouched_stance_used:
        violations.append("crouching_not_lexicalized_as_crouched_stance")

    if depth_authorized and not slight_lowering_used:
        violations.append("authorized_hips_slightly_lowered_modifier_missing")
    if any_hips_lowering_used and not depth_authorized:
        violations.append("hips_lowered_language_without_depth_authority")

    # The promoted relation is deliberately moderate.  Never let surface prose
    # strengthen it into a deep/low crouch.
    if deep_or_low_crouch_used:
        violations.append("unsupported_deep_or_low_crouch_language")

    audit["violations"] = sorted(set(violations))
    audit["canonical_broad_pose"] = canonical_pose
    audit["crouched_stance_language_used"] = crouched_stance_used
    audit["crouched_stance_depth_authorized"] = depth_authorized
    audit["hips_slightly_lowered_language_used"] = slight_lowering_used
    audit["deep_or_low_crouch_language_used"] = deep_or_low_crouch_used
    return audit


def _retry_prompt(original_prompt: str, caption: str, violations: list[str]) -> str:
    prompt = _BASE_RETRY_PROMPT(original_prompt, caption, violations)
    extra: list[str] = []

    if "crouching_not_lexicalized_as_crouched_stance" in violations:
        extra.append(
            "For the authoritative crouching pose, use the phrase 'holds a crouched stance' rather than the bare verb 'crouches'."
        )
    if "authorized_hips_slightly_lowered_modifier_missing" in violations:
        extra.append(
            "The authoritative crouch-depth modifier is 'hips slightly lowered'. Preserve that conservative modifier naturally, preferably as 'holds a crouched stance with her hips slightly lowered'."
        )
    if "hips_lowered_language_without_depth_authority" in violations:
        extra.append(
            "Remove all claims that the hips are lowered; no crouch-depth modifier is authorized for this image."
        )
    if "unsupported_deep_or_low_crouch_language" in violations:
        extra.append(
            "Remove 'deep' and 'low' crouch wording. The evidence does not authorize a deep or low crouch."
        )

    if not extra:
        return prompt
    return prompt + "\nCROUCHED-STANCE REVISION:\n- " + "\n- ".join(extra)


def main() -> int:
    # v09 retains every validated Phase-5.8 audit/retry guard.  Patch only the
    # projection and extend its audit/retry contract for crouched-stance depth.
    phase58.phase57._projection = _projection
    phase58._caption_audit = _caption_audit
    phase58._retry_prompt = _retry_prompt
    phase58.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase58.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase58.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase58.SCHEMA_VERSION = SCHEMA_VERSION
    phase58.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase58.main()


if __name__ == "__main__":
    raise SystemExit(main())
