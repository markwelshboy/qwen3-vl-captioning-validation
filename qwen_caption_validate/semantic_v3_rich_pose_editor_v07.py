from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor_v03 as v03
from . import semantic_v3_rich_pose_editor_v06 as v06
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_editor_v07.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-editor-0.7"
RUN_VERSION = "semantic-v3-rich-pose-editor-0.7-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-editor-v0.7"

_BASE_BUILD_EDITOR_INPUT = v06.build_editor_input
_BASE_QUALITY_AUDIT = v06.quality_audit

# The blind set exposed two remaining deterministic boundaries:
#   * primary-subject identity protection must not erase clearly secondary/background people;
#   * stable haircut/length phrases can contain commas or compound modifiers that v0.6 split too literally.
_SECONDARY_PREFIX_PATTERNS = (
    re.compile(r"\b(?:another|other|second)\s+(?:person|woman|man|child|girl|boy|passenger|figure)\b[^.!?]{0,100}$", re.IGNORECASE),
    re.compile(r"\b(?:further|farther)\s+in\s+the\s+background\b[^.!?]{0,140}$", re.IGNORECASE),
    re.compile(r"\bin\s+the\s+background\b[^.!?]{0,140}$", re.IGNORECASE),
    re.compile(r"\bbehind\s+(?:her|him|them|the\s+subject)\b[^.!?]{0,140}$", re.IGNORECASE),
    re.compile(r"\b(?:next\s+to|beside)\s+(?:her|him|them|the\s+subject)\b[^.!?]{0,140}$", re.IGNORECASE),
)

_COMMA_CUT_LENGTH_RE = re.compile(
    r"\bhair\b\s*,?\s*(?:is\s+)?(?P<value>cut\s+to\s+"
    r"(?:shoulder|chin|jaw|neck|waist|hip)\s+length)\b",
    re.IGNORECASE,
)
_COMPOUND_BOB_RE = re.compile(
    r"\bhair\b[^.!?]{0,85}\b(?P<value>"
    r"(?:(?:soft|short)\s*,?\s*)?(?:layered\s+)?bob)\b",
    re.IGNORECASE,
)

# Keep transient state compact. v0.6 could preserve a whole clause such as
# "hair, cut to shoulder length, is swept back..." and thereby smuggle the protected
# length wording into the editor's ALLOWED/PRESERVE section.
_CLEAN_STRAND_STATE_RE = re.compile(
    r"\b(?P<value>(?:(?:some|a\s+few)\s+)?(?:loose\s+)?strands?\s+"
    r"(?:framing|falling|hanging|covering|swept|tucked)[^,.;!?]{0,75})",
    re.IGNORECASE,
)
_CLEAN_BACK_STATE_RE = re.compile(
    r"\b(?:hair\b[^.!?]{0,40}?\b)?(?P<value>"
    r"(?:swept|pulled|gathered)\s+back"
    r"(?:\s+from\s+(?:her|his|their|the)?\s*(?:face|forehead))?)\b",
    re.IGNORECASE,
)
_TRANSIENT_TAIL_RE = re.compile(
    r"\s*,\s*(?:wears?|smiles?|is\s+captured|is\s+shown|looks?|while\s+wearing)\b.*$",
    re.IGNORECASE,
)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value).strip(" \t,;")
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _whole_occurrences(text: str, value: str) -> list[re.Match[str]]:
    clean = value.strip()
    if not clean:
        return []
    if re.fullmatch(r"[\w-]+", clean):
        pattern = re.compile(rf"\b{re.escape(clean)}\b", re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(clean), re.IGNORECASE)
    return list(pattern.finditer(text))


def _is_secondary_occurrence(text: str, start: int) -> bool:
    sentence_start, _ = v06._sentence_bounds(text, start)
    prefix = text[sentence_start:start]
    return any(pattern.search(prefix) for pattern in _SECONDARY_PREFIX_PATTERNS)


def _is_scene_content_occurrence(text: str, start: int) -> bool:
    return v06._is_depicted_occurrence(text, start) or _is_secondary_occurrence(text, start)


def _filter_primary_mentions(text: str, values: list[str]) -> list[str]:
    """Keep a mention only if it has at least one primary-subject occurrence."""
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean:
            continue
        matches = _whole_occurrences(text, clean)
        if not matches:
            # Some legacy detectors return normalized fragments. Preserve them unless
            # they are known to occur only in scene-content context.
            result.append(clean)
            continue
        if any(not _is_scene_content_occurrence(text, match.start()) for match in matches):
            result.append(clean)
    return _dedupe(result)


def _additional_primary_hair_mentions(text: str) -> list[str]:
    values: list[str] = []
    for pattern in (_COMMA_CUT_LENGTH_RE, _COMPOUND_BOB_RE):
        for match in pattern.finditer(text):
            if _is_scene_content_occurrence(text, match.start()):
                continue
            values.append(match.group("value"))
    return _dedupe(values)


def _prefer_longest(values: list[str]) -> list[str]:
    """Prefer context-rich protected phrases over nested bare tokens."""
    clean = _dedupe(values)
    result: list[str] = []
    for value in clean:
        low = value.lower()
        if any(
            low != other.lower()
            and len(other) > len(value)
            and re.search(rf"\b{re.escape(low)}\b", other.lower())
            for other in clean
        ):
            continue
        result.append(value)
    return result


def _clean_transient_value(value: str) -> str:
    clean = _TRANSIENT_TAIL_RE.sub("", str(value)).strip(" \t,;")
    return clean


def _clean_transient_mentions(
    draft: str,
    values: list[str],
    protected_hair: list[str],
) -> list[str]:
    protected = [value.lower() for value in protected_hair if value.strip()]
    result: list[str] = []

    for value in values:
        clean = _clean_transient_value(value)
        if not clean:
            continue
        low = clean.lower()
        # A preserve hint containing protected length/cut/color/texture is unsafe.
        if any(item in low for item in protected):
            continue
        result.append(clean)

    for pattern in (_CLEAN_STRAND_STATE_RE, _CLEAN_BACK_STATE_RE):
        for match in pattern.finditer(draft):
            if _is_scene_content_occurrence(draft, match.start()):
                continue
            result.append(match.group("value"))

    return _dedupe(result)


def _revised_redactions(draft: str, base: dict[str, Any]) -> dict[str, list[str]]:
    hair = _filter_primary_mentions(
        draft,
        [str(value) for value in (base.get("protected_hair_mentions") or [])],
    )
    hair.extend(_additional_primary_hair_mentions(draft))
    hair = _prefer_longest(hair)

    other = _filter_primary_mentions(
        draft,
        [str(value) for value in (base.get("protected_other_identity_mentions") or [])],
    )

    transient = _clean_transient_mentions(
        draft,
        [str(value) for value in (base.get("transient_hair_mentions_to_preserve") or [])],
        hair,
    )

    laterality = [
        str(value).strip()
        for value in (base.get("unsupported_anatomical_laterality") or [])
        if str(value).strip()
    ]

    return {
        "protected_hair_mentions": hair,
        "protected_other_identity_mentions": _dedupe(other),
        "transient_hair_mentions_to_preserve": transient,
        "unsupported_anatomical_laterality": _dedupe(laterality),
    }


def _pose_correction_text(corrections: list[str]) -> str:
    if corrections:
        return "\n".join(f"- {value}" for value in corrections)
    return "- None. Preserve the draft geometry except for removing unsupported anatomical laterality."


def build_editor_input(
    rich: dict[str, Any],
    pose: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    base = _BASE_BUILD_EDITOR_INPUT(rich, pose, prompt_template)
    draft = str(base["rich_draft"])
    redactions = _revised_redactions(draft, base.get("mandatory_redactions") or {})
    corrections = [str(value) for value in (base.get("pose_corrections") or [])]

    prompt = (
        prompt_template.replace("{{RICH_DRAFT}}", draft)
        .replace("{{POSE_CORRECTIONS}}", _pose_correction_text(corrections))
        .replace("{{MANDATORY_REDACTIONS}}", v03._redaction_text(redactions))
    )
    base["mandatory_redactions"] = redactions
    base["editor_prompt"] = prompt
    return base


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _BASE_QUALITY_AUDIT(draft, edited, pose)

    identity = _filter_primary_mentions(
        edited,
        [str(value) for value in (base.get("identity_leaks") or [])],
    )
    identity.extend(_additional_primary_hair_mentions(edited))
    identity = _prefer_longest(identity)

    generic = _filter_primary_mentions(
        edited,
        [str(value) for value in (base.get("generic_identity_paraphrase_leaks") or [])],
    )
    dye = _filter_primary_mentions(
        edited,
        [str(value) for value in (base.get("hair_dye_detail_leaks") or [])],
    )
    additional = _additional_primary_hair_mentions(edited)

    warnings = [
        value
        for value in (base.get("warnings") or [])
        if value != "intrinsic_identity_leakage"
    ]
    if identity or generic or dye:
        warnings.append("intrinsic_identity_leakage")

    base["identity_leaks"] = identity
    base["generic_identity_paraphrase_leaks"] = generic
    base["hair_dye_detail_leaks"] = dye
    base["additional_protected_hair_leaks"] = additional
    base["warnings"] = _dedupe(warnings)
    base["passes_basic_gate"] = not base["warnings"]
    return base


def _inject_default_output_dir() -> None:
    if "--output-dir" in sys.argv or len(sys.argv) < 2:
        return
    run_dir = Path(sys.argv[1]).expanduser().resolve()
    model_arg = "32b-fp8"
    if "--model" in sys.argv:
        index = sys.argv.index("--model")
        if index + 1 < len(sys.argv):
            model_arg = sys.argv[index + 1]
    slug = model_slug(resolve_model_id(model_arg))
    output_dir = run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / slug
    sys.argv.extend(["--output-dir", str(output_dir)])


def main() -> int:
    # Reuse v0.6 mechanics and v0.5 Pose behavior. v0.7 changes only the
    # primary-subject identity context, compact transient allowlist, and provenance.
    v06.DEFAULT_PROMPT = DEFAULT_PROMPT
    v06.ARTIFACT_VERSION = ARTIFACT_VERSION
    v06.RUN_VERSION = RUN_VERSION
    v06.build_editor_input = build_editor_input
    v06.quality_audit = quality_audit
    v06._inject_default_output_dir = _inject_default_output_dir
    return v06.main()


if __name__ == "__main__":
    raise SystemExit(main())
