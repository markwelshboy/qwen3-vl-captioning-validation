from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor as v01
from . import semantic_v3_rich_pose_editor_v03 as v03
from . import semantic_v3_rich_pose_editor_v04 as v04
from . import semantic_v3_rich_pose_editor_v05 as v05
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_editor_v06.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-editor-0.6"
RUN_VERSION = "semantic-v3-rich-pose-editor-0.6-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-editor-v0.6"

_BASE_BUILD_EDITOR_INPUT = v04.build_editor_input
_V05_QUALITY_AUDIT = v05.quality_audit

# Blind-set calibration showed two classes of deterministic misses:
#  1. stable subject-hair traits expressed with vocabulary outside v0.5's literal patterns;
#  2. protected-looking words inside depicted scene content (for example a tattoo portrait).
# Keep these rules contextual and narrow rather than broadening every color/length token globally.
_DEPICTED_PREFIX_RE = re.compile(
    r"\b(?:tattoo|portrait|artwork|photo(?:graph)?|painting|illustration|poster)\b"
    r"[^.!?]{0,180}\b(?:depicting|showing|of)\b[^.!?]{0,120}$",
    re.IGNORECASE,
)

_COLOR_EUPHEMISM_RE = re.compile(
    r"\b(?:light|dark)[- ](?:colored|toned)\s+hair\b",
    re.IGNORECASE,
)
_POST_HAIR_COLOR_RE = re.compile(
    r"\bhair\s+(?:is|appears|looks)\s+[^.!?]{0,55}?\b"
    r"(?P<value>(?:dark|light)\s+(?:brown|blond(?:e)?|black|red|gray|grey|silver|white)"
    r"|brown|blond(?:e)?|black|red|auburn|gray|grey|silver|white)\b",
    re.IGNORECASE,
)
_POST_HAIR_LENGTH_RE = re.compile(
    r"\bhair\s+(?:is|appears|looks)\s+[^.!?]{0,40}?\b"
    r"(?P<value>medium\s+to\s+long|medium[- ]length|short|long)\b",
    re.IGNORECASE,
)
_CUT_LENGTH_RE = re.compile(
    r"\bhair\s+(?:is\s+)?cut\s+to\s+"
    r"(?P<value>(?:shoulder|chin|jaw|neck|waist|hip)\s+length)\b",
    re.IGNORECASE,
)
_SHOULDER_REACH_RE = re.compile(
    r"\bhair\b[^.!?]{0,65}\b(?P<value>"
    r"(?:falls|hangs|reaches|extends)\s+(?:loosely\s+)?"
    r"(?:to|over|around)\s+(?:her|his|their|the)?\s*"
    r"(?:shoulders?|upper\s+back|mid[- ]back|lower\s+back|waist|hips?))\b",
    re.IGNORECASE,
)
_BOB_RE = re.compile(
    r"\bhair\b[^.!?]{0,70}\b(?:styled\s+(?:in|into|as)\s+)?"
    r"(?:a\s+)?(?P<value>(?:soft\s+|short\s+)?bob)\b",
    re.IGNORECASE,
)
_LAYER_RE = re.compile(
    r"\bhair\b[^.!?]{0,70}\b(?P<value>soft\s+layers?|layered)\b",
    re.IGNORECASE,
)
_WAVE_CURL_RE = re.compile(
    r"\bhair\b[^.!?]{0,80}\b(?P<value>(?:soft|gentle|natural)\s+(?:waves?|curls?))\b",
    re.IGNORECASE,
)
_NATURAL_TEXTURE_RE = re.compile(
    r"\bhair\b[^.!?]{0,90}\b(?P<value>"
    r"naturally\s+textured"
    r"|(?:appears|looks|is)[^.!?]{0,40}\btextured"
    r"|volume\s+and\s+texture)\b",
    re.IGNORECASE,
)

_ADDITIONAL_HAIR_PATTERNS: tuple[tuple[re.Pattern[str], str | None], ...] = (
    (_COLOR_EUPHEMISM_RE, None),
    (_POST_HAIR_COLOR_RE, "value"),
    (_POST_HAIR_LENGTH_RE, "value"),
    (_CUT_LENGTH_RE, "value"),
    (_SHOULDER_REACH_RE, "value"),
    (_BOB_RE, "value"),
    (_LAYER_RE, "value"),
    (_WAVE_CURL_RE, "value"),
    (_NATURAL_TEXTURE_RE, "value"),
)

# v0.3 already preserves many transient forms. These additional patterns recover
# image-specific strand/placement wording when a protected modifier interrupted the
# older literal detector in the original draft.
_ADDITIONAL_TRANSIENT_PATTERNS = (
    re.compile(
        r"\b(?:some\s+|a\s+few\s+)?(?:loose\s+)?strands?\b[^.!?]{0,55}\b"
        r"(?:framing|frame|tucked|swept|falling|covering)\b[^.!?]{0,55}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhair\b[^.!?]{0,70}\b(?:is\s+)?(?:swept|pulled|gathered)\s+back\b"
        r"[^.!?]{0,45}",
        re.IGNORECASE,
    ),
)

_HAIR_IDENTITY_TOKEN_RE = re.compile(
    r"^(?:"
    r"blond(?:e)?|brunette|auburn|ginger|red|brown|black|gray|grey|silver|white|dark|light"
    r"|long|short|medium-length|shoulder-length|chin-length|jaw-length|neck-length"
    r"|waist-length|hip-length|mid-back-length"
    r"|curly|straight|wavy|coily|kinky|layered|thick|fine"
    r"|highlights?|roots?|streaks?"
    r")$",
    re.IGNORECASE,
)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _sentence_bounds(text: str, index: int) -> tuple[int, int]:
    start = max(text.rfind(".", 0, index), text.rfind("!", 0, index), text.rfind("?", 0, index))
    start = 0 if start < 0 else start + 1
    ends = [pos for pos in (text.find(".", index), text.find("!", index), text.find("?", index)) if pos >= 0]
    end = min(ends) + 1 if ends else len(text)
    return start, end


def _is_depicted_occurrence(text: str, start: int) -> bool:
    sentence_start, _ = _sentence_bounds(text, start)
    prefix = text[sentence_start:start]
    return bool(_DEPICTED_PREFIX_RE.search(prefix))


def _whole_word_occurrences(text: str, value: str) -> list[re.Match[str]]:
    if re.fullmatch(r"[\w-]+", value):
        pattern = re.compile(rf"\b{re.escape(value)}\b", re.IGNORECASE)
    else:
        pattern = re.compile(re.escape(value), re.IGNORECASE)
    return list(pattern.finditer(text))


def _mention_has_primary_hair_context(text: str, value: str) -> bool:
    matches = _whole_word_occurrences(text, value)
    if not matches:
        return False
    for match in matches:
        if _is_depicted_occurrence(text, match.start()):
            continue
        sentence_start, sentence_end = _sentence_bounds(text, match.start())
        sentence = text[sentence_start:sentence_end]
        local_start = max(0, match.start() - sentence_start - 90)
        local_end = min(len(sentence), match.end() - sentence_start + 90)
        if re.search(r"\bhair\b", sentence[local_start:local_end], re.IGNORECASE):
            return True
    return False


def _mention_is_only_depicted(text: str, value: str) -> bool:
    matches = _whole_word_occurrences(text, value)
    return bool(matches) and all(_is_depicted_occurrence(text, match.start()) for match in matches)


def _filter_base_hair_mentions(text: str, values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean:
            continue
        if _mention_is_only_depicted(text, clean):
            continue
        # The old modifier-chain regex could start inside a larger word, producing
        # the phantom token "red" from "colored hair". Require a real whole-word
        # occurrence and nearby primary-subject hair context for bare hair tokens.
        if _HAIR_IDENTITY_TOKEN_RE.fullmatch(clean):
            if not _mention_has_primary_hair_context(text, clean):
                continue
        result.append(clean)
    return _dedupe(result)


def _filter_other_identity_mentions(text: str, values: list[str]) -> list[str]:
    return _dedupe([
        str(value).strip()
        for value in values
        if str(value).strip() and not _mention_is_only_depicted(text, str(value).strip())
    ])


def _additional_protected_hair_mentions(text: str) -> list[str]:
    values: list[str] = []
    for pattern, group in _ADDITIONAL_HAIR_PATTERNS:
        for match in pattern.finditer(text):
            if _is_depicted_occurrence(text, match.start()):
                continue
            value = match.group(group) if group else match.group(0)
            values.append(value.strip())
    return _dedupe(values)


def _additional_transient_hair_mentions(text: str) -> list[str]:
    values: list[str] = []
    for pattern in _ADDITIONAL_TRANSIENT_PATTERNS:
        for match in pattern.finditer(text):
            if _is_depicted_occurrence(text, match.start()):
                continue
            values.append(match.group(0).strip())
    return _dedupe(values)


def _revised_redactions(draft: str, base_redactions: dict[str, Any]) -> dict[str, list[str]]:
    hair = _filter_base_hair_mentions(
        draft,
        [str(v) for v in (base_redactions.get("protected_hair_mentions") or [])],
    )
    hair.extend(_additional_protected_hair_mentions(draft))

    other = _filter_other_identity_mentions(
        draft,
        [str(v) for v in (base_redactions.get("protected_other_identity_mentions") or [])],
    )

    transient = [
        str(v).strip()
        for v in (base_redactions.get("transient_hair_mentions_to_preserve") or [])
        if str(v).strip()
    ]
    transient.extend(_additional_transient_hair_mentions(draft))

    laterality = [
        str(v).strip()
        for v in (base_redactions.get("unsupported_anatomical_laterality") or [])
        if str(v).strip()
    ]

    return {
        "protected_hair_mentions": _dedupe(hair),
        "protected_other_identity_mentions": _dedupe(other),
        "transient_hair_mentions_to_preserve": _dedupe(transient),
        "unsupported_anatomical_laterality": _dedupe(laterality),
    }


def _pose_correction_text(corrections: list[str]) -> str:
    if corrections:
        return "\n".join(f"- {value}" for value in corrections)
    return "- None. Preserve the draft geometry except for removing unsupported anatomical laterality."


def build_editor_input(rich: dict[str, Any], pose: dict[str, Any], prompt_template: str) -> dict[str, Any]:
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


def _contextual_identity_leaks(text: str, base_values: list[str]) -> list[str]:
    values: list[str] = []
    for value in base_values:
        clean = str(value).strip()
        if not clean:
            continue
        if _mention_is_only_depicted(text, clean):
            continue
        if _HAIR_IDENTITY_TOKEN_RE.fullmatch(clean) and not _mention_has_primary_hair_context(text, clean):
            continue
        values.append(clean)
    values.extend(_additional_protected_hair_mentions(text))
    return _dedupe(values)


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _V05_QUALITY_AUDIT(draft, edited, pose)

    identity_leaks = _contextual_identity_leaks(
        edited,
        [str(value) for value in (base.get("identity_leaks") or [])],
    )
    additional_hair = _additional_protected_hair_mentions(edited)
    generic = list(base.get("generic_identity_paraphrase_leaks") or [])
    dye = list(base.get("hair_dye_detail_leaks") or [])

    warnings = [
        value
        for value in (base.get("warnings") or [])
        if value != "intrinsic_identity_leakage"
    ]
    if identity_leaks or generic or dye:
        warnings.append("intrinsic_identity_leakage")

    base["identity_leaks"] = identity_leaks
    base["additional_protected_hair_leaks"] = additional_hair
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
    # Reuse the proven editor mechanics and v0.5 Pose-conflict vocabulary.
    # v0.6 changes only prompt/provenance plus blind-calibrated identity context.
    v04.DEFAULT_PROMPT = DEFAULT_PROMPT
    v04.ARTIFACT_VERSION = ARTIFACT_VERSION
    v04.RUN_VERSION = RUN_VERSION
    v04.build_editor_input = build_editor_input
    v04.quality_audit = quality_audit
    v04._inject_default_output_dir = _inject_default_output_dir
    return v04.main()


if __name__ == "__main__":
    raise SystemExit(main())
