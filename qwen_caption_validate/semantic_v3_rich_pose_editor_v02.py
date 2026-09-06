from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor as v01
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_editor_v02.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-editor-0.2"
RUN_VERSION = "semantic-v3-rich-pose-editor-0.2-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-editor-v0.2"

_V01_BUILD_EDITOR_INPUT = v01.build_editor_input
_V01_QUALITY_AUDIT = v01.quality_audit


_HAIR_COLOR_PATTERNS = (
    re.compile(r"\b(?:blond(?:e)?|brunette|auburn|ginger|red|brown|black|gray|grey|silver|white|dark|light)(?:[- ](?:brown|blond(?:e)?|red|black|gray|grey|silver|white))?\s+hair\b", re.IGNORECASE),
    re.compile(r"\bhair\s+(?:is|appears|looks)\s+(?:blond(?:e)?|brunette|auburn|ginger|red|brown|black|gray|grey|silver|white|dark|light)\b", re.IGNORECASE),
    re.compile(r"\b(?:lighter|darker|blond(?:e)?|brown|black|red|gray|grey|silver)\s+(?:highlights?|roots?)\b", re.IGNORECASE),
)
_HAIR_LENGTH_PATTERNS = (
    re.compile(r"\b(?:very\s+)?(?:long|short|medium-length|shoulder-length|chin-length|jaw-length|neck-length|waist-length|hip-length|mid-back-length)\s+hair\b", re.IGNORECASE),
    re.compile(r"\bhair\s+(?:is|appears|looks|falls|reaches)\s+(?:very\s+)?(?:long|short|medium-length|shoulder-length|chin-length|jaw-length|neck-length|waist-length|hip-length|mid-back-length)\b", re.IGNORECASE),
    re.compile(r"\b(?:shoulder-length|chin-length|jaw-length|neck-length|waist-length|hip-length|mid-back-length)\b", re.IGNORECASE),
)
_EYE_COLOR_PATTERNS = (
    re.compile(r"\b(?:blue|green|brown|hazel|gray|grey|dark|light)\s+eyes\b", re.IGNORECASE),
)
_SKIN_TONE_PATTERNS = (
    re.compile(r"\b(?:fair|pale|light|dark|brown|olive|tan|tanned)\s+(?:skin|complexion)\b", re.IGNORECASE),
    re.compile(r"\b(?:appears?|looks?)\s+tanned\b", re.IGNORECASE),
)
_AGE_PATTERNS = (
    re.compile(r"\b(?:young|middle-aged|elderly|older|teenage|teenaged)\s+(?:woman|man|person|subject)\b", re.IGNORECASE),
)
_META_REDACTION_RE = re.compile(
    r"\b(?:no specific|not specified|not described|not emphasized|omitted|left unspecified|without mentioning|identity detail)\b",
    re.IGNORECASE,
)
_BODY_SIDE_RE = re.compile(
    r"\b(left|right)\s+(hand|fist|wrist|arm|elbow|shoulder|hip|knee|leg|foot)\b",
    re.IGNORECASE,
)
_POSSESSIVE_SIDE_RE = re.compile(
    r"\b(?:her|his|their)\s+(left|right)\s+side\b",
    re.IGNORECASE,
)


def _unique_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            key = value.lower()
            if key not in seen:
                seen.add(key)
                values.append(value)
    return values


def _identity_mentions(text: str) -> list[str]:
    return _unique_matches(
        text,
        _HAIR_COLOR_PATTERNS
        + _HAIR_LENGTH_PATTERNS
        + _EYE_COLOR_PATTERNS
        + _SKIN_TONE_PATTERNS
        + _AGE_PATTERNS,
    )


def _unsupported_laterality_mentions(text: str, pose: dict[str, Any]) -> list[str]:
    authorized = v01._authorized_laterality(pose)
    values: list[str] = []
    seen: set[str] = set()
    for match in _BODY_SIDE_RE.finditer(text):
        pair = (match.group(1).lower(), match.group(2).lower())
        if pair in authorized:
            continue
        value = match.group(0)
        if value.lower() not in seen:
            seen.add(value.lower())
            values.append(value)
    # Possessive body-side wording is never independently governed by the current Pose language contract.
    for match in _POSSESSIVE_SIDE_RE.finditer(text):
        value = match.group(0)
        if value.lower() not in seen:
            seen.add(value.lower())
            values.append(value)
    return values


def _mandatory_redactions(draft: str, pose: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "protected_identity_mentions": _identity_mentions(draft),
        "unsupported_anatomical_laterality": _unsupported_laterality_mentions(draft, pose),
    }


def _redaction_text(redactions: dict[str, list[str]]) -> str:
    lines: list[str] = []
    identity = redactions["protected_identity_mentions"]
    laterality = redactions["unsupported_anatomical_laterality"]
    if identity:
        lines.append("- Protected identity mentions that MUST be removed without paraphrase: " + "; ".join(identity))
    else:
        lines.append("- Protected identity mentions detected in draft: none")
    if laterality:
        lines.append("- Unsupported anatomical laterality that MUST be removed or neutralized: " + "; ".join(laterality))
    else:
        lines.append("- Unsupported anatomical laterality detected in draft: none")
    return "\n".join(lines)


def build_editor_input(rich: dict[str, Any], pose: dict[str, Any], prompt_template: str) -> dict[str, Any]:
    base = _V01_BUILD_EDITOR_INPUT(rich, pose, prompt_template)
    draft = base["rich_draft"]
    redactions = _mandatory_redactions(draft, pose)
    prompt = base["editor_prompt"].replace("{{MANDATORY_REDACTIONS}}", _redaction_text(redactions))
    base["mandatory_redactions"] = redactions
    base["editor_prompt"] = prompt
    return base


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _V01_QUALITY_AUDIT(draft, edited, pose)
    identity_leaks = _identity_mentions(edited)
    unsupported_laterality = _unsupported_laterality_mentions(edited, pose)
    meta_redaction_language = sorted({m.group(0) for m in _META_REDACTION_RE.finditer(edited)}, key=str.lower)

    warnings = [
        value
        for value in base.get("warnings") or []
        if value not in {"intrinsic_identity_leakage", "unsupported_anatomical_laterality"}
    ]
    if identity_leaks:
        warnings.append("intrinsic_identity_leakage")
    if unsupported_laterality:
        warnings.append("unsupported_anatomical_laterality")
    if meta_redaction_language:
        warnings.append("meta_redaction_language")

    base["identity_leaks"] = sorted(set(identity_leaks), key=str.lower)
    base["unauthorized_anatomical_laterality"] = sorted(set(unsupported_laterality), key=str.lower)
    base["meta_redaction_language"] = meta_redaction_language
    base["warnings"] = warnings
    base["passes_basic_gate"] = not warnings
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
    v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    v01.RUN_VERSION = RUN_VERSION
    v01.build_editor_input = build_editor_input
    v01.quality_audit = quality_audit
    _inject_default_output_dir()
    return v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
