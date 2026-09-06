from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor as v01
from . import semantic_v3_rich_pose_editor_v02 as v02
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_editor_v03.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-editor-0.3"
RUN_VERSION = "semantic-v3-rich-pose-editor-0.3-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-editor-v0.3"

_BASE_BUILD_EDITOR_INPUT = v01.build_editor_input
_V02_QUALITY_AUDIT = v02.quality_audit

_HAIR_ONLY_TOKEN_RE = re.compile(
    rf"^(?:{v02._HAIR_COLOR_WORDS}|{v02._HAIR_LENGTH_WORDS}|{v02._HAIR_STRUCTURE_WORDS})$",
    re.IGNORECASE,
)
_HAIR_DYE_TOKEN_RE = re.compile(r"\b(?:highlights?|roots?|streaks?)\b", re.IGNORECASE)
_HAIR_WORD_RE = re.compile(r"\bhair\b", re.IGNORECASE)
_HAIR_DYE_FORWARD_CONTEXT_RE = re.compile(
    r"\bhair\b[^.!?]{0,90}\b(?:with|featuring|showing|including|having|has)\b"
    r"[^.!?]{0,60}\b(?:highlights?|roots?|streaks?)\b",
    re.IGNORECASE,
)
_HAIR_DYE_REVERSE_CONTEXT_RE = re.compile(
    r"\b(?:highlights?|roots?|streaks?)\b[^.!?]{0,50}\b(?:in|through|throughout)\b"
    r"[^.!?]{0,30}\b(?:her|his|their|the)?\s*hair\b",
    re.IGNORECASE,
)
_TRANSIENT_FORWARD_RE = re.compile(
    r"\bhair\s+(?:is\s+)?(?:falling|falls|hanging|hangs|swept|sweeps)\s+forward"
    r"(?:[^.!?]{0,80}(?:obscur(?:es|ing)?|cover(?:s|ing)?)[^.!?]{0,40}(?:face|eye|eyes|forehead|cheek))?",
    re.IGNORECASE,
)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def _hair_dye_residue(text: str) -> list[str]:
    """Return dye/color-treatment nouns only when the sentence actually describes hair treatment.

    This intentionally does NOT flag lighting language such as 'gentle highlights on her hair'.
    """
    values: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not _HAIR_WORD_RE.search(sentence):
            continue
        if not (
            _HAIR_DYE_FORWARD_CONTEXT_RE.search(sentence)
            or _HAIR_DYE_REVERSE_CONTEXT_RE.search(sentence)
        ):
            continue
        values.extend(match.group(0) for match in _HAIR_DYE_TOKEN_RE.finditer(sentence))
    return _dedupe(values)


def _protected_hair_mentions(draft: str) -> list[str]:
    raw = v02._identity_mentions(draft)
    hair_values: list[str] = []
    for value in raw:
        low = value.lower()
        if "hair" in low or "highlight" in low or "root" in low or "streak" in low or _HAIR_ONLY_TOKEN_RE.fullmatch(value.strip()):
            hair_values.append(value)

    # Drop redundant bare tokens when a longer hair-context phrase already contains them.
    longer = [value for value in hair_values if len(value.split()) > 1 or "hair" in value.lower()]
    filtered: list[str] = []
    for value in hair_values:
        stripped = value.strip()
        if len(stripped.split()) == 1 and any(
            re.search(rf"\b{re.escape(stripped)}\b", other, re.IGNORECASE)
            for other in longer
            if other.lower() != stripped.lower()
        ):
            continue
        filtered.append(stripped)

    # If a draft explicitly describes dye/color-treatment details, remove the nouns too so
    # editing 'lighter highlights' does not leave meaningless 'highlights'.
    filtered.extend(_hair_dye_residue(draft))
    return _dedupe(filtered)


def _protected_other_identity_mentions(draft: str) -> list[str]:
    hair = {value.lower() for value in _protected_hair_mentions(draft)}
    return [
        value
        for value in v02._identity_mentions(draft)
        if value.lower() not in hair and not _HAIR_ONLY_TOKEN_RE.fullmatch(value.strip())
    ]


def _transient_hair_mentions(draft: str) -> list[str]:
    values = list(v02._transient_hair_mentions(draft))
    values.extend(match.group(0).strip() for match in _TRANSIENT_FORWARD_RE.finditer(draft))
    return _dedupe(values)


def _mandatory_redactions(draft: str, pose: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "protected_hair_mentions": _protected_hair_mentions(draft),
        "protected_other_identity_mentions": _protected_other_identity_mentions(draft),
        "transient_hair_mentions_to_preserve": _transient_hair_mentions(draft),
        "unsupported_anatomical_laterality": v02._unsupported_laterality_mentions(draft, pose),
    }


def _redaction_text(redactions: dict[str, list[str]]) -> str:
    lines: list[str] = []
    hair = redactions["protected_hair_mentions"]
    other = redactions["protected_other_identity_mentions"]
    transient = redactions["transient_hair_mentions_to_preserve"]
    laterality = redactions["unsupported_anatomical_laterality"]

    if hair:
        lines.append(
            "- Protected HAIR-ONLY identity details that MUST be removed from hair wording only: "
            + "; ".join(hair)
        )
    else:
        lines.append("- Protected hair identity details detected in draft: none")
    if other:
        lines.append("- Other protected identity mentions that MUST be removed: " + "; ".join(other))
    else:
        lines.append("- Other protected identity mentions detected in draft: none")
    if transient:
        lines.append(
            "- Transient/image-specific hair state that SHOULD be preserved while protected hair details are removed: "
            + "; ".join(transient)
        )
    else:
        lines.append("- Transient/image-specific hair state detected in draft: none")
    if laterality:
        lines.append("- Unsupported anatomical laterality that MUST be removed or neutralized: " + "; ".join(laterality))
    else:
        lines.append("- Unsupported anatomical laterality detected in draft: none")
    return "\n".join(lines)


def build_editor_input(rich: dict[str, Any], pose: dict[str, Any], prompt_template: str) -> dict[str, Any]:
    base = _BASE_BUILD_EDITOR_INPUT(rich, pose, prompt_template)
    draft = base["rich_draft"]
    redactions = _mandatory_redactions(draft, pose)
    base["editor_prompt"] = base["editor_prompt"].replace(
        "{{MANDATORY_REDACTIONS}}", _redaction_text(redactions)
    )
    base["mandatory_redactions"] = redactions
    return base


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _V02_QUALITY_AUDIT(draft, edited, pose)
    dye_residue = _hair_dye_residue(edited)
    if dye_residue:
        identity_leaks = list(base.get("identity_leaks") or [])
        identity_leaks.extend(dye_residue)
        base["identity_leaks"] = sorted(set(identity_leaks), key=str.lower)
        warnings = list(base.get("warnings") or [])
        if "intrinsic_identity_leakage" not in warnings:
            warnings.append("intrinsic_identity_leakage")
        base["warnings"] = warnings
        base["passes_basic_gate"] = False
    base["hair_dye_detail_leaks"] = dye_residue
    base["transient_hair_mentions"] = _transient_hair_mentions(edited)
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
    v02.DEFAULT_PROMPT = DEFAULT_PROMPT
    v02.ARTIFACT_VERSION = ARTIFACT_VERSION
    v02.RUN_VERSION = RUN_VERSION
    v02.build_editor_input = build_editor_input
    v02.quality_audit = quality_audit
    v02._inject_default_output_dir = _inject_default_output_dir
    return v02.main()


if __name__ == "__main__":
    raise SystemExit(main())
