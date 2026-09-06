from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor_v03 as v03
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_editor_v04.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-editor-0.4"
RUN_VERSION = "semantic-v3-rich-pose-editor-0.4-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-editor-v0.4"

_V03_BUILD_EDITOR_INPUT = v03.build_editor_input
_V03_QUALITY_AUDIT = v03.quality_audit

_GENERIC_IDENTITY_PATTERNS = (
    re.compile(r"\bnatural\s+(?:hair\s+)?(?:texture|length|color)\b", re.IGNORECASE),
    re.compile(r"\bnatural\s+(?:skin\s+tone|complexion)\b", re.IGNORECASE),
    re.compile(r"\bnatural\s+texture\s+and\s+length\b", re.IGNORECASE),
    re.compile(
        r"\b(?:hair|length)\b[^.!?]{0,60}\b(?:falls|reaches|extends|hangs)\s+"
        r"(?:down\s+)?(?:to|past)\s+(?:the\s+)?"
        r"(?:shoulders?|upper\s+back|mid[- ]back|lower\s+back|waist|hips?|chin|jaw|neck)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhair\b[^.!?]{0,90}\b(?:with|has|showing|featuring)\s+(?:a\s+)?"
        r"(?:slight|gentle|natural)\s+(?:wave|curl)\b",
        re.IGNORECASE,
    ),
)

_SIDE_ON_CORRECTION_RE = re.compile(
    r"\b(?:partly|strongly|nearly)\s+side-on\s+to\s+the\s+camera\b",
    re.IGNORECASE,
)
_WEAK_TOWARD_CAMERA_RE = re.compile(
    r"\b(?:turned\s+slightly|slightly\s+turned|angled\s+slightly|slightly\s+angled)\s+"
    r"toward(?:s)?\s+the\s+camera\b",
    re.IGNORECASE,
)
_HEAD_CONTEXT_RE = re.compile(r"\b(?:head|face)\b", re.IGNORECASE)


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


def _generic_identity_paraphrase_leaks(text: str) -> list[str]:
    values: list[str] = []
    for pattern in _GENERIC_IDENTITY_PATTERNS:
        values.extend(match.group(0) for match in pattern.finditer(text))

    # Catch constructions such as "hair falls naturally around her face, with a slight wave"
    # without treating ordinary transient placement as protected. The texture phrase itself is the leak.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not re.search(r"\bhair\b", sentence, re.IGNORECASE):
            continue
        for match in re.finditer(r"\b(?:a\s+)?(?:slight|gentle|natural)\s+(?:wave|curl)\b", sentence, re.IGNORECASE):
            values.append(match.group(0))
    return _dedupe(values)


def _pose_conflict_leaks(text: str, pose: dict[str, Any]) -> list[str]:
    corrections = [str(value) for value in (pose.get("caption_ready_phrases") or [])]
    if not any(_SIDE_ON_CORRECTION_RE.search(value) for value in corrections):
        return []

    values: list[str] = []
    for match in _WEAK_TOWARD_CAMERA_RE.finditer(text):
        # A head-specific turn is compatible with a side-on torso and should remain.
        local_prefix = text[max(0, match.start() - 28) : match.start()]
        if _HEAD_CONTEXT_RE.search(local_prefix):
            continue
        values.append(match.group(0))
    return _dedupe(values)


def build_editor_input(rich: dict[str, Any], pose: dict[str, Any], prompt_template: str) -> dict[str, Any]:
    return _V03_BUILD_EDITOR_INPUT(rich, pose, prompt_template)


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _V03_QUALITY_AUDIT(draft, edited, pose)
    generic_identity = _generic_identity_paraphrase_leaks(edited)
    pose_conflicts = _pose_conflict_leaks(edited, pose)

    warnings = list(base.get("warnings") or [])
    if generic_identity:
        if "intrinsic_identity_leakage" not in warnings:
            warnings.append("intrinsic_identity_leakage")
        warnings.append("generic_identity_paraphrase")
    if pose_conflicts:
        warnings.append("conflicting_pose_wording")

    base["generic_identity_paraphrase_leaks"] = generic_identity
    base["conflicting_pose_wording"] = pose_conflicts
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
    v03.DEFAULT_PROMPT = DEFAULT_PROMPT
    v03.ARTIFACT_VERSION = ARTIFACT_VERSION
    v03.RUN_VERSION = RUN_VERSION
    v03.build_editor_input = build_editor_input
    v03.quality_audit = quality_audit
    v03._inject_default_output_dir = _inject_default_output_dir
    return v03.main()


if __name__ == "__main__":
    raise SystemExit(main())
