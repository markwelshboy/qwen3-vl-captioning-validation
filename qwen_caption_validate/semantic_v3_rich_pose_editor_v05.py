from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor_v04 as v04
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_editor_v05.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-editor-0.5"
RUN_VERSION = "semantic-v3-rich-pose-editor-0.5-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-editor-v0.5"

_V04_QUALITY_AUDIT = v04.quality_audit

# Pose-language v0.1 emits three governed camera-relative orientation vocabularies:
#   nearly side-on to the camera
#   strongly turned sideways to the camera
#   partly turned sideways to the camera
# v0.4's conflict gate recognized only the first family. Keep the editor/audit
# vocabulary aligned with the actual frozen Pose-language contract.
_GOVERNED_SIDEWAYS_CORRECTION_RE = re.compile(
    r"\b(?:"
    r"nearly\s+side-on\s+to\s+the\s+camera"
    r"|strongly\s+turned\s+sideways\s+to\s+the\s+camera"
    r"|partly\s+turned\s+sideways\s+to\s+the\s+camera"
    r")\b",
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


def _pose_conflict_leaks(text: str, pose: dict[str, Any]) -> list[str]:
    corrections = [str(value) for value in (pose.get("caption_ready_phrases") or [])]
    if not any(_GOVERNED_SIDEWAYS_CORRECTION_RE.search(value) for value in corrections):
        return []

    values: list[str] = []
    for match in _WEAK_TOWARD_CAMERA_RE.finditer(text):
        # A distinct head/face turn toward the camera is compatible with a governed
        # sideways torso/upper-body correction and must remain eligible to survive.
        local_prefix = text[max(0, match.start() - 28) : match.start()]
        if _HEAD_CONTEXT_RE.search(local_prefix):
            continue
        values.append(match.group(0))
    return _dedupe(values)


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _V04_QUALITY_AUDIT(draft, edited, pose)
    pose_conflicts = _pose_conflict_leaks(edited, pose)

    # Replace v0.4's narrower pose-conflict result with the vocabulary-complete
    # v0.5 result while preserving every other v0.4 audit rule unchanged.
    warnings = [
        value
        for value in (base.get("warnings") or [])
        if value != "conflicting_pose_wording"
    ]
    if pose_conflicts:
        warnings.append("conflicting_pose_wording")

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
    # Reuse the proven v0.4 editor implementation. Only the prompt wording,
    # artifact provenance/output tree, and pose-conflict audit are versioned here.
    v04.DEFAULT_PROMPT = DEFAULT_PROMPT
    v04.ARTIFACT_VERSION = ARTIFACT_VERSION
    v04.RUN_VERSION = RUN_VERSION
    v04.quality_audit = quality_audit
    v04._inject_default_output_dir = _inject_default_output_dir
    return v04.main()


if __name__ == "__main__":
    raise SystemExit(main())
