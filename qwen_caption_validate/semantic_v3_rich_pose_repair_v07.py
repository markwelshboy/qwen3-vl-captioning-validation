from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import semantic_v3_rich_pose_editor_v07 as editor_v07
from . import semantic_v3_rich_pose_repair as repair_v01
from . import semantic_v3_rich_pose_repair_v06 as repair_v06


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_pose_repair_v07.txt"
ARTIFACT_VERSION = "semantic-v3-rich-pose-repair-0.7"
RUN_VERSION = "semantic-v3-rich-pose-repair-0.7-run"
DEFAULT_OUTPUT_SUBDIR = "rich-pose-repair-v0.7"

_BASE_QUALITY_AUDIT = repair_v06.quality_audit
_BASE_FAILURE_LINES = repair_v06._failure_lines

# Keep this deliberately narrow. The rich-caption contract excludes apparent age,
# and the blind set exposed a few age-signaling facial-detail euphemisms that survived
# editor v0.7. Ordinary skin texture, freckles, blemishes, etc. are not included.
_AGE_PROXY_PATTERNS = (
    re.compile(
        r"\b(?P<value>(?:visible\s+)?fine\s+lines"
        r"(?:\s+(?:around|near)\s+the\s+(?:eyes?|forehead)"
        r"(?:\s+and\s+(?:the\s+)?forehead)?)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<value>(?:visible\s+)?crow['’]s\s+feet"
        r"(?:\s+(?:at|around|near)\s+[^,.;!?]{0,48})?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<value>(?:(?:visible|natural|prominent)\s+)?facial\s+lines)\b",
        re.IGNORECASE,
    ),
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


def _primary_subject_age_proxy_leaks(text: str) -> list[str]:
    values: list[str] = []
    for pattern in _AGE_PROXY_PATTERNS:
        for match in pattern.finditer(text):
            # Reuse editor v0.7's calibrated distinction between primary-subject
            # identity and clearly secondary/background/depicted scene content.
            if editor_v07._is_scene_content_occurrence(text, match.start()):
                continue
            values.append(match.group("value"))
    return _dedupe(values)


def quality_audit(draft: str, edited: str, pose: dict[str, Any]) -> dict[str, Any]:
    base = _BASE_QUALITY_AUDIT(draft, edited, pose)
    leaks = _primary_subject_age_proxy_leaks(edited)
    warnings = list(base.get("warnings") or [])
    if leaks:
        warnings.append("age_proxy_identity_leakage")

    base["age_proxy_identity_leaks"] = leaks
    base["warnings"] = list(dict.fromkeys(warnings))
    base["passes_basic_gate"] = not base["warnings"]
    return base


def _failure_lines(audit: dict[str, Any]) -> list[str]:
    lines = list(_BASE_FAILURE_LINES(audit))
    leaks = [
        str(value).strip()
        for value in (audit.get("age_proxy_identity_leaks") or [])
        if str(value).strip()
    ]
    if leaks:
        lines.append(
            "- Primary-subject apparent-age proxy: "
            + "; ".join(leaks)
            + ". Delete only these age-signaling facial-detail phrases and repair the local grammar; do not replace them with another age cue."
        )
    return lines


def main() -> int:
    repair_v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    repair_v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    repair_v01.RUN_VERSION = RUN_VERSION
    repair_v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    repair_v01.quality_audit = quality_audit
    repair_v01._failure_lines = _failure_lines
    return repair_v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
