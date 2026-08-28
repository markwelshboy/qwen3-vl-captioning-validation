from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_140 import build_caption_projection as _build_140
from .caption_projection_140 import lint_caption as _lint_140

_COLOR = r"(?:black|white|gray|grey|blue|green|red|yellow|pink|orange|purple|brown|beige|tan|cream|teal|navy|dark|light)"
_SUNGLASSES_HEAD_RE = re.compile(
    rf"\b(?P<item>(?:{_COLOR}\s+)?sunglasses)\s+(?:(?:are|were)\s+)?(?:perched|resting|sitting)\s+(?:on|atop)\s+(?:(?:her|his|their|the)\s+)?head\b",
    re.IGNORECASE,
)
_MASK_CHIN_RE = re.compile(
    rf"\b(?P<item>(?:{_COLOR}\s+)?(?:(?:face|surgical)\s+)?mask)\s+(?:(?:is|was)\s+|(?:hangs?|rests?|sits?)\s+)?(?:pulled\s+down\s+)?(?:below|under)\s+(?:(?:her|his|their|the)\s+)?chin\b",
    re.IGNORECASE,
)
_WRISTBAND_RE = re.compile(rf"\b(?P<item>(?:{_COLOR}\s+)?wristband)\b", re.IGNORECASE)


def _clean_phrase(value: str) -> str:
    """Normalize summary-derived appearance text into a caption-fragment descriptor.

    Analyze summaries often begin a sentence with the accessory name (for example
    ``Sunglasses perched on head``). Caption evidence descriptors are fragments,
    so sentence-initial capitalization should not leak into their canonical form.
    Only the first character is normalized; the remainder is preserved.
    """
    text = re.sub(r"\s+", " ", value).strip(" ,.;:-")
    if text and text[0].isalpha():
        text = text[0].lower() + text[1:]
    return text


def _specific_accessory_states(analysis: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (family, safe descriptor) pairs from summary-only transient state text.

    These patterns intentionally preserve only non-identity accessory state. They do
    not carry anatomical left/right from Analyze and cannot introduce body geometry.
    """
    summary = str(analysis.get("image_summary") or "")
    out: list[tuple[str, str]] = []

    match = _SUNGLASSES_HEAD_RE.search(summary)
    if match:
        out.append(("sunglasses", f"{_clean_phrase(match.group('item'))} perched on head"))

    match = _MASK_CHIN_RE.search(summary)
    if match:
        out.append(("mask", f"{_clean_phrase(match.group('item'))} below chin"))

    for match in _WRISTBAND_RE.finditer(summary):
        out.append(("wristband", _clean_phrase(match.group("item"))))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for family, descriptor in out:
        key = (family, descriptor.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append((family, descriptor))
    return deduped


def _is_less_specific_descriptor(value: Any, family: str) -> bool:
    text = str(value or "").lower().strip()
    if family == "sunglasses":
        return text == "sunglasses" or text.endswith(" sunglasses")
    if family == "mask":
        return bool(re.search(r"\bmask\b$", text))
    if family == "wristband":
        return text == "wristband"
    return False


def _enrich_specific_accessory_states(
    evidence: dict[str, Any],
    analysis: dict[str, Any],
    projection: dict[str, Any],
) -> None:
    transient = evidence.setdefault("transient_appearance", {})
    descriptors = [str(value) for value in (transient.get("descriptors") or []) if str(value).strip()]
    added: list[str] = []
    replaced: list[str] = []

    for family, descriptor in _specific_accessory_states(analysis):
        retained: list[str] = []
        for existing in descriptors:
            if _is_less_specific_descriptor(existing, family):
                replaced.append(existing)
                continue
            retained.append(existing)
        descriptors = retained
        if descriptor.lower() not in {value.lower() for value in descriptors}:
            descriptors.append(descriptor)
            added.append(descriptor)

    transient["descriptors"] = descriptors
    if added or replaced:
        projection.setdefault("allowed", []).append(
            {
                "path": "analysis.image_summary[appearance-state quarantine]",
                "reason": "safe_transient_accessory_state_is_more_informative_than_bare_item",
                "added": added,
                "replaced": replaced,
            }
        )


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_140(fused_payload, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.4.1"
    projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    if isinstance(projection, dict):
        projection["schema_version"] = "caption-projection-audit-1.4.1"
    _enrich_specific_accessory_states(evidence, analysis, projection if isinstance(projection, dict) else audit)
    return evidence, audit


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_140(caption, evidence))
    result["schema_version"] = "caption-authority-lint-1.4.1"
    return result
