from __future__ import annotations

import re
from typing import Any


_ANATOMICAL_LATERALITY_RE = re.compile(
    r"\b(left|right)\s+(hand|wrist|forearm|arm|elbow|shoulder|hip|knee|ankle|leg|foot|torso|body|head|eye|ear|side)\b",
    re.IGNORECASE,
)
_BOTH_HANDS_RE = re.compile(r"\bboth\s+hands\b|\bhands\b", re.IGNORECASE)
_META_RE = re.compile(
    r"\b(?:sam3d|dwpose|fusion|keypoints?|evidence|confidence|reconstruction|hidden anatomy|not visible|inferred)\b",
    re.IGNORECASE,
)

_DEPTH_TERMS = r"(?:stagger|depth|three[- ]dimensional|3d|rotat|turn|closer|farther|forward|back)"
_SHOULDER_DEPTH_RE = re.compile(
    rf"(?:shoulders?.{{0,50}}{_DEPTH_TERMS}|{_DEPTH_TERMS}.{{0,50}}shoulders?)",
    re.IGNORECASE,
)
_PELVIS_DEPTH_RE = re.compile(
    rf"(?:(?:hips?|pelvis).{{0,50}}{_DEPTH_TERMS}|{_DEPTH_TERMS}.{{0,50}}(?:hips?|pelvis))",
    re.IGNORECASE,
)
_TORSO_DEPTH_RE = re.compile(
    rf"(?:(?:torso|upper body).{{0,50}}{_DEPTH_TERMS}|{_DEPTH_TERMS}.{{0,50}}(?:torso|upper body))",
    re.IGNORECASE,
)


def _body_family(value: Any) -> str | None:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    for family, tokens in (
        ("hand", ("hand", "finger")),
        ("wrist", ("wrist",)),
        ("forearm", ("forearm",)),
        ("arm", ("upper arm", " arm", "arm ", "arm")),
        ("elbow", ("elbow",)),
        ("shoulder", ("shoulder",)),
        ("hip", ("hip", "pelvis")),
        ("knee", ("knee",)),
        ("ankle", ("ankle",)),
        ("leg", ("leg", "thigh", "calf")),
        ("foot", ("foot", "feet")),
        ("head", ("head",)),
        ("eye", ("eye",)),
        ("ear", ("ear",)),
    ):
        if any(token in text for token in tokens):
            return family
    return None


def _qualified_laterality(evidence: dict[str, Any]) -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    for item in evidence.get("visible_subject_parts") or []:
        if not isinstance(item, dict) or not item.get("laterality_qualified"):
            continue
        side = str(item.get("anatomical_side") or "unknown").lower()
        if side not in {"left", "right"}:
            continue
        values = [item.get("part"), *(item.get("visible_subparts") or [])]
        for value in values:
            family = _body_family(value)
            if family:
                allowed.add((side, family))
    for item in evidence.get("qualified_interactions") or []:
        if not isinstance(item, dict) or not item.get("laterality_qualified"):
            continue
        side = str(item.get("actor_anatomical_side") or "unknown").lower()
        family = _body_family(item.get("actor_part"))
        if side in {"left", "right"} and family:
            allowed.add((side, family))
    return allowed


def _qualified_hand_sides(evidence: dict[str, Any]) -> set[str]:
    sides: set[str] = set()
    for item in evidence.get("visible_subject_parts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("ownership") != "target" or not item.get("laterality_qualified"):
            continue
        family = _body_family(item.get("part"))
        subfamilies = {_body_family(value) for value in (item.get("visible_subparts") or [])}
        if family != "hand" and "hand" not in subfamilies:
            continue
        side = str(item.get("anatomical_side") or "unknown").lower()
        if side in {"left", "right"}:
            sides.add(side)
    return sides


def _visibility_family(name: str) -> tuple[str | None, str | None]:
    text = name.lower().replace("-", "_")
    side = "left" if text.startswith("left_") else "right" if text.startswith("right_") else None
    for family in ("hip", "knee", "ankle"):
        if family in text:
            return side, family
    return side, None


def _required_claim_present(caption: str, claim: dict[str, Any]) -> bool:
    claim_id = str(claim.get("id") or "")
    if claim_id == "shoulder_girdle_depth_rotation":
        return bool(_SHOULDER_DEPTH_RE.search(caption))
    if claim_id == "pelvis_depth_rotation":
        return bool(_PELVIS_DEPTH_RE.search(caption))
    if claim_id == "combined_torso_depth_rotation":
        return bool(_TORSO_DEPTH_RE.search(caption))
    return True


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Lint a generated caption against the caption-safe evidence contract.

    This is deliberately conservative and heuristic. Hard violations are intended
    to catch clear contract breaks; required-claim misses are warnings so the
    harness can expose model instruction-following without silently regenerating.
    """
    text = caption.strip()
    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    allowed_laterality = _qualified_laterality(evidence)
    for match in _ANATOMICAL_LATERALITY_RE.finditer(text):
        side = match.group(1).lower()
        body = match.group(2).lower()
        # "right side of the frame" is explicitly frame-relative, not anatomy.
        if body == "side" and re.match(r"\s+of\s+(?:the\s+)?(?:image\s+)?frame\b", text[match.end() :], re.I):
            continue
        family = "arm" if body == "body" else body
        if family == "side":
            family = "torso"
        if (side, family) not in allowed_laterality:
            violations.append(
                {
                    "type": "unqualified_anatomical_laterality",
                    "text": match.group(0),
                    "side": side,
                    "body_family": family,
                }
            )

    not_visible = (evidence.get("visibility_constraints") or {}).get("not_visible") or []
    not_visible_pairs = {_visibility_family(str(name)) for name in not_visible}
    for family in ("hip", "knee", "ankle"):
        sides = {side for side, found_family in not_visible_pairs if found_family == family and side}
        if sides == {"left", "right"}:
            pattern = re.compile(rf"\b{family}s?\b", re.I)
            if pattern.search(text):
                violations.append(
                    {
                        "type": "mentions_hard_not_visible_anatomy",
                        "body_family": family,
                        "visibility": "both_sides_not_visible",
                    }
                )
        else:
            for side in sides:
                pattern = re.compile(rf"\b{side}\s+{family}\b", re.I)
                if pattern.search(text):
                    violations.append(
                        {
                            "type": "mentions_hard_not_visible_anatomy",
                            "body_family": family,
                            "side": side,
                        }
                    )

    if _BOTH_HANDS_RE.search(text) and len(_qualified_hand_sides(evidence)) < 2:
        violations.append(
            {
                "type": "unsupported_plural_hands",
                "qualified_distinct_hand_sides": sorted(_qualified_hand_sides(evidence)),
            }
        )

    for match in _META_RE.finditer(text):
        violations.append(
            {
                "type": "pipeline_meta_language",
                "text": match.group(0),
            }
        )

    for claim in evidence.get("required_claims") or []:
        if isinstance(claim, dict) and not _required_claim_present(text, claim):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": claim.get("id"),
                    "magnitude_band": claim.get("magnitude_band"),
                }
            )

    return {
        "schema_version": "caption-authority-lint-1.0",
        "passed": not violations,
        "violation_count": len(violations),
        "warning_count": len(warnings),
        "violations": violations,
        "warnings": warnings,
    }
