from __future__ import annotations

from typing import Any


ANALYZE_REQUIRED_PATHS = (
    "framing",
    "target_subject.transient_appearance",
    "target_subject.visible_body_parts",
    "target_subject.geometry_landmark_visibility",
    "target_subject.orientation_cues",
    "target_subject.gaze",
    "target_subject.interactions",
    "entities",
    "relations",
    "scene",
    "hypotheses.posture",
    "hypotheses.torso_orientation",
    "hypotheses.head_orientation",
    "hypotheses.camera",
    "hypotheses.actions",
    "uncertainties",
)

GESTALT_REQUIRED_PATHS = (
    "framing",
    "entities",
    "relations",
    "scene.environment_candidate",
    "scene.background_regions",
    "composition_observations",
    "hypotheses.torso_orientation",
    "hypotheses.head_body_relation",
    "hypotheses.camera",
    "hypotheses.capture",
    "hypotheses.support_context",
    "uncertainties",
)


def _has_path(value: Any, path: str) -> bool:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def audit_extract_contract(extract: dict[str, Any]) -> dict[str, Any]:
    analyze_missing = [path for path in ANALYZE_REQUIRED_PATHS if not _has_path(extract, path)]
    gestalt_missing = [path for path in GESTALT_REQUIRED_PATHS if not _has_path(extract, path)]
    return {
        "schema_version": "visual-extract-contract-audit-0.1",
        "analyze_reconstructable": not analyze_missing,
        "gestalt_reconstructable": not gestalt_missing,
        "analyze_missing_paths": analyze_missing,
        "gestalt_missing_paths": gestalt_missing,
    }
