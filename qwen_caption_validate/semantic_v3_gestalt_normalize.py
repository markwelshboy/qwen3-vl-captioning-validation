from __future__ import annotations

from copy import deepcopy
from typing import Any


NORMALIZER_VERSION = "semantic-v3-gestalt-representation-0.1"

# composition-gestalt-1.4 deliberately uses a smaller evidence-status vocabulary
# than Visual Extract.  Qwen can occasionally copy Extract's source-side
# "hypothesis" provenance token into a Gestalt evidence_status field even though
# the supplied output contract says "inferred".  That is representation drift,
# not a semantic distinction, so canonicalize it mechanically while preserving
# the raw/model output elsewhere in the artifact.
_EVIDENCE_STATUS_COLLECTIONS = (
    "background_regions",
    "support_context",
    "foreground_relations",
    "salient_body_configuration",
)


def normalize_gestalt_representation(
    gestalt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonicalize unambiguous lexical drift before v1.4 schema validation.

    The function is intentionally narrow: it does not repair camera, capture,
    orientation, support ownership, confidence, or any other semantic claim.
    Input is never mutated.
    """
    normalized = deepcopy(gestalt)
    actions: list[dict[str, Any]] = []

    for collection in _EVIDENCE_STATUS_COLLECTIONS:
        rows = normalized.get(collection)
        if not isinstance(rows, list):
            continue
        for index, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            if item.get("evidence_status") != "hypothesis":
                continue
            item["evidence_status"] = "inferred"
            actions.append(
                {
                    "field": f"{collection}[{index}].evidence_status",
                    "original": "hypothesis",
                    "canonical": "inferred",
                    "reason": "Visual Extract source provenance token mapped to composition-gestalt-1.4 evidence vocabulary",
                }
            )

    audit = {
        "schema_version": NORMALIZER_VERSION,
        "policy": {
            "hypothesis": "inferred",
            "semantic_repair": "none",
            "raw_model_output_mutated": False,
        },
        "action_count": len(actions),
        "actions": actions,
    }
    return normalized, audit
