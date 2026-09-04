from __future__ import annotations

from copy import deepcopy
from typing import Any


NORMALIZER_VERSION = "semantic-v3-analyze-representation-0.1"


def normalize_analyze_representation(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize unambiguous wire vocabulary without changing semantic judgments."""
    out = deepcopy(data)
    actions: list[dict[str, Any]] = []

    for collection in ("actions", "interactions"):
        rows = out.get(collection)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            if row.get("evidence_status") == "hypothesis":
                row["evidence_status"] = "inferred"
                actions.append(
                    {
                        "path": f"{collection}.{index}.evidence_status",
                        "from": "hypothesis",
                        "to": "inferred",
                    }
                )

    audit = {
        "schema_version": NORMALIZER_VERSION,
        "policy": "representation only; semantic judgments unchanged",
        "action_count": len(actions),
        "actions": actions,
    }
    return out, audit
