from __future__ import annotations

"""Deterministic x3p3 wire-to-canonical expansion."""

from typing import Any

from .extract_v3_models import BodyPart, VisualExtractV3
from .extract_v3_models_x3p3 import ExtractWireX3P3Runtime
from .extract_v3_wire_contract import (
    CONFIDENCE_BANDS,
    expand_extract_wire as _expand_x3p2_shape,
)


def expand_extract_wire(
    wire: ExtractWireX3P3Runtime,
) -> tuple[VisualExtractV3, dict[str, Any]]:
    """Expand normalized x3p3 without adding image semantics.

    The x3p2-shaped expander handles unchanged fields. x3p3's explicit
    human-fragment channel is then folded into canonical ``visible_body_parts``.
    Raw VLM output remains external provenance; ``wire.normalization_report()``
    records every mechanical pre-validation governance action.
    """

    canonical, metadata = _expand_x3p2_shape(wire)  # structural duck-typing is intentional

    fragment_parts: list[BodyPart] = []
    for item in wire.subject.human_fragments:
        visible_subparts: list[str] = []
        if item.visible_count is not None:
            visible_subparts.append(f"visible_count={item.visible_count}")

        fragment_parts.append(
            BodyPart(
                part=item.part,
                reported_anatomical_side=item.side,
                ownership_candidate=item.ownership,
                visibility="fragment",
                visible_subparts=visible_subparts,
                connectivity_to_target_chain=item.connectivity,
                geometry_cues=item.geometry_cues,
                contact_cues=item.contact_cues,
                frame_location=item.frame_location,
                confidence=CONFIDENCE_BANDS[item.confidence],
            )
        )

    if fragment_parts:
        target = canonical.target_subject.model_copy(
            update={
                "visible_body_parts": [
                    *canonical.target_subject.visible_body_parts,
                    *fragment_parts,
                ]
            }
        )
        canonical = canonical.model_copy(update={"target_subject": target})
        canonical = VisualExtractV3.model_validate(
            canonical.model_dump(mode="json", by_alias=True)
        )

    semantic_warnings = wire.semantic_warnings()
    normalization = wire.normalization_report()
    warnings = list(metadata.get("warnings") or [])
    warnings.extend(semantic_warnings)

    metadata.update(
        {
            "wire_schema_version": "x3p3",
            "wire_contract": "Pydantic ExtractWireX3P3Runtime",
            "canonical_contract": "Pydantic VisualExtractV3",
            "ambiguous_human_fragment_count": len(fragment_parts),
            "normalization": normalization,
            "semantic_warnings": semantic_warnings,
            "warnings": warnings,
        }
    )
    return canonical, metadata
