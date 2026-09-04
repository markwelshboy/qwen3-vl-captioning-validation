from __future__ import annotations

"""Runtime x3p2 wire contract with structural hard gates and semantic warnings.

The first x3p2 smoke showed that hypothesis/crop consistency checks do not belong
in Pydantic hard validation: a perfectly complete structured response was being
discarded because two non-authoritative semantic fields disagreed.  This module
keeps malformed graph state as a hard failure while surfacing semantic
inconsistencies for audit without destroying the Extract.

The field layout and aliases intentionally match ``ExtractWireV1`` x3p2, so the
VLM-facing constrained-output grammar is unchanged apart from non-semantic model
metadata such as class identity.
"""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .extract_v3_models import (
    WireComposition,
    WireFraming,
    WireHypotheses,
    WireRelation,
    WireScene,
    WireSubject,
    WireEntity,
    _WireModel,
)


class ExtractWireV1Runtime(_WireModel):
    """x3p2 wire record with only structural/referential hard validation."""

    # Keep the generated schema title stable for provenance/hash comparisons.
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        title="ExtractWireV1",
    )

    schema_version: Literal["x3p2"] = Field(alias="v")
    overview: str | None = Field(alias="o")
    framing: WireFraming = Field(alias="f")
    subject: WireSubject = Field(alias="s")
    entities: list[WireEntity] = Field(alias="e")
    relations: list[WireRelation] = Field(alias="r")
    scene: WireScene = Field(alias="sc")
    composition: WireComposition = Field(alias="co")
    hypotheses: WireHypotheses = Field(alias="h")
    uncertainties: list[str] = Field(alias="u")

    @model_validator(mode="after")
    def _structural_invariants(self) -> "ExtractWireV1Runtime":
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity ids must be unique")

        known = {"t", *ids}

        def require_ref(value: str | None, path: str) -> None:
            if value is not None and value not in known:
                raise ValueError(f"{path} references missing entity {value!r}")

        for index, relation in enumerate(self.relations):
            require_ref(relation.subject_ref, f"relations.{index}.subject_ref")
            require_ref(relation.object_ref, f"relations.{index}.object_ref")
            if relation.object_ref is not None and relation.subject_ref == relation.object_ref:
                raise ValueError(f"relations.{index} cannot be a self-relation")

        for index, interaction in enumerate(self.subject.interactions):
            require_ref(interaction.target_ref, f"subject.interactions.{index}.target_ref")

        for index, support in enumerate(self.hypotheses.support):
            require_ref(support.target_ref, f"hypotheses.support.{index}.target_ref")

        return self

    def semantic_warnings(self) -> list[str]:
        """Return reviewable semantic inconsistencies without rejecting the record."""

        warnings: list[str] = []
        ids = [entity.entity_id for entity in self.entities]
        expected = [f"e{i}" for i in range(1, len(ids) + 1)]
        if sorted(ids, key=lambda value: int(value[1:])) != expected:
            warnings.append("entity ids are non-contiguous; valid references are preserved")

        banned_target_classes = {
            "target",
            "target_subject",
            "target subject",
            "main_subject",
            "main subject",
            "primary_subject",
            "primary subject",
        }
        for entity in self.entities:
            if entity.class_name.strip().lower() in banned_target_classes:
                warnings.append(
                    f"entity {entity.entity_id} may duplicate the target subject"
                )

        if self.framing.shot_scale == "full_body" and self.framing.subject_coverage == "face_dominant":
            warnings.append("framing inconsistency: full_body with face_dominant coverage")

        if self.hypotheses.torso.band == "frontal" and self.hypotheses.torso.faces_frame != "unknown":
            warnings.append(
                "torso hypothesis inconsistency: frontal orientation has left/right body_faces_frame"
            )

        return warnings
