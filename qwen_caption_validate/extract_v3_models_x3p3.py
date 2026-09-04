from __future__ import annotations

"""x3p3 VLM-facing wire contract.

x3p3 keeps the x3p2 Pydantic/xgrammar architecture but adds one explicit
representation for ambiguous human-body fragments, tightens postural-support
semantics, and makes body markings precision-first.

The canonical persistent model remains ``VisualExtractV3``. The x3p3
wire-to-canonical expander deterministically folds ambiguous fragments into the
canonical ``visible_body_parts`` list with fragment/unknown ownership semantics.

A narrow governance normalizer runs after raw JSON decoding and before Pydantic
structural validation. The original raw response is retained by the runner; the
normalizer only performs conservative, record-internal downgrades/repairs and
attaches a complete action report to the validated model.
"""

import json
from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from .extract_v3_models import (
    ConfidenceBand,
    Side,
    WireAction,
    WireAppearance,
    WireBodyPart,
    WireCamera,
    WireCapture,
    WireComposition,
    WireEntity,
    WireFraming,
    WireGaze,
    WireHeadBody,
    WireHeadOrientation,
    WireInteraction,
    WireLandmarks,
    WireOrientationCues,
    WirePosture,
    WireRelation,
    WireScene,
    WireTorsoOrientation,
    _WireModel,
)
from .extract_v3_normalize_x3p3 import normalize_x3p3_wire

_NON_TARGET_ENTITY_REF_PATTERN = r"^e[1-9][0-9]*$"


class WireMarkingX3P3(_WireModel):
    """Precision-first target marking.

    If confidence is not high enough to emit ``q='h'``, omit the item entirely.
    This intentionally trades marking recall for precision because false tattoos
    or body markings are especially harmful identity-caption evidence.
    """

    category: str = Field(alias="c", min_length=1)
    descriptors: list[str] = Field(alias="d")
    frame_location: str = Field(alias="l")
    visibility: Literal["full", "partial", "fragment", "occluded", "unknown"] = Field(alias="v")
    confidence: Literal["h"] = Field(alias="q")


class WireHumanFragmentX3P3(_WireModel):
    """Human-looking anatomy that cannot be traced into the target body chain."""

    part: Literal[
        "fingers",
        "hand_fragment",
        "arm_fragment",
        "leg_fragment",
        "foot_fragment",
        "human_fragment",
        "unknown",
    ] = Field(alias="p")
    visible_count: int | None = Field(default=None, alias="n", ge=1, le=10)
    side: Side = Field(alias="a")
    ownership: Literal["other", "unknown"] = Field(alias="o")
    connectivity: Literal["disconnected_in_crop", "unknown"] = Field(alias="k")
    geometry_cues: list[str] = Field(alias="g")
    contact_cues: list[str] = Field(alias="c")
    frame_location: str = Field(alias="l")
    confidence: ConfidenceBand = Field(alias="q")


class WireSubjectX3P3(_WireModel):
    clothing: list[WireAppearance] = Field(alias="cl")
    accessories: list[WireAppearance] = Field(alias="ac")
    markings: list[WireMarkingX3P3] = Field(alias="mk")
    hair_state: list[str] = Field(alias="hs")
    expression_state: list[str] = Field(alias="ex")
    # bp is reserved for anatomy visibly traceable into the target body chain.
    body_parts: list[WireBodyPart] = Field(alias="bp")
    # hf is the explicit escape hatch for isolated/ambiguous human fragments.
    human_fragments: list[WireHumanFragmentX3P3] = Field(alias="hf")
    landmarks: WireLandmarks = Field(alias="lm")
    orientation_cues: WireOrientationCues = Field(alias="or")
    gaze: WireGaze = Field(alias="g")
    interactions: list[WireInteraction] = Field(alias="ix")


class WireSupportX3P3(_WireModel):
    relation: Literal[
        "lying_on",
        "reclining_on",
        "seated_on",
        "standing_on",
        "leaning_against",
        "resting_on",
        "braced_on",
    ] = Field(alias="r")
    # A support target can never be the target subject itself.
    target_ref: str | None = Field(alias="t", pattern=_NON_TARGET_ENTITY_REF_PATTERN)
    target_description: str | None = Field(alias="d")
    evidence: Literal["observed", "contextual", "hypothesis"] = Field(alias="e")
    confidence: Literal["h", "m", "l"] = Field(alias="q")
    cues: list[str] = Field(alias="c")

    @model_validator(mode="after")
    def _target_is_present(self) -> "WireSupportX3P3":
        if self.target_ref is None and not self.target_description:
            raise ValueError("support requires target_ref or target_description; otherwise omit it")
        return self


class WireHypothesesX3P3(_WireModel):
    posture: WirePosture = Field(alias="p")
    torso: WireTorsoOrientation = Field(alias="to")
    head: WireHeadOrientation = Field(alias="ho")
    head_body: WireHeadBody = Field(alias="hb")
    camera: WireCamera = Field(alias="cam")
    capture: WireCapture = Field(alias="cap")
    support: list[WireSupportX3P3] = Field(alias="sup")
    actions: list[WireAction] = Field(alias="act")


class ExtractWireX3P3Runtime(_WireModel):
    """x3p3 runtime contract: normalized structural hard gates + semantic warnings."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, title="ExtractWireV1")

    schema_version: Literal["x3p3"] = Field(alias="v")
    overview: str | None = Field(alias="o")
    framing: WireFraming = Field(alias="f")
    subject: WireSubjectX3P3 = Field(alias="s")
    entities: list[WireEntity] = Field(alias="e")
    relations: list[WireRelation] = Field(alias="r")
    scene: WireScene = Field(alias="sc")
    composition: WireComposition = Field(alias="co")
    hypotheses: WireHypothesesX3P3 = Field(alias="h")
    uncertainties: list[str] = Field(alias="u")

    _normalization_report: dict[str, Any] = PrivateAttr(
        default_factory=lambda: {"version": None, "action_count": 0, "actions": []}
    )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> "ExtractWireX3P3Runtime":
        """Decode, mechanically normalize, then run ordinary Pydantic validation.

        Invalid JSON is delegated back to Pydantic so callers continue receiving
        a normal ``ValidationError`` rather than a raw ``JSONDecodeError``.
        """

        try:
            if isinstance(json_data, (bytes, bytearray)):
                parsed = json.loads(bytes(json_data).decode("utf-8"))
            else:
                parsed = json.loads(json_data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return super().model_validate_json(
                json_data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )

        if not isinstance(parsed, dict):
            return super().model_validate_json(
                json_data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )

        normalized, report = normalize_x3p3_wire(parsed)
        model = cls.model_validate(
            normalized,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )
        model._normalization_report = report
        return model

    def normalization_report(self) -> dict[str, Any]:
        return {
            "version": self._normalization_report.get("version"),
            "action_count": int(self._normalization_report.get("action_count") or 0),
            "actions": list(self._normalization_report.get("actions") or []),
        }

    @model_validator(mode="after")
    def _structural_invariants(self) -> "ExtractWireX3P3Runtime":
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
                warnings.append(f"entity {entity.entity_id} may duplicate the target subject")

        if self.framing.shot_scale == "full_body" and self.framing.subject_coverage == "face_dominant":
            warnings.append("framing inconsistency: full_body with face_dominant coverage")

        if self.hypotheses.torso.band == "frontal" and self.hypotheses.torso.faces_frame != "unknown":
            warnings.append(
                "torso hypothesis inconsistency: frontal orientation has left/right body_faces_frame"
            )

        posture = self.hypotheses.posture.value
        for index, support in enumerate(self.hypotheses.support):
            relation = support.relation
            conflict = (
                (posture == "seated" and relation in {"lying_on", "reclining_on", "standing_on"})
                or (posture == "lying" and relation in {"seated_on", "standing_on"})
                or (posture == "standing" and relation in {"lying_on", "reclining_on", "seated_on"})
                or (posture == "reclining" and relation == "standing_on")
            )
            if conflict:
                warnings.append(
                    f"support hypothesis inconsistency: posture={posture} with support.{index}={relation}"
                )

        # Broad posture remains non-authoritative, but close crops without any
        # lower-body/support evidence deserve an audit warning rather than silent
        # confidence. This does not alter the hypothesis value.
        if (
            self.framing.shot_scale in {"extreme_close_up", "close_up", "medium_close_up"}
            and posture in {"seated", "standing", "kneeling", "squatting", "crouching"}
            and not self.hypotheses.support
            and self.subject.landmarks.left_hip.visibility == "not_visible"
            and self.subject.landmarks.right_hip.visibility == "not_visible"
        ):
            warnings.append(
                f"posture hypothesis weakly constrained by crop: {posture} with hips not_visible and no support"
            )

        return warnings
