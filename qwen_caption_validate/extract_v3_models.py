from __future__ import annotations

"""Typed contracts for the V3 observe-once visual Extract.

`ExtractWireV1` is the VLM-facing transport contract. Its compact aliases are
used to generate the JSON Schema handed to vLLM/xgrammar. Python code uses long
descriptive field names. x3p2 deliberately spends a few extra characters on
anatomy aliases where x3p1's abbreviations proved semantically ambiguous.

`VisualExtractV3` is the canonical persistent contract consumed downstream. The
wire-to-canonical transform is deterministic and contains no image semantics.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ConfidenceBand = Literal["h", "m", "l", "u"]
Visibility = Literal["full", "partial", "fragment", "occluded", "unknown"]
LandmarkVisibility = Literal["visible", "partial", "not_visible", "unknown"]
Side = Literal["left", "right", "midline", "unknown"]
Ownership = Literal["target", "other", "unknown"]
Connectivity = Literal[
    "connected_visible", "connected_but_occluded", "disconnected_in_crop", "unknown"
]
EvidenceStatus = Literal["observed", "contextual", "hypothesis", "unknown"]

_ENTITY_REF_PATTERN = r"^(t|e[1-9][0-9]*)$"


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class WireFraming(_WireModel):
    shot_scale: Literal[
        "extreme_close_up", "close_up", "medium_close_up", "medium",
        "three_quarter", "near_full_body", "full_body", "unknown",
    ] = Field(alias="z")
    visible_extent: str | None = Field(alias="x")
    subject_coverage: Literal["small", "medium", "large", "face_dominant", "unknown"] = Field(alias="c")
    observations: list[str] = Field(alias="o")


class WireAppearance(_WireModel):
    category: str = Field(alias="c")
    descriptors: list[str] = Field(alias="d")
    frame_location: str = Field(alias="l")
    visibility: Visibility = Field(alias="v")
    confidence: ConfidenceBand = Field(alias="q")


class WireBodyPart(_WireModel):
    part: str = Field(alias="p")
    side: Side = Field(alias="a")
    ownership: Ownership = Field(alias="o")
    visibility: Literal["full", "partial", "fragment"] = Field(alias="v")
    visible_subparts: list[str] = Field(alias="s")
    connectivity: Connectivity = Field(alias="k")
    geometry_cues: list[str] = Field(alias="g")
    contact_cues: list[str] = Field(alias="c")
    frame_location: str = Field(alias="l")
    confidence: ConfidenceBand = Field(alias="q")


class WireLandmark(_WireModel):
    visibility: LandmarkVisibility = Field(alias="v")
    confidence: ConfidenceBand = Field(alias="q")
    evidence: str | None = Field(alias="e")


class WireLandmarks(_WireModel):
    # x3p1 used lh/rh for hip landmarks; Qwen repeatedly interpreted those as
    # left/right hand. x3p2 makes anatomy aliases explicit while keeping the
    # surrounding transport compact.
    head: WireLandmark = Field(alias="head")
    left_shoulder: WireLandmark = Field(alias="lshoulder")
    right_shoulder: WireLandmark = Field(alias="rshoulder")
    left_hip: WireLandmark = Field(alias="lhip")
    right_hip: WireLandmark = Field(alias="rhip")
    left_knee: WireLandmark = Field(alias="lknee")
    right_knee: WireLandmark = Field(alias="rknee")
    left_ankle: WireLandmark = Field(alias="lankle")
    right_ankle: WireLandmark = Field(alias="rankle")


class WireOrientationCues(_WireModel):
    torso: list[str] = Field(alias="t")
    head: list[str] = Field(alias="h")
    image_axis: list[str] = Field(alias="a")


class WireGaze(_WireModel):
    target: Literal[
        "camera_lens", "near_camera", "object", "down", "up", "off_camera", "unknown"
    ] = Field(alias="t")
    image_direction: Literal["image_left", "image_center", "image_right", "unknown"] = Field(alias="d")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")


class WireInteraction(_WireModel):
    # Wearing belongs to transient appearance; support belongs to support_context.
    kind: Literal["holding", "contact", "reaching", "crossing", "gesture", "unknown"] = Field(alias="k")
    actor_part: str = Field(alias="p")
    ownership: Ownership = Field(alias="o")
    target_ref: str | None = Field(alias="r", pattern=_ENTITY_REF_PATTERN)
    target_text: str | None = Field(alias="x")
    evidence: EvidenceStatus = Field(alias="e")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")


class WireSubject(_WireModel):
    clothing: list[WireAppearance] = Field(alias="cl")
    accessories: list[WireAppearance] = Field(alias="ac")
    markings: list[WireAppearance] = Field(alias="mk")
    hair_state: list[str] = Field(alias="hs")
    expression_state: list[str] = Field(alias="ex")
    body_parts: list[WireBodyPart] = Field(alias="bp")
    landmarks: WireLandmarks = Field(alias="lm")
    orientation_cues: WireOrientationCues = Field(alias="or")
    gaze: WireGaze = Field(alias="g")
    interactions: list[WireInteraction] = Field(alias="ix")


class WireEntity(_WireModel):
    entity_id: str = Field(alias="i", pattern=r"^e[1-9][0-9]*$")
    kind: Literal["person", "object", "furniture", "architecture", "vehicle", "device", "animal", "depiction", "reflection", "region", "other"] = Field(alias="t")
    class_name: str = Field(alias="c", min_length=1)
    descriptors: list[str] = Field(alias="d")
    visibility: Literal["full", "partial", "fragment", "occluded", "blurred", "unknown"] = Field(alias="v")
    frame_location: str = Field(alias="l")
    depth_band: Literal["foreground", "subject_plane", "background", "through_opening", "unknown"] = Field(alias="z")
    confidence: ConfidenceBand = Field(alias="q")


class WireRelation(_WireModel):
    # Generic relations are scene/spatial only. Holding/contact live in ix,
    # clothing/accessories/markings live under s, and support lives under h.sup.
    subject_ref: str = Field(alias="s", pattern=_ENTITY_REF_PATTERN)
    predicate: Literal[
        "visible_through", "reflected_in", "behind", "in_front_of", "beside", "near",
        "overlapping", "inside", "occludes", "other",
    ] = Field(alias="p")
    object_ref: str | None = Field(alias="o", pattern=_ENTITY_REF_PATTERN)
    object_text: str | None = Field(alias="x")
    evidence: EvidenceStatus = Field(alias="e")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")

    @model_validator(mode="after")
    def _object_is_present(self) -> "WireRelation":
        if self.object_ref is None and not self.object_text:
            raise ValueError("relation requires object_ref or object_text")
        return self


class WireEnvironment(_WireModel):
    candidate: Literal["indoor", "outdoor", "vehicle", "elevator", "studio_like", "mixed", "ambiguous", "unknown"] = Field(alias="v")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")
    counterevidence: list[str] = Field(alias="x")


class WireIllumination(_WireModel):
    kind: Literal["natural", "artificial", "mixed", "unknown"] = Field(alias="t")
    directionality: Literal["flat", "directional", "mixed", "unknown"] = Field(alias="d")
    contrast: Literal["low", "medium", "high", "unknown"] = Field(alias="k")
    observations: list[str] = Field(alias="o")


class WireBackgroundStructure(_WireModel):
    texture: Literal["low", "medium", "high", "unknown"] = Field(alias="t")
    structural: Literal["low", "medium", "high", "unknown"] = Field(alias="s")
    specular: Literal["none", "low", "medium", "high", "unknown"] = Field(alias="p")
    repeated_geometry: bool | None = Field(alias="r")
    lines_angles: Literal["low", "medium", "high", "unknown"] = Field(alias="l")
    reflections: bool | None = Field(alias="f")
    observations: list[str] = Field(alias="o")


class WireBackgroundRegion(_WireModel):
    description: str = Field(alias="d")
    relation: Literal["behind_subject", "foreground", "beside_subject", "surrounding", "unknown"] = Field(alias="r")
    frame_location: Literal["left", "center", "right", "spanning", "unknown"] = Field(alias="l")
    evidence: EvidenceStatus = Field(alias="e")
    confidence: ConfidenceBand = Field(alias="q")


class WireNuisanceRegion(_WireModel):
    description: str = Field(alias="d")
    frame_location: str = Field(alias="l")
    coverage: Literal["small", "medium", "large"] = Field(alias="c")
    texture: Literal["low", "medium", "high", "unknown"] = Field(alias="t")
    structural: Literal["low", "medium", "high", "unknown"] = Field(alias="s")
    specular: Literal["none", "low", "medium", "high", "unknown"] = Field(alias="p")
    entropy_focus: bool = Field(alias="e")


class WireScene(_WireModel):
    environment: WireEnvironment = Field(alias="env")
    illumination: WireIllumination = Field(alias="ill")
    background: WireBackgroundStructure = Field(alias="bg")
    background_regions: list[WireBackgroundRegion] = Field(alias="br")
    nuisance_regions: list[WireNuisanceRegion] = Field(alias="nr")


class WireComposition(_WireModel):
    subject_dominance: Literal["dominant", "balanced", "minor", "unknown"] = Field(alias="d")
    foreground_relations: list[str] = Field(alias="f")
    visual_thrust: list[str] = Field(alias="v")


class WirePosture(_WireModel):
    value: Literal["standing", "seated", "squatting", "crouching", "kneeling", "lying", "reclining", "unknown"] = Field(alias="v")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")
    limitations: list[str] = Field(alias="l")


class WireTorsoOrientation(_WireModel):
    band: Literal["frontal", "slightly_angled", "three_quarter", "side_on", "rear_three_quarter", "rear", "unknown"] = Field(alias="b")
    faces_frame: Literal["left", "right", "unknown"] = Field(alias="f")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")
    limitations: list[str] = Field(alias="l")


class WireHeadOrientation(_WireModel):
    yaw: Literal["frontal", "turned_left", "turned_right", "back_to_camera", "unknown"] = Field(alias="y")
    pitch: Literal["up", "down", "neutral", "unknown"] = Field(alias="p")
    roll: Literal["image_left", "image_right", "neutral", "unknown"] = Field(alias="r")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")
    limitations: list[str] = Field(alias="l")


class WireHeadBody(_WireModel):
    value: Literal["aligned", "turned_toward_camera", "turned_away_from_camera", "unknown"] = Field(alias="v")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")
    limitations: list[str] = Field(alias="l")


class WireCamera(_WireModel):
    elevation: Literal["very_low", "low", "eye_level", "high", "very_high", "unknown"] = Field(alias="e")
    pitch: Literal["upward", "level", "downward", "unknown"] = Field(alias="p")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")
    counterevidence: list[str] = Field(alias="x")


class WireCapture(_WireModel):
    mode: Literal["handheld_selfie", "mirror_selfie", "external_camera", "unknown"] = Field(alias="m")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")


class WireSupport(_WireModel):
    relation: Literal["lying_on", "reclining_on", "seated_on", "standing_on", "leaning_against", "resting_on", "braced_on", "unknown"] = Field(alias="r")
    target_ref: str | None = Field(alias="t", pattern=_ENTITY_REF_PATTERN)
    target_description: str | None = Field(alias="d")
    evidence: EvidenceStatus = Field(alias="e")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")


class WireAction(_WireModel):
    value: str = Field(alias="v")
    confidence: ConfidenceBand = Field(alias="q")
    cues: list[str] = Field(alias="c")
    limitations: list[str] = Field(alias="l")


class WireHypotheses(_WireModel):
    posture: WirePosture = Field(alias="p")
    torso: WireTorsoOrientation = Field(alias="to")
    head: WireHeadOrientation = Field(alias="ho")
    head_body: WireHeadBody = Field(alias="hb")
    camera: WireCamera = Field(alias="cam")
    capture: WireCapture = Field(alias="cap")
    support: list[WireSupport] = Field(alias="sup")
    actions: list[WireAction] = Field(alias="act")


class ExtractWireV1(_WireModel):
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
    def _semantic_contract_invariants(self) -> "ExtractWireV1":
        ids = [entity.entity_id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("entity ids must be unique")

        expected = {f"e{i}" for i in range(1, len(ids) + 1)}
        if set(ids) != expected:
            raise ValueError("entity ids must be contiguous e1..eN")

        known = {"t", *ids}

        def require_ref(value: str | None, path: str) -> None:
            if value is not None and value not in known:
                raise ValueError(f"{path} references missing entity {value!r}")

        banned_target_classes = {
            "target", "target_subject", "target subject", "main_subject", "main subject",
            "primary_subject", "primary subject",
        }
        for entity in self.entities:
            if entity.class_name.strip().lower() in banned_target_classes:
                raise ValueError("target subject must not be duplicated in entities")

        for index, relation in enumerate(self.relations):
            require_ref(relation.subject_ref, f"relations.{index}.subject_ref")
            require_ref(relation.object_ref, f"relations.{index}.object_ref")
            if relation.object_ref is not None and relation.subject_ref == relation.object_ref:
                raise ValueError(f"relations.{index} cannot be a self-relation")

        for index, interaction in enumerate(self.subject.interactions):
            require_ref(interaction.target_ref, f"subject.interactions.{index}.target_ref")

        for index, support in enumerate(self.hypotheses.support):
            require_ref(support.target_ref, f"hypotheses.support.{index}.target_ref")

        if self.framing.shot_scale == "full_body" and self.framing.subject_coverage == "face_dominant":
            raise ValueError("full_body framing cannot be face_dominant")

        if self.hypotheses.torso.band == "frontal" and self.hypotheses.torso.faces_frame != "unknown":
            raise ValueError("frontal torso orientation must use body_faces_frame=unknown")

        return self


# Canonical persistent model -------------------------------------------------

class _CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AppearanceItem(_CanonicalModel):
    id: str
    category: str
    descriptors: list[str]
    frame_location: str
    visibility: Visibility
    confidence: float = Field(ge=0.0, le=1.0)


class BodyPart(_CanonicalModel):
    part: str
    reported_anatomical_side: Side
    ownership_candidate: Ownership
    visibility: Literal["full", "partial", "fragment"]
    visible_subparts: list[str]
    connectivity_to_target_chain: Connectivity
    geometry_cues: list[str]
    contact_cues: list[str]
    frame_location: str
    confidence: float = Field(ge=0.0, le=1.0)


class Landmark(_CanonicalModel):
    visibility: LandmarkVisibility
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str | None


class LandmarkMap(_CanonicalModel):
    head: Landmark
    left_shoulder: Landmark
    right_shoulder: Landmark
    left_hip: Landmark
    right_hip: Landmark
    left_knee: Landmark
    right_knee: Landmark
    left_ankle: Landmark
    right_ankle: Landmark


class TransientAppearance(_CanonicalModel):
    clothing: list[AppearanceItem]
    accessories: list[AppearanceItem]
    markings: list[AppearanceItem]
    hair_state: list[str]
    expression_state: list[str]


class OrientationCues(_CanonicalModel):
    torso: list[str]
    head: list[str]
    image_plane_body_axis: list[str]


class Gaze(_CanonicalModel):
    target_candidate: Literal["camera_lens", "near_camera", "object", "down", "up", "off_camera", "unknown"]
    image_direction: Literal["image_left", "image_center", "image_right", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]


class Interaction(_CanonicalModel):
    type: Literal["holding", "contact", "reaching", "crossing", "gesture", "unknown"]
    actor_part: str
    actor_ownership_candidate: Ownership
    target_ref: str | None
    target_text: str | None
    evidence_status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]


class TargetSubject(_CanonicalModel):
    entity_ref: Literal["target_subject"]
    transient_appearance: TransientAppearance
    visible_body_parts: list[BodyPart]
    geometry_landmark_visibility: LandmarkMap
    orientation_cues: OrientationCues
    gaze: Gaze
    interactions: list[Interaction]


class Entity(_CanonicalModel):
    id: str
    type: Literal["person", "object", "furniture", "architecture", "vehicle", "device", "animal", "depiction", "reflection", "region", "other"]
    class_name: str = Field(alias="class")
    descriptors: list[str]
    visibility: Literal["full", "partial", "fragment", "occluded", "blurred", "unknown"]
    frame_location: str
    depth_band: Literal["foreground", "subject_plane", "background", "through_opening", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class Relation(_CanonicalModel):
    subject_ref: str
    predicate: Literal[
        "visible_through", "reflected_in", "behind", "in_front_of", "beside", "near",
        "overlapping", "inside", "occludes", "other",
    ]
    object_ref: str | None
    object_text: str | None
    evidence_status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]


class Illumination(_CanonicalModel):
    type: Literal["natural", "artificial", "mixed", "unknown"]
    directionality: Literal["flat", "directional", "mixed", "unknown"]
    contrast: Literal["low", "medium", "high", "unknown"]
    observations: list[str]


class BackgroundStructure(_CanonicalModel):
    texture_complexity: Literal["low", "medium", "high", "unknown"]
    structural_complexity: Literal["low", "medium", "high", "unknown"]
    specular_reflective: Literal["none", "low", "medium", "high", "unknown"]
    repeated_geometry: bool | None
    strong_lines_or_angles: Literal["low", "medium", "high", "unknown"]
    reflections_present: bool | None
    observations: list[str]


class BackgroundRegion(_CanonicalModel):
    description: str
    relation_to_subject: Literal["behind_subject", "foreground", "beside_subject", "surrounding", "unknown"]
    frame_location: Literal["left", "center", "right", "spanning", "unknown"]
    evidence_status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)


class NuisanceRegion(_CanonicalModel):
    description: str
    frame_location: str
    frame_coverage: Literal["small", "medium", "large"]
    texture_complexity: Literal["low", "medium", "high", "unknown"]
    structural_complexity: Literal["low", "medium", "high", "unknown"]
    specular_reflective: Literal["none", "low", "medium", "high", "unknown"]
    entropy_focus_candidate: bool


class Scene(_CanonicalModel):
    environment_candidate: Literal["indoor", "outdoor", "vehicle", "elevator", "studio_like", "mixed", "ambiguous", "unknown"]
    environment_confidence: float = Field(ge=0.0, le=1.0)
    environment_cues: list[str]
    environment_counterevidence: list[str]
    illumination: Illumination
    background_structure: BackgroundStructure
    background_regions: list[BackgroundRegion]
    nuisance_regions: list[NuisanceRegion]


class CompositionObservations(_CanonicalModel):
    subject_dominance: Literal["dominant", "balanced", "minor", "unknown"]
    foreground_relations: list[str]
    visual_thrust_cues: list[str]


class Posture(_CanonicalModel):
    value: Literal["standing", "seated", "squatting", "crouching", "kneeling", "lying", "reclining", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]
    limitations: list[str]


class TorsoOrientation(_CanonicalModel):
    orientation_band: Literal["frontal", "slightly_angled", "three_quarter", "side_on", "rear_three_quarter", "rear", "unknown"]
    body_faces_frame: Literal["left", "right", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]
    limitations: list[str]


class HeadOrientation(_CanonicalModel):
    yaw: Literal["frontal", "turned_left", "turned_right", "back_to_camera", "unknown"]
    pitch: Literal["up", "down", "neutral", "unknown"]
    roll: Literal["image_left", "image_right", "neutral", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]
    limitations: list[str]


class HeadBodyRelation(_CanonicalModel):
    value: Literal["aligned", "turned_toward_camera", "turned_away_from_camera", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]
    limitations: list[str]


class Camera(_CanonicalModel):
    elevation: Literal["very_low", "low", "eye_level", "high", "very_high", "unknown"]
    pitch: Literal["upward", "level", "downward", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]
    counterevidence: list[str]


class Capture(_CanonicalModel):
    mode: Literal["handheld_selfie", "mirror_selfie", "external_camera", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]


class SupportContext(_CanonicalModel):
    subject_relation: Literal["lying_on", "reclining_on", "seated_on", "standing_on", "leaning_against", "resting_on", "braced_on", "unknown"]
    target_ref: str | None
    target_description: str | None
    evidence_status: EvidenceStatus
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]


class Action(_CanonicalModel):
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    cues: list[str]
    limitations: list[str]


class Hypotheses(_CanonicalModel):
    posture: Posture
    torso_orientation: TorsoOrientation
    head_orientation: HeadOrientation
    head_body_relation: HeadBodyRelation
    camera: Camera
    capture: Capture
    support_context: list[SupportContext]
    actions: list[Action]


class Framing(_CanonicalModel):
    shot_scale_candidate: Literal[
        "extreme_close_up", "close_up", "medium_close_up", "medium",
        "three_quarter", "near_full_body", "full_body", "unknown",
    ]
    visible_extent: str | None
    subject_frame_coverage: Literal["small", "medium", "large", "face_dominant", "unknown"]
    frame_observations: list[str]


class VisualExtractV3(_CanonicalModel):
    schema_version: Literal["visual-extract-3.0"]
    image_overview: str | None
    framing: Framing
    target_subject: TargetSubject
    entities: list[Entity]
    relations: list[Relation]
    scene: Scene
    composition_observations: CompositionObservations
    hypotheses: Hypotheses
    uncertainties: list[str]
