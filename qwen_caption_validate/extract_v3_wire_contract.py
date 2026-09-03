from __future__ import annotations

from typing import Any

from .extract_v3_models import (
    Action,
    AppearanceItem,
    BackgroundRegion,
    BackgroundStructure,
    BodyPart,
    Camera,
    Capture,
    CompositionObservations,
    Entity,
    ExtractWireV1,
    Framing,
    Gaze,
    HeadBodyRelation,
    HeadOrientation,
    Hypotheses,
    Illumination,
    Interaction,
    Landmark,
    LandmarkMap,
    NuisanceRegion,
    OrientationCues,
    Posture,
    Relation,
    Scene,
    SupportContext,
    TargetSubject,
    TorsoOrientation,
    TransientAppearance,
    VisualExtractV3,
)

CONFIDENCE_BANDS = {
    "h": 0.90,
    "m": 0.65,
    "l": 0.35,
    "u": 0.00,
}


def _confidence(value: str) -> float:
    return CONFIDENCE_BANDS[value]


def _entity_ref(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "t":
        return "target_subject"
    if value.startswith("e") and value[1:].isdigit():
        return f"entity_{int(value[1:]):02d}"
    return value


def _known_entity_ids(wire: ExtractWireV1) -> set[str]:
    return {entity.entity_id for entity in wire.entities}


def _check_ref(value: str | None, known: set[str], warnings: list[str], path: str) -> None:
    if value is None or value == "t":
        return
    if value not in known:
        warnings.append(f"{path}: reference {value!r} has no matching entity")


def _appearance(items: list[Any], start_index: int) -> tuple[list[AppearanceItem], int]:
    out: list[AppearanceItem] = []
    index = start_index
    for item in items:
        out.append(
            AppearanceItem(
                id=f"appearance_{index:02d}",
                category=item.category,
                descriptors=item.descriptors,
                frame_location=item.frame_location,
                visibility=item.visibility,
                confidence=_confidence(item.confidence),
            )
        )
        index += 1
    return out, index


def expand_extract_wire(wire: ExtractWireV1) -> tuple[VisualExtractV3, dict[str, Any]]:
    """Deterministically expand `x3p1` into canonical `visual-extract-3.0`.

    No image semantics are added here. The transform only expands aliases,
    stable short references, generated appearance IDs and confidence bands.
    Both input and output are Pydantic-validated typed contracts.
    """

    warnings: list[str] = []
    known_entities = _known_entity_ids(wire)

    clothing, next_appearance = _appearance(wire.subject.clothing, 1)
    accessories, _ = _appearance(wire.subject.accessories, next_appearance)

    landmarks = wire.subject.landmarks
    landmark_map = LandmarkMap(
        head=Landmark(visibility=landmarks.head.visibility, confidence=_confidence(landmarks.head.confidence), evidence=landmarks.head.evidence),
        left_shoulder=Landmark(visibility=landmarks.left_shoulder.visibility, confidence=_confidence(landmarks.left_shoulder.confidence), evidence=landmarks.left_shoulder.evidence),
        right_shoulder=Landmark(visibility=landmarks.right_shoulder.visibility, confidence=_confidence(landmarks.right_shoulder.confidence), evidence=landmarks.right_shoulder.evidence),
        left_hip=Landmark(visibility=landmarks.left_hip.visibility, confidence=_confidence(landmarks.left_hip.confidence), evidence=landmarks.left_hip.evidence),
        right_hip=Landmark(visibility=landmarks.right_hip.visibility, confidence=_confidence(landmarks.right_hip.confidence), evidence=landmarks.right_hip.evidence),
        left_knee=Landmark(visibility=landmarks.left_knee.visibility, confidence=_confidence(landmarks.left_knee.confidence), evidence=landmarks.left_knee.evidence),
        right_knee=Landmark(visibility=landmarks.right_knee.visibility, confidence=_confidence(landmarks.right_knee.confidence), evidence=landmarks.right_knee.evidence),
        left_ankle=Landmark(visibility=landmarks.left_ankle.visibility, confidence=_confidence(landmarks.left_ankle.confidence), evidence=landmarks.left_ankle.evidence),
        right_ankle=Landmark(visibility=landmarks.right_ankle.visibility, confidence=_confidence(landmarks.right_ankle.confidence), evidence=landmarks.right_ankle.evidence),
    )

    interactions: list[Interaction] = []
    for index, item in enumerate(wire.subject.interactions):
        _check_ref(item.target_ref, known_entities, warnings, f"subject.interactions.{index}.target_ref")
        interactions.append(
            Interaction(
                type=item.kind,
                actor_part=item.actor_part,
                actor_ownership_candidate=item.ownership,
                target_ref=_entity_ref(item.target_ref),
                target_text=item.target_text,
                evidence_status=item.evidence,
                confidence=_confidence(item.confidence),
                cues=item.cues,
            )
        )

    entities: list[Entity] = []
    entity_ref_mapping: dict[str, str] = {}
    for item in wire.entities:
        canonical_id = _entity_ref(item.entity_id)
        assert canonical_id is not None
        entity_ref_mapping[item.entity_id] = canonical_id
        entities.append(
            Entity(
                id=canonical_id,
                type=item.kind,
                class_name=item.class_name,
                descriptors=item.descriptors,
                visibility=item.visibility,
                frame_location=item.frame_location,
                depth_band=item.depth_band,
                confidence=_confidence(item.confidence),
            )
        )

    relations: list[Relation] = []
    for index, item in enumerate(wire.relations):
        _check_ref(item.subject_ref, known_entities, warnings, f"relations.{index}.subject_ref")
        _check_ref(item.object_ref, known_entities, warnings, f"relations.{index}.object_ref")
        relations.append(
            Relation(
                subject_ref=_entity_ref(item.subject_ref) or item.subject_ref,
                predicate=item.predicate,
                object_ref=_entity_ref(item.object_ref),
                object_text=item.object_text,
                evidence_status=item.evidence,
                confidence=_confidence(item.confidence),
                cues=item.cues,
            )
        )

    supports: list[SupportContext] = []
    for index, item in enumerate(wire.hypotheses.support):
        _check_ref(item.target_ref, known_entities, warnings, f"hypotheses.support.{index}.target_ref")
        supports.append(
            SupportContext(
                subject_relation=item.relation,
                target_ref=_entity_ref(item.target_ref),
                target_description=item.target_description,
                evidence_status=item.evidence,
                confidence=_confidence(item.confidence),
                cues=item.cues,
            )
        )

    canonical = VisualExtractV3(
        schema_version="visual-extract-3.0",
        image_overview=wire.overview,
        framing=Framing(
            shot_scale_candidate=wire.framing.shot_scale,
            visible_extent=wire.framing.visible_extent,
            subject_frame_coverage=wire.framing.subject_coverage,
            frame_observations=wire.framing.observations,
        ),
        target_subject=TargetSubject(
            entity_ref="target_subject",
            transient_appearance=TransientAppearance(
                clothing=clothing,
                accessories=accessories,
                hair_state=wire.subject.hair_state,
                expression_state=wire.subject.expression_state,
            ),
            visible_body_parts=[
                BodyPart(
                    part=item.part,
                    reported_anatomical_side=item.side,
                    ownership_candidate=item.ownership,
                    visibility=item.visibility,
                    visible_subparts=item.visible_subparts,
                    connectivity_to_target_chain=item.connectivity,
                    geometry_cues=item.geometry_cues,
                    contact_cues=item.contact_cues,
                    frame_location=item.frame_location,
                    confidence=_confidence(item.confidence),
                )
                for item in wire.subject.body_parts
            ],
            geometry_landmark_visibility=landmark_map,
            orientation_cues=OrientationCues(
                torso=wire.subject.orientation_cues.torso,
                head=wire.subject.orientation_cues.head,
                image_plane_body_axis=wire.subject.orientation_cues.image_axis,
            ),
            gaze=Gaze(
                target_candidate=wire.subject.gaze.target,
                image_direction=wire.subject.gaze.image_direction,
                confidence=_confidence(wire.subject.gaze.confidence),
                cues=wire.subject.gaze.cues,
            ),
            interactions=interactions,
        ),
        entities=entities,
        relations=relations,
        scene=Scene(
            environment_candidate=wire.scene.environment.candidate,
            environment_confidence=_confidence(wire.scene.environment.confidence),
            environment_cues=wire.scene.environment.cues,
            environment_counterevidence=wire.scene.environment.counterevidence,
            illumination=Illumination(
                type=wire.scene.illumination.kind,
                directionality=wire.scene.illumination.directionality,
                contrast=wire.scene.illumination.contrast,
                observations=wire.scene.illumination.observations,
            ),
            background_structure=BackgroundStructure(
                texture_complexity=wire.scene.background.texture,
                structural_complexity=wire.scene.background.structural,
                specular_reflective=wire.scene.background.specular,
                repeated_geometry=wire.scene.background.repeated_geometry,
                strong_lines_or_angles=wire.scene.background.lines_angles,
                reflections_present=wire.scene.background.reflections,
                observations=wire.scene.background.observations,
            ),
            background_regions=[
                BackgroundRegion(
                    description=item.description,
                    relation_to_subject=item.relation,
                    frame_location=item.frame_location,
                    evidence_status=item.evidence,
                    confidence=_confidence(item.confidence),
                )
                for item in wire.scene.background_regions
            ],
            nuisance_regions=[
                NuisanceRegion(
                    description=item.description,
                    frame_location=item.frame_location,
                    frame_coverage=item.coverage,
                    texture_complexity=item.texture,
                    structural_complexity=item.structural,
                    specular_reflective=item.specular,
                    entropy_focus_candidate=item.entropy_focus,
                )
                for item in wire.scene.nuisance_regions
            ],
        ),
        composition_observations=CompositionObservations(
            subject_dominance=wire.composition.subject_dominance,
            foreground_relations=wire.composition.foreground_relations,
            visual_thrust_cues=wire.composition.visual_thrust,
        ),
        hypotheses=Hypotheses(
            posture=Posture(
                value=wire.hypotheses.posture.value,
                confidence=_confidence(wire.hypotheses.posture.confidence),
                cues=wire.hypotheses.posture.cues,
                limitations=wire.hypotheses.posture.limitations,
            ),
            torso_orientation=TorsoOrientation(
                orientation_band=wire.hypotheses.torso.band,
                body_faces_frame=wire.hypotheses.torso.faces_frame,
                confidence=_confidence(wire.hypotheses.torso.confidence),
                cues=wire.hypotheses.torso.cues,
                limitations=wire.hypotheses.torso.limitations,
            ),
            head_orientation=HeadOrientation(
                yaw=wire.hypotheses.head.yaw,
                pitch=wire.hypotheses.head.pitch,
                roll=wire.hypotheses.head.roll,
                confidence=_confidence(wire.hypotheses.head.confidence),
                cues=wire.hypotheses.head.cues,
                limitations=wire.hypotheses.head.limitations,
            ),
            head_body_relation=HeadBodyRelation(
                value=wire.hypotheses.head_body.value,
                confidence=_confidence(wire.hypotheses.head_body.confidence),
                cues=wire.hypotheses.head_body.cues,
                limitations=wire.hypotheses.head_body.limitations,
            ),
            camera=Camera(
                elevation=wire.hypotheses.camera.elevation,
                pitch=wire.hypotheses.camera.pitch,
                confidence=_confidence(wire.hypotheses.camera.confidence),
                cues=wire.hypotheses.camera.cues,
                counterevidence=wire.hypotheses.camera.counterevidence,
            ),
            capture=Capture(
                mode=wire.hypotheses.capture.mode,
                confidence=_confidence(wire.hypotheses.capture.confidence),
                cues=wire.hypotheses.capture.cues,
            ),
            support_context=supports,
            actions=[
                Action(
                    value=item.value,
                    confidence=_confidence(item.confidence),
                    cues=item.cues,
                    limitations=item.limitations,
                )
                for item in wire.hypotheses.actions
            ],
        ),
        uncertainties=wire.uncertainties,
    )

    metadata = {
        "wire_schema_version": "x3p1",
        "wire_contract": "Pydantic ExtractWireV1",
        "canonical_contract": "Pydantic VisualExtractV3",
        "confidence_band_mapping": dict(CONFIDENCE_BANDS),
        "entity_ref_mapping": entity_ref_mapping,
        "warnings": warnings,
    }
    return canonical, metadata
