from __future__ import annotations

from qwen_caption_validate import fact_sheet_specialist_normalizer_v16 as phase4b15


def _sheet(configuration, *, left, right):
    return {
        "schema_version": "caption-fact-sheet-0.2.14",
        "status": "ok",
        "image_key": "synthetic",
        "facts": {
            "body": {
                "configuration": configuration,
                "anatomical_laterality": {
                    "available": True,
                    "authority": "dwpose_anatomical_joint_labels",
                    "sides": {
                        "left": {"visible_joints": list(left), "visible_joint_count": len(left)},
                        "right": {"visible_joints": list(right), "visible_joint_count": len(right)},
                    },
                },
            }
        },
        "audit": {"violations": [], "warnings": [], "invariants": {}},
    }


def _item(text):
    return {
        "text": text,
        "composer_text": text,
        "normalized_text": text,
        "domain": "configuration",
        "authority": "route_scoped_candidate",
        "promotion_status": "accepted_route_scoped_candidate",
    }


def test_head_support_collapses_duplicates_and_binds_only_hand_side():
    sheet = _sheet(
        [
            _item("fist under the chin with forearm beneath it"),
            _item("forearm held across the torso"),
            _item("hand supporting the chin/head"),
        ],
        left=("shoulder", "wrist"),
        right=("shoulder", "elbow"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    body = out["facts"]["body"]
    config = body["configuration"]

    assert len(config) == 2
    canonical = config[0]
    assert canonical["composer_text"] == (
        "chin resting on the left fist, with the forearm beneath/supporting the pose"
    )
    assert canonical["head_support_binding"]["anatomical_side"] == "left"
    assert canonical["head_support_binding"]["forearm_side_publishable"] is False

    assert config[1]["composer_text"] == "forearm held across the torso"

    adjudication = body["head_support_adjudication"]
    assert adjudication["status"] == "adjudicated"
    assert adjudication["applied"] is True
    assert adjudication["composer_authoritative"] is True
    assert adjudication["qwen_laterality_trusted"] is False
    assert adjudication["forearm_side_published"] is False
    assert "broad_pose" not in body


def test_non_head_support_configuration_is_left_unchanged():
    configuration = [
        _item("hand holding a white cup near the mouth"),
        _item("forearm resting on a surface with the hand partially visible"),
        _item("torso angled slightly forward relative to the surface"),
        _item("elbow bent and positioned near the torso"),
        _item("arm extended upward with the forearm visible beneath the cup"),
    ]
    sheet = _sheet(
        configuration,
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    body = out["facts"]["body"]

    assert body["configuration"] == configuration
    assert body["head_support_adjudication"]["status"] == "not_applicable"
    assert body["head_support_adjudication"]["applied"] is False


def test_generic_forearm_held_language_does_not_create_head_support():
    sheet = _sheet(
        [_item("forearm held across the torso")],
        left=("shoulder", "wrist"),
        right=("shoulder", "elbow"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    body = out["facts"]["body"]

    assert body["configuration"][0]["composer_text"] == "forearm held across the torso"
    assert body["head_support_adjudication"]["status"] == "not_applicable"


def test_bilateral_wrists_keep_head_support_side_neutral():
    sheet = _sheet(
        [
            _item("fist under the chin with forearm beneath it"),
            _item("hand supporting the chin/head"),
        ],
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    canonical = out["facts"]["body"]["configuration"][0]
    adjudication = out["facts"]["body"]["head_support_adjudication"]

    assert canonical["composer_text"] == (
        "chin resting on a fist, with the forearm beneath/supporting the pose"
    )
    assert "left" not in canonical["composer_text"]
    assert "right" not in canonical["composer_text"]
    assert adjudication["laterality_binding"]["status"] == "unresolved"


def _face_points(*, near: bool):
    points = {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (100.0, 0.0),
        "left_hip": (10.0, 150.0),
        "right_hip": (90.0, 150.0),
        "neck": (50.0, -20.0),
        "nose": (50.0, -60.0),
        "left_eye": (42.0, -65.0),
        "right_eye": (58.0, -65.0),
        "left_ear": (30.0, -60.0),
        "right_ear": (70.0, -60.0),
    }
    if near:
        points.update({
            "left_elbow": (20.0, -10.0),
            "left_wrist": (45.0, -55.0),
            "right_elbow": (120.0, 80.0),
            "right_wrist": (140.0, 140.0),
        })
    else:
        points.update({
            "left_elbow": (5.0, 105.0),
            "left_wrist": (12.0, 145.0),
            "right_elbow": (95.0, 105.0),
            "right_wrist": (88.0, 145.0),
        })
    return points


def test_face_relation_truth_gate_withholds_far_direct_relation():
    sheet = _sheet(
        [_item("hand near the face"), _item("forearm held beneath the chin/hand arrangement")],
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_face_relation_truth_gate(
        sheet,
        points=_face_points(near=False),
    )
    body = out["facts"]["body"]
    config = body["configuration"]

    assert config[0]["composer_text"] is None
    assert config[1]["composer_text"] is None
    assert config[0]["promotion_status"] == "withheld_by_face_relation_geometry_contradiction"
    gate = body["face_relation_truth_adjudication"]
    assert gate["status"] == "adjudicated"
    assert gate["withheld_count"] == 2
    assert gate["nearest_upper_limb_to_face_norm_body"] > (
        phase4b15.FACE_RELATION_CONTRADICTION_MIN_NORM_BODY
    )


def test_face_relation_truth_gate_preserves_near_direct_relation():
    sheet = _sheet(
        [_item("hand positioned near the chin with index finger extended")],
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_face_relation_truth_gate(
        sheet,
        points=_face_points(near=True),
    )
    body = out["facts"]["body"]

    assert body["configuration"][0]["composer_text"] == (
        "hand positioned near the chin with index finger extended"
    )
    assert body["face_relation_truth_adjudication"]["status"] == "preserved"


def test_face_relation_truth_gate_preserves_generic_relation_with_device_context():
    sheet = _sheet(
        [
            _item("hand holding a smartphone in front of the face"),
            _item("forearm extended with hand positioned near the lower face region"),
        ],
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_face_relation_truth_gate(
        sheet,
        points=_face_points(near=False),
    )
    body = out["facts"]["body"]

    assert body["configuration"][1]["composer_text"] == (
        "forearm extended with hand positioned near the lower face region"
    )
    gate = body["face_relation_truth_adjudication"]
    assert gate["status"] == "preserved_device_context"
    assert gate["applied"] is False
    assert gate["withheld_count"] == 0
    assert gate["device_context_relations"] == [
        "hand holding a smartphone in front of the face"
    ]


def test_face_relation_truth_gate_exempts_device_mediated_relation():
    sheet = _sheet(
        [_item("hand holding a smartphone in front of the face")],
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_face_relation_truth_gate(
        sheet,
        points=_face_points(near=False),
    )
    body = out["facts"]["body"]

    assert body["configuration"][0]["composer_text"] == (
        "hand holding a smartphone in front of the face"
    )
    assert body["face_relation_truth_adjudication"]["status"] == "not_applicable"


def test_face_relation_truth_gate_does_not_treat_negative_relation_as_positive():
    sheet = _sheet(
        [_item("no visible hand or fist contact with face or chin")],
        left=("shoulder",),
        right=("shoulder",),
    )

    out = phase4b15._apply_face_relation_truth_gate(sheet, points={})
    body = out["facts"]["body"]

    assert body["configuration"][0]["composer_text"] == (
        "no visible hand or fist contact with face or chin"
    )
    assert body["face_relation_truth_adjudication"]["candidate_count"] == 0


def test_face_relation_truth_gate_abstains_when_geometry_missing():
    sheet = _sheet(
        [_item("hand near the face")],
        left=(),
        right=(),
    )

    out = phase4b15._apply_face_relation_truth_gate(
        sheet,
        points={},
    )
    body = out["facts"]["body"]

    assert body["configuration"][0]["composer_text"] == "hand near the face"
    assert body["face_relation_truth_adjudication"]["status"] == "insufficient_evidence"


def _partial_pose_language(*, authority=0.9):
    return {
        "components": {
            "orientation": {
                "label": "three_quarter",
                "scope": "upper body",
                "phrase": "upper body partly turned sideways to the camera",
                "authority": authority,
                "camera_relative_only": True,
            },
        },
        "conditional_hints": [
            {
                "kind": "posture",
                "value": "crouching",
                "phrase": "crouching",
                "requires": "semantic_corroboration",
            },
        ],
    }


def test_partial_upper_body_orientation_recovers_only_local_geometry():
    sheet = _sheet([], left=("shoulder", "elbow"), right=("shoulder", "elbow"))
    sheet["policy"] = {"mode": "configuration"}
    sheet["facts"]["body"]["torso_geometry"] = {
        "available": True,
        "composer_eligible": False,
        "semantic_scope": "diagnostic_only_insufficient_torso_observation",
    }

    out = phase4b15._apply_partial_upper_body_orientation(
        sheet,
        pose_language=_partial_pose_language(),
        pose_language_source="/tmp/pose-language/00028.pose_language.json",
    )
    body = out["facts"]["body"]

    assert [item["composer_text"] for item in body["configuration"]] == [
        "upper body partly turned sideways to the camera"
    ]
    assert body.get("pose_candidate") is None
    gate = body["partial_upper_body_orientation_adjudication"]
    assert gate["status"] == "adjudicated"
    assert gate["broad_pose_created"] is False
    assert gate["conditional_posture_hint_promoted"] is False
    assert gate["ignored_conditional_hints"][0]["value"] == "crouching"


def test_partial_upper_body_orientation_does_not_displace_existing_configuration():
    sheet = _sheet(
        [_item("arm bent at the elbow")],
        left=("shoulder", "elbow"),
        right=("shoulder", "elbow"),
    )
    sheet["policy"] = {"mode": "configuration"}
    sheet["facts"]["body"]["torso_geometry"] = {
        "available": True,
        "composer_eligible": False,
    }

    out = phase4b15._apply_partial_upper_body_orientation(
        sheet,
        pose_language=_partial_pose_language(),
    )

    assert out["facts"]["body"]["configuration"][0]["composer_text"] == "arm bent at the elbow"
    assert len(out["facts"]["body"]["configuration"]) == 1
    assert (
        out["facts"]["body"]["partial_upper_body_orientation_adjudication"]["reason"]
        == "existing_local_configuration_already_composer_visible"
    )


def test_partial_upper_body_orientation_requires_strong_pose_language_authority():
    sheet = _sheet([], left=("shoulder",), right=("shoulder",))
    sheet["policy"] = {"mode": "configuration"}
    sheet["facts"]["body"]["torso_geometry"] = {
        "available": True,
        "composer_eligible": False,
    }

    out = phase4b15._apply_partial_upper_body_orientation(
        sheet,
        pose_language=_partial_pose_language(authority=0.7),
    )

    assert out["facts"]["body"]["configuration"] == []
    assert (
        out["facts"]["body"]["partial_upper_body_orientation_adjudication"]["reason"]
        == "upper_body_orientation_authority_below_threshold"
    )


def test_large_corroborated_roll_publishes_generic_tilt_when_other_axes_unresolved():
    sheet = {
        "facts": {
            "head_pose": {
                "available": True,
                "horizontal": {"publishable": False},
                "vertical": {"publishable": False},
                "roll": {
                    "publishable": False,
                    "authority": "corroborated",
                    "degrees": 32.7,
                },
            },
        },
    }

    out = phase4b15._apply_head_roll_only_semantics(sheet)
    roll = out["facts"]["head_pose"]["roll"]

    assert roll["publishable"] is True
    assert roll["caption_semantics"]["composer_text"] == "head tilted noticeably"
    assert roll["caption_semantics"]["direction_publishable"] is False


def test_roll_only_semantics_abstains_when_yaw_or_pitch_already_publishes():
    sheet = {
        "facts": {
            "head_pose": {
                "available": True,
                "horizontal": {"publishable": True, "value": "frame_left"},
                "vertical": {"publishable": False},
                "roll": {
                    "publishable": False,
                    "authority": "corroborated",
                    "degrees": 32.7,
                },
            },
        },
    }

    out = phase4b15._apply_head_roll_only_semantics(sheet)

    assert out["facts"]["head_pose"]["roll"]["publishable"] is False
    assert (
        out["facts"]["head_pose"]["roll_only_adjudication"]["reason"]
        == "yaw_or_pitch_already_caption_authoritative"
    )


def test_gaze_context_conflict_forces_surface_abstention_without_promoting_context():
    sheet = {
        "facts": {
            "gaze": {
                "available": True,
                "publishable": True,
                "source": "l2cs",
                "camera_relationship": "off_camera",
                "yaw_deg": 35.6,
                "pitch_deg": -12.8,
                "caption_semantics": {
                    "available": True,
                    "publishable": True,
                    "horizontal": {
                        "composer_value": "frame_left",
                        "publishable": True,
                    },
                    "vertical": {
                        "composer_value": "down",
                        "publishable": True,
                    },
                    "camera_relationship": {
                        "composer_value": "off_camera",
                        "publishable": True,
                    },
                },
            },
        },
        "context_only": {
            "expression_action": [
                {
                    "text": "looking toward camera",
                    "authority": "context_only",
                },
            ],
        },
    }

    out = phase4b15._apply_gaze_semantic_conflict_gate(sheet)
    gaze = out["facts"]["gaze"]
    semantics = gaze["caption_semantics"]

    assert gaze["publishable"] is True
    assert gaze["yaw_deg"] == 35.6
    assert semantics["publishable"] is False
    assert semantics["horizontal"]["composer_value"] is None
    assert semantics["vertical"]["composer_value"] is None
    assert semantics["camera_relationship"]["composer_value"] is None
    gate = gaze["semantic_conflict_adjudication"]
    assert gate["status"] == "conflict_abstain"
    assert gate["context_only_can_create_gaze_authority"] is False
    assert gate["raw_gaze_measurement_preserved"] is True


def test_gaze_context_agreement_does_not_change_specialist_surface():
    sheet = {
        "facts": {
            "gaze": {
                "available": True,
                "publishable": True,
                "camera_relationship": "off_camera",
                "caption_semantics": {
                    "available": True,
                    "publishable": True,
                    "horizontal": {"composer_value": "frame_left", "publishable": True},
                    "camera_relationship": {
                        "composer_value": "off_camera",
                        "publishable": True,
                    },
                },
            },
        },
        "context_only": {
            "expression_action": [{"text": "looking away from the camera"}],
        },
    }

    out = phase4b15._apply_gaze_semantic_conflict_gate(sheet)

    assert out["facts"]["gaze"]["caption_semantics"]["publishable"] is True
    assert (
        out["facts"]["gaze"]["semantic_conflict_adjudication"]["reason"]
        == "no_explicit_camera_relationship_conflict"
    )
