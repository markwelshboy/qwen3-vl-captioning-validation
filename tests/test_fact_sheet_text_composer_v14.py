from qwen_caption_validate import fact_sheet_text_composer_v14 as v14


def _projection(capture=None):
    auth = {}
    if capture is not None:
        auth["capture"] = capture
    return {
        "subject": {},
        "authoritative_facts": auth,
        "omitted_review_conflict_domains": [],
    }


def test_capture_firewall_preserves_local_arm_geometry_when_capture_absent():
    values, withheld = v14._capture_safe_configuration(
        [
            "arm extended forward, holding the camera",
            "arm resting along the body with forearm visible",
        ],
        capture_authorized=False,
    )

    assert values == [
        "arm extended forward",
        "arm resting along the body with forearm visible",
    ]
    assert withheld == ["arm extended forward, holding the camera"]


def test_capture_firewall_drops_pure_capture_mechanism_when_capture_absent():
    values, withheld = v14._capture_safe_configuration(
        [
            "holding the camera",
            "taking a photo",
            "elbow bent near the side of the torso",
        ],
        capture_authorized=False,
    )

    assert values == ["elbow bent near the side of the torso"]
    assert withheld == ["holding the camera", "taking a photo"]


def test_capture_firewall_preserves_camera_holding_when_capture_authorized():
    source = ["arm extended forward, holding the camera"]
    values, withheld = v14._capture_safe_configuration(
        source,
        capture_authorized=True,
    )

    assert values == source
    assert withheld == []


def test_mirror_frontal_torso_is_suppressed():
    body = {
        "configuration": ["hand holding a smartphone near the face"],
        "torso_orientation": {
            "camera_orientation": "frontal",
        },
    }

    changed = v14._suppress_mirror_neutral_torso(body)

    assert changed is True
    assert "torso_orientation" not in body
    assert body["configuration"]


def test_mirror_meaningful_torso_turn_is_preserved():
    body = {
        "torso_orientation": {
            "camera_orientation": "three_quarter",
            "approx_yaw_deg": 40,
            "turn_direction": "frame_right",
        },
    }

    changed = v14._suppress_mirror_neutral_torso(body)

    assert changed is False
    assert body["torso_orientation"]["turn_direction"] == "frame_right"


def test_audit_rejects_camera_holding_without_capture_authority():
    audit = v14._caption_audit(
        "She holds the camera with one arm extended forward.",
        _projection(),
    )

    assert "capture_mechanism_language_without_capture_authority" in audit["violations"]


def test_audit_allows_camera_holding_with_direct_selfie_authority():
    audit = v14._caption_audit(
        "A selfie-style photo with one arm holding the camera.",
        _projection(
            {
                "family": "selfie",
                "subtype": "direct_selfie",
                "composer_text": "selfie-style capture",
            }
        ),
    )

    assert "capture_mechanism_language_without_capture_authority" not in audit["violations"]


def test_audit_rejects_negative_body_geometry_completion():
    audit = v14._caption_audit(
        "A mirror selfie with her torso frontal, with no visible turn or tilt.",
        _projection(
            {
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
                "spatial_surface_mode": "depicted_frame",
                "directional_reference_system": "frame_relative_only",
                "anatomical_laterality_policy": "withhold",
            }
        ),
    )

    assert "unsupported_negative_body_geometry_language" in audit["violations"]


def test_audit_rejects_mirror_reflection_tautology():
    audit = v14._caption_audit(
        "A mirror selfie in an elevator. The mirror surface reflects the scene.",
        _projection(
            {
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
                "spatial_surface_mode": "depicted_frame",
                "directional_reference_system": "frame_relative_only",
                "anatomical_laterality_policy": "withhold",
            }
        ),
    )

    assert "mirror_reflection_tautology" in audit["violations"]


def test_audit_rejects_meta_composition_narration():
    audit = v14._caption_audit(
        "The mirror selfie composition emphasizes the phone near her face.",
        _projection(
            {
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
                "spatial_surface_mode": "depicted_frame",
                "directional_reference_system": "frame_relative_only",
                "anatomical_laterality_policy": "withhold",
            }
        ),
    )

    assert "meta_composition_narration" in audit["violations"]


def test_audit_rejects_setting_to_expression_causality():
    audit = v14._caption_audit(
        "Reflective steel walls contribute to a pensive expression.",
        _projection(),
    )

    assert "unsupported_setting_to_mood_causality" in audit["violations"]


def test_audit_rejects_self_contained_moment_narration():
    audit = v14._caption_audit(
        "The image presents a close, self-contained moment.",
        _projection(),
    )

    assert "interpretive_moment_narration" in audit["violations"]


def test_retry_prompt_explicitly_removes_unauthorized_broad_pose():
    prompt = v14._retry_prompt(
        "BASE PROMPT",
        "sH1VX is lying on a gray fabric surface.",
        ["unauthorized_broad_pose:lying"],
    )

    assert "Remove the unsupported broad-pose/posture claim(s) lying entirely." in prompt
    assert "remove that dependent relation too" in prompt
    assert "do not replace" in prompt.lower()



def test_secondary_person_seated_does_not_count_as_primary_pose():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {},
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "sH1VX is shown in a close-up inside an aircraft cabin. "
        "Behind her, a person in a navy shirt and jeans is seated.",
        projection,
    )

    assert "unauthorized_broad_pose:seated" not in audit["violations"]
    assert audit["primary_subject_used_pose_groups"] == []


def test_nonhuman_object_sits_does_not_count_as_primary_seated_pose():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {},
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "She holds a yellow teacup with a blue handle on a saucer, "
        "and a clear glass tumbler sits nearby.",
        projection,
    )

    assert "unauthorized_broad_pose:seated" not in audit["violations"]
    assert audit["primary_subject_used_pose_groups"] == []


def test_local_torso_lying_does_not_count_as_primary_global_pose():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {},
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "Her torso is lying flat on a surface with visible white bedding.",
        projection,
    )

    assert "unauthorized_broad_pose:lying" not in audit["violations"]
    assert audit["primary_subject_used_pose_groups"] == []


def test_primary_subject_seated_still_requires_authority():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {},
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "sH1VX is shown in a close-up. She is seated beside a window.",
        projection,
    )

    assert "unauthorized_broad_pose:seated" in audit["violations"]
    assert audit["primary_subject_used_pose_groups"] == ["seated"]


def test_primary_subject_seated_is_allowed_when_broad_pose_authorizes_it():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {
                "broad_pose": "seated",
            },
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "sH1VX is shown in a medium shot. She is seated beside a window.",
        projection,
    )

    assert "unauthorized_broad_pose:seated" not in audit["violations"]
    assert audit["primary_subject_allowed_pose_groups"] == ["seated"]


def test_mirror_phone_hardware_surface_strips_triple_camera_sentence_variant():
    removed = []
    value = {
        "objects": [
            "black smartphone",
            "the smartphone features a triple camera",
        ]
    }

    out = v14._mirror_phone_hardware_surface(value, removed)

    assert out["objects"] == ["black smartphone", "the smartphone"]
    assert removed == ["the smartphone features a triple camera"]


def test_mirror_phone_hardware_audit_rejects_triple_camera_variant():
    projection = {
        "subject": {},
        "authoritative_facts": {
            "capture": {
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
            }
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "A mirror selfie in an elevator. The smartphone features a triple camera.",
        projection,
    )

    assert "mirror_phone_hardware_leak" in audit["violations"]
    assert audit["mirror_phone_hardware_phrases"] == ["triple camera"]


def test_audit_rejects_setting_to_cozy_feel_causality():
    audit = v14._caption_audit(
        "The wooden room gives the setting a cozy, lived-in feel.",
        _projection(),
    )

    assert "unsupported_setting_to_mood_causality" in audit["violations"]



def test_negative_visibility_suffix_is_removed_but_positive_geometry_survives():
    out = v14._strip_negative_visibility_contact(
        "arm extended outward with elbow bent, though the hand is not visible in the crop"
    )

    assert out == "arm extended outward with elbow bent"


def test_pure_negative_contact_relation_is_withheld():
    out = v14._strip_negative_visibility_contact(
        "no contact with her face or chin"
    )

    assert out is None


def test_configuration_sanitizer_removes_negative_visibility_and_contact_bookkeeping():
    values, sanitized = v14._sanitize_negative_visibility_contact_configuration(
        [
            "arm extended outward with elbow bent",
            "hand not visible in crop",
            "no contact with her face or chin",
        ]
    )

    assert values == ["arm extended outward with elbow bent"]
    assert sanitized == [
        "hand not visible in crop",
        "no contact with her face or chin",
    ]


def test_audit_rejects_negative_visibility_and_contact_prose():
    audit = v14._caption_audit(
        "Her arm is extended outward with the elbow bent, though the hand is not visible "
        "in the crop and there is no contact with her face or chin.",
        _projection(),
    )

    assert "negative_visibility_or_contact_language" in audit["violations"]
    assert any(
        "hand is not visible" in phrase
        for phrase in audit["negative_visibility_contact_phrases"]
    )
    assert any(
        "no contact with her face or chin" in phrase
        for phrase in audit["negative_visibility_contact_phrases"]
    )



def test_negative_contact_subject_variant_is_removed_from_suffix():
    out = v14._strip_negative_visibility_contact(
        "arm extends outward with the elbow bent, though no hand or fist makes contact with her face or chin"
    )

    assert out == "arm extends outward with the elbow bent"


def test_audit_rejects_no_hand_or_fist_makes_contact_variant():
    audit = v14._caption_audit(
        "Her arm extends outward with the elbow bent, though no hand or fist makes contact with her face or chin.",
        _projection(),
    )

    assert "negative_visibility_or_contact_language" in audit["violations"]
    assert any(
        "no hand or fist makes contact with her face or chin" in phrase
        for phrase in audit["negative_visibility_contact_phrases"]
    )



def test_primary_subject_squatting_is_recorded_when_authorized():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {
                "broad_pose": "squatting",
            },
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "sH1VX is framed from around the shoulders through the knees, "
        "squatting with her torso bent forward.",
        projection,
    )

    assert "unauthorized_broad_pose:squatting" not in audit["violations"]
    assert audit["primary_subject_allowed_pose_groups"] == ["squatting"]
    assert audit["primary_subject_used_pose_groups"] == ["squatting"]


def test_primary_subject_squatting_requires_authority():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {},
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "sH1VX is shown in a medium shot. She is squatting with both knees bent.",
        projection,
    )

    assert "unauthorized_broad_pose:squatting" in audit["violations"]
    assert audit["primary_subject_used_pose_groups"] == ["squatting"]


def test_secondary_person_squatting_does_not_count_as_primary_pose():
    projection = {
        "subject": {
            "trigger_token": "sH1VX",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "body": {},
        },
        "omitted_review_conflict_domains": [],
    }

    audit = v14._caption_audit(
        "sH1VX is shown in a close-up. Behind her, a person is squatting near a bench.",
        projection,
    )

    assert "unauthorized_broad_pose:squatting" not in audit["violations"]
    assert audit["primary_subject_used_pose_groups"] == []
