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
