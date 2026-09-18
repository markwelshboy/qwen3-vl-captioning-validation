from qwen_caption_validate import fact_sheet_text_composer_v13 as v13


def test_neutral_head_bookkeeping_is_removed():
    head = {
        "horizontal": "center",
        "vertical": "center",
        "yaw_strength": "frontal",
    }
    assert v13._natural_head_surface(head) is None


def test_directional_head_is_lexicalized_naturally():
    head = {
        "horizontal": "frame_right",
        "vertical": "down",
        "yaw_strength": "strong_turn",
    }
    out = v13._natural_head_surface(head)
    assert out["composer_text"] == (
        "head turned strongly toward frame right and tilted downward"
    )
    assert "yaw" not in out["composer_text"]


def test_vertical_only_head_is_natural():
    head = {
        "horizontal": "center",
        "vertical": "down",
        "yaw_strength": "frontal",
    }
    out = v13._natural_head_surface(head)
    assert out["composer_text"] == "head tilted downward"


def test_frontal_torso_drops_small_numeric_yaw():
    torso = {
        "camera_orientation": "frontal",
        "yaw_magnitude_deg": 9.5,
        "approx_yaw_deg": 10,
    }
    out = v13._compact_neutral_torso(torso)
    assert out == {"camera_orientation": "frontal"}


def test_nonfrontal_torso_keeps_useful_magnitude_and_direction():
    torso = {
        "camera_orientation": "three_quarter",
        "yaw_magnitude_deg": 40.7,
        "approx_yaw_deg": 40,
        "turn_direction": "frame_left",
    }
    assert v13._compact_neutral_torso(torso) == torso


def test_mirror_phone_hardware_detail_is_suppressed():
    assert (
        v13._strip_mirror_phone_hardware(
            "black smartphone with multiple rear cameras"
        )
        == "black smartphone"
    )
    assert (
        v13._strip_mirror_phone_hardware(
            "black smartphone with a triple camera module"
        )
        == "black smartphone"
    )


def test_mirror_low_value_head_phone_alignment_relation_is_removed():
    value = {
        "body": {
            "configuration": [
                "hand holding a smartphone in front of the face",
                "forearm extended forward with hand gripping the device",
                "head positioned above the smartphone, aligned with the device's camera orientation",
            ]
        }
    }
    out = v13._mirror_semantic_economy(value)
    assert out["body"]["configuration"] == [
        "hand holding a smartphone in front of the face",
        "forearm extended forward with hand gripping the device",
    ]


def test_coordinate_policy_language_is_audit_violation():
    projection = {
        "subject": {},
        "authoritative_facts": {
            "capture": {
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
                "spatial_surface_mode": "depicted_frame",
                "directional_reference_system": "frame_relative_only",
                "anatomical_laterality_policy": "withhold",
            }
        },
        "omitted_review_conflict_domains": [],
    }
    audit = v13._caption_audit(
        "A mirror selfie with no anatomical laterality specified.",
        projection,
    )
    assert "coordinate_or_authority_policy_language_leak" in audit["violations"]


def test_frontal_yaw_instrumentation_is_audit_violation():
    projection = {
        "subject": {},
        "authoritative_facts": {},
        "omitted_review_conflict_domains": [],
    }
    audit = v13._caption_audit(
        "Her head has a frontal yaw orientation.",
        projection,
    )
    assert "head_axis_instrumentation_language_leak" in audit["violations"]
