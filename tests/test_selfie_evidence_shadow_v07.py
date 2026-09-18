from qwen_caption_validate import selfie_evidence_shadow_v07 as v07


def _gestalt(**fields):
    return {"acquisition": fields}


def test_generic_selfie_plus_multiple_rear_cameras_is_mirror_supported():
    evidence = v07._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="selfie taken in an elevator using a smartphone",
            expression_action=["looking at phone screen"],
            objects=["black smartphone with multiple rear cameras"],
        )
    )

    assert evidence["grade"] == "strong"
    assert evidence["supported"] is True
    assert (
        evidence["reason"]
        == "neutral_selfie_semantic_plus_visible_rear_phone_camera_semantic"
    )
    assert evidence["rear_phone_camera_text"]


def test_phone_with_rear_camera_but_without_selfie_semantic_is_not_mirror_selfie():
    evidence = v07._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="person standing in an elevator",
            expression_action=["holding a phone"],
            objects=["black smartphone with multiple rear cameras"],
        )
    )

    assert evidence["supported"] is False


def test_generic_selfie_plus_plain_phone_is_not_upgraded():
    evidence = v07._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="casual selfie indoors",
            expression_action=["holding a phone"],
            objects=["black smartphone"],
        )
    )

    assert evidence["supported"] is False


def test_explicit_mirror_selfie_remains_supported():
    evidence = v07._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="mirror selfie in an elevator",
            objects=["black smartphone"],
        )
    )

    assert evidence["supported"] is True
    assert evidence["reason"] == "explicit_mirror_selfie_language_in_neutral_gestalt"


def test_holding_phone_with_both_hands_negative_control_stays_negative():
    evidence = v07._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="woman in casual indoor setting focused on her smartphone",
            expression_action=["looking down at phone"],
            objects=["blue smartphone held in hands"],
        )
    )

    assert evidence["grade"] == "none"
    assert evidence["supported"] is False
