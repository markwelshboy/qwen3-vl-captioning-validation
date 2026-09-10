from qwen_caption_validate.caption_refiner_text_fusion_v142 import (
    _fact_is_explicitly_checkable,
    _semantic_veto_reasons,
    strict_caption_safe_laterality_facts,
)


def test_laterality_not_exposed_for_generic_anatomical_mention_without_frame_claim():
    facts = [
        {"joint": "wrist", "anatomical_side": "subject-right", "inside_frame": True, "frame_side": "frame-left", "source": "dwpose"},
        {"joint": "wrist", "anatomical_side": "subject-left", "inside_frame": True, "frame_side": "frame-right", "source": "dwpose"},
    ]
    safe = strict_caption_safe_laterality_facts(
        facts,
        current_caption="She holds a phone in her right hand while her left hand rests at her side.",
        accepted_pose_candidate=None,
    )
    assert safe == []


def test_laterality_exposed_for_explicit_joint_frame_mapping():
    fact = {"joint": "wrist", "anatomical_side": "subject-right", "inside_frame": True, "frame_side": "frame-left", "source": "dwpose"}
    caption = "Her right hand appears on the left side of the frame."
    assert _fact_is_explicitly_checkable(fact, caption)
    safe = strict_caption_safe_laterality_facts([fact], current_caption=caption, accepted_pose_candidate=None)
    assert len(safe) == 1
    assert safe[0]["frame_side"] == "frame-left"


def test_forward_foot_claim_does_not_unlock_2d_laterality():
    fact = {"joint": "ankle", "anatomical_side": "subject-right", "inside_frame": True, "frame_side": "frame-right", "source": "dwpose"}
    caption = "Her right foot is positioned slightly forward."
    assert not _fact_is_explicitly_checkable(fact, caption)


def test_uncertain_camera_relationship_vetoes_camera_claim():
    hg = {
        "head": {"available": True, "authority": "corroborated", "axis_authority": {}},
        "gaze": {"available": True, "publishable": True, "authority": "high", "camera_relationship": "uncertain"},
    }
    reasons = _semantic_veto_reasons(
        "Her gaze is directed toward image-left rather than straight at the camera.",
        head_gaze=hg,
        pose_gate={"status": "unavailable"},
        laterality_facts=[],
    )
    assert "camera_relationship_not_authoritative" in reasons


def test_unavailable_gaze_vetoes_gaze_language():
    hg = {
        "head": {"available": True, "authority": "corroborated", "axis_authority": {}},
        "gaze": {"available": False, "publishable": False, "camera_relationship": None},
    }
    reasons = _semantic_veto_reasons(
        "Her gaze is directed downward.",
        head_gaze=hg,
        pose_gate={"status": "unavailable"},
        laterality_facts=[],
    )
    assert "gaze_language_without_publishable_gaze" in reasons


def test_body_geometry_without_pose_or_laterality_is_vetoed():
    hg = {"head": {}, "gaze": {}}
    reasons = _semantic_veto_reasons(
        "Her torso is oriented diagonally across the frame.",
        head_gaze=hg,
        pose_gate={"status": "unavailable"},
        laterality_facts=[],
    )
    assert "body_geometry_without_governed_body_evidence" in reasons


def test_reliable_off_camera_gaze_is_not_vetoed():
    hg = {
        "head": {"available": True, "authority": "corroborated", "axis_authority": {}},
        "gaze": {"available": True, "publishable": True, "authority": "high", "camera_relationship": "off_camera"},
    }
    reasons = _semantic_veto_reasons(
        "Her gaze is directed toward image-left and downward, away from the camera.",
        head_gaze=hg,
        pose_gate={"status": "unavailable"},
        laterality_facts=[],
    )
    assert reasons == []
