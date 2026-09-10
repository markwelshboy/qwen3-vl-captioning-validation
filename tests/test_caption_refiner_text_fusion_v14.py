from qwen_caption_validate.caption_refiner_text_fusion_v14 import (
    caption_safe_laterality_facts,
    format_caption_safe_laterality,
    govern_pose_candidate,
    pose_candidate_prompt_text,
    render_prompt,
    validate_output_contract,
)


def test_accepts_one_narrow_trunk_geometry_candidate():
    gate = govern_pose_candidate("Her torso is turned three-quarter toward image-left.")
    assert gate["status"] == "accepted"
    assert gate["domains"] == ["trunk"]
    assert gate["accepted_text"] == "Her torso is turned three-quarter toward image-left."


def test_rejects_multi_sentence_or_multi_domain_pose_prose():
    gate = govern_pose_candidate(
        "Her torso is turned toward image-left. Her right arm is extended forward."
    )
    assert gate["status"] == "rejected"
    assert "not_exactly_one_sentence" in gate["reasons"]
    assert "multiple_body_domains" in gate["reasons"]


def test_rejects_head_gaze_and_semantic_posture_from_pose_candidate():
    gate = govern_pose_candidate(
        "While standing, her head is turned toward image-right and her gaze is off-camera."
    )
    assert gate["status"] == "rejected"
    assert "semantic_posture_language" in gate["reasons"]
    assert "head_or_gaze_language" in gate["reasons"]


def test_rejected_pose_text_is_not_exposed_to_prompt():
    raw = "Her head is turned left while standing beside the sofa."
    gate = govern_pose_candidate(raw)
    prompt_piece = pose_candidate_prompt_text(gate)
    assert gate["status"] == "rejected"
    assert raw not in prompt_piece
    assert "No pose geometry candidate is available" in prompt_piece


def test_laterality_filters_out_of_frame_and_irrelevant_joints():
    facts = [
        {"joint": "wrist", "anatomical_side": "subject-left", "inside_frame": True, "frame_side": "frame-right", "source": "dwpose"},
        {"joint": "wrist", "anatomical_side": "subject-right", "inside_frame": False, "frame_side": "outside photograph", "source": "dwpose"},
        {"joint": "knee", "anatomical_side": "subject-left", "inside_frame": True, "frame_side": "frame-left", "source": "dwpose"},
    ]
    safe = caption_safe_laterality_facts(
        facts,
        current_caption="Her left arm crosses in front of her torso.",
        accepted_pose_candidate=None,
    )
    assert safe == [
        {"joint": "wrist", "anatomical_side": "subject-left", "frame_side": "frame-right", "source": "dwpose"}
    ]
    text = format_caption_safe_laterality(safe)
    assert "subject-left wrist: frame-right" in text
    assert "knee" not in text


def test_prompt_is_text_only_and_has_no_rejected_raw_candidate():
    template = (
        "CAPTION={{CURRENT_CAPTION}}\nPOSE={{POSE_CANDIDATE}}\n"
        "LAT={{LATERALITY_FACTS}}\nHG={{HEAD_GAZE_FACTS}}"
    )
    rendered = render_prompt(
        template,
        current_caption="Current caption.",
        pose_candidate_text="- No pose geometry candidate is available.",
        laterality_text="- No laterality facts.",
        head_gaze_text="- Reliable head yaw: turned toward image-left.",
    )
    assert "Current caption." in rendered
    assert "turned toward image-left" in rendered
    assert "{{" not in rendered


def test_output_contract_accepts_no_correction_or_one_sentence_only():
    assert validate_output_contract("NO_CORRECTION") == (True, None)
    assert validate_output_contract("Her gaze is directed toward image-left.") == (True, None)
    ok, reason = validate_output_contract("Her gaze is left. Her head is turned right.")
    assert not ok
    assert reason == "not_exactly_one_sentence"
