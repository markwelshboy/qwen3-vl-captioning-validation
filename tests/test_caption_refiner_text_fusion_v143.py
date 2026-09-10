from qwen_caption_validate.caption_refiner_text_fusion_v143 import govern_pose_candidate_v143


def _delta(text: str):
    return {"text": text}


def test_two_domain_torso_upper_limb_candidate_is_accepted():
    gate = govern_pose_candidate_v143(
        _delta(
            "The torso is angled forward with a slight bend at the waist, and both arms are bent at the elbows with the forearms crossing near the midline."
        )
    )
    assert gate["status"] == "accepted"
    assert gate["domains"] == ["trunk", "upper_limb"]


def test_sam3d_only_left_right_language_is_rejected():
    gate = govern_pose_candidate_v143(
        _delta(
            "The right arm is bent at the elbow while the left arm hangs straight along the torso."
        )
    )
    assert gate["status"] == "rejected"
    assert "ungrounded_laterality_language" in gate["reasons"]


def test_three_body_domains_remain_too_broad():
    gate = govern_pose_candidate_v143(
        _delta(
            "The torso is angled forward, both arms are bent across the body, and the knees and feet are staggered in depth."
        )
    )
    assert gate["status"] == "rejected"
    assert "too_many_body_domains" in gate["reasons"]


def test_existing_semantic_guards_are_preserved():
    gate = govern_pose_candidate_v143(
        _delta("The torso is angled forward while she is standing beside a chair.")
    )
    assert gate["status"] == "rejected"
    assert "semantic_posture_language" in gate["reasons"]
    assert "support_or_scene_language" in gate["reasons"]


def test_no_correction_is_preserved():
    gate = govern_pose_candidate_v143(_delta("NO_CORRECTION"))
    assert gate["status"] == "no_correction"
