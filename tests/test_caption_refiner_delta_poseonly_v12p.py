from pathlib import Path

from qwen_caption_validate import caption_refiner_delta_poseonly_v12p as poseonly


def test_pose_candidate_disables_dwpose_and_laterality():
    assert poseonly._disable_dwpose(Path("/tmp/run"), Path("/tmp/dwpose")) is None
    assert poseonly._no_laterality(None, 0, 0, None) == []
    assert poseonly._no_laterality_text([]) == ""


def test_pose_candidate_prompt_has_no_laterality_or_head_gaze_inputs():
    prompt = poseonly.DEFAULT_PROMPT.read_text(encoding="utf-8")
    assert "{{CURRENT_CAPTION}}" in prompt
    assert "{{LATERALITY_FACTS}}" not in prompt
    assert "{{HEAD_GAZE_FACTS}}" not in prompt
    assert "Do NOT use anatomical left/right" in prompt
    assert "Do NOT describe head direction" in prompt


def test_pose_candidate_scope_is_explicit():
    assert poseonly.CANDIDATE_SCOPE == "sam3d_pose_card_body_geometry_only"
    assert "sam3d-only" in poseonly.ARTIFACT_VERSION
