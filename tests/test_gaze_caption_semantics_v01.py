from __future__ import annotations

from qwen_caption_validate.gaze_caption_semantics_v01 import build_gaze_caption_semantics


def _gaze(yaw: float, *, horizontal: str = "frame_left", vertical: str = "center", camera: str = "uncertain") -> dict:
    return {
        "available": True,
        "publishable": True,
        "horizontal": horizontal,
        "vertical": vertical,
        "camera_relationship": camera,
        "yaw_deg": yaw,
        "pitch_deg": -8.38,
    }


def test_00051_like_gaze_is_near_center_and_not_caption_publishable():
    out = build_gaze_caption_semantics(_gaze(14.602951275676432))
    assert out["horizontal"]["semantic_class"] == "near_center"
    assert out["horizontal"]["composer_value"] is None
    assert out["horizontal"]["publishable"] is False
    assert out["vertical"]["semantic_class"] == "near_center"
    assert out["camera_relationship"]["publishable"] is False
    assert out["publishable"] is False


def test_mild_single_model_lateral_gaze_is_retained_but_suppressed():
    out = build_gaze_caption_semantics(_gaze(20.0))
    assert out["horizontal"]["semantic_class"] == "mild_lateral_uncorroborated"
    assert out["horizontal"]["raw_yaw_deg"] == 20.0
    assert out["horizontal"]["composer_value"] is None
    assert out["publishable"] is False


def test_clear_lateral_gaze_is_caption_publishable():
    out = build_gaze_caption_semantics(_gaze(25.0))
    assert out["horizontal"]["semantic_class"] == "clear_lateral"
    assert out["horizontal"]["composer_value"] == "frame_left"
    assert out["horizontal"]["publishable"] is True
    assert out["publishable"] is True


def test_clear_rightward_gaze_preserves_direction():
    out = build_gaze_caption_semantics(_gaze(-31.0, horizontal="frame_right"))
    assert out["horizontal"]["composer_value"] == "frame_right"
    assert out["horizontal"]["publishable"] is True


def test_resolved_camera_relationship_can_publish_without_lateral_axis():
    out = build_gaze_caption_semantics(_gaze(4.0, horizontal="center", camera="toward_camera"))
    assert out["horizontal"]["publishable"] is False
    assert out["camera_relationship"]["composer_value"] == "toward_camera"
    assert out["publishable"] is True
