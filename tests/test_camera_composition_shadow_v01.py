import numpy as np

from qwen_caption_validate import camera_composition_shadow_v01 as shadow
from qwen_caption_validate import selfie_composition_observer_v01 as observer


def _dwpose_shoulders():
    return {
        "derived": {
            "target": {
                "visible_body_landmarks": [
                    "left_shoulder",
                    "right_shoulder",
                ]
            }
        }
    }


def _arrays():
    points = np.zeros((70, 3), dtype=np.float64)
    points[5] = [-0.20, 0.00, 0.20]
    points[6] = [0.20, 0.00, 0.50]
    points[1] = [-0.03, -0.35, 0.10]
    points[2] = [0.03, -0.35, 0.10]
    points[3] = [-0.08, -0.33, 0.12]
    points[4] = [0.08, -0.33, 0.12]
    points[0] = [0.00, -0.35, 0.02]
    points[9] = [-0.15, 0.50, 0.20]
    points[10] = [0.15, 0.50, 0.20]
    points[11] = [-0.15, 0.95, 0.20]
    points[12] = [0.15, 0.95, 0.20]
    points[13] = [-0.15, 1.35, 0.20]
    points[14] = [0.15, 1.35, 0.20]
    points[69] = [0.00, -0.05, 0.15]
    return {
        "global_rot": np.zeros(3, dtype=np.float64),
        "pred_cam_t": np.array([0.0, 0.0, 2.0], dtype=np.float64),
        "pred_keypoints_3d": points,
    }


def test_camera_viewpoint_provisional_elevated_and_downward_bands():
    elevated = shadow._camera_viewpoint({
        "camera_relative_subject": {
            "vertical_vs_eye": 0.20,
            "vertical_band": "above_eye_level",
            "optical_axis_pitch_deg": -18.0,
            "camera_pose_pattern": "camera_above_subject_aimed_down",
        }
    })
    assert elevated["classification"] == "elevated_downward"
    assert elevated["publishable_candidate"] is True

    downward = shadow._camera_viewpoint({
        "camera_relative_subject": {
            "vertical_vs_eye": 0.08,
            "vertical_band": "near_eye_level",
            "optical_axis_pitch_deg": -11.0,
            "camera_pose_pattern": None,
        }
    })
    assert downward["classification"] == "downward_aimed"
    assert downward["publishable_candidate"] is True

    borderline = shadow._camera_viewpoint({
        "camera_relative_subject": {
            "vertical_vs_eye": 0.03,
            "vertical_band": "near_eye_level",
            "optical_axis_pitch_deg": -4.9,
            "camera_pose_pattern": None,
        }
    })
    assert borderline["classification"] == "withheld"
    assert borderline["publishable_candidate"] is False


def test_shoulder_depth_binds_anatomical_nearer_side_only_with_bilateral_observation():
    result = shadow._shoulder_depth(_arrays(), _dwpose_shoulders())
    assert result["status"] == "clear"
    assert result["anatomical_side_nearer"] == "left"
    assert result["distance_and_z_order_agree"] is True
    assert result["composer_text"] == "the left shoulder is nearer the camera"

    withheld = shadow._shoulder_depth(
        _arrays(),
        {"derived": {"target": {"visible_body_landmarks": ["left_shoulder"]}}},
    )
    assert withheld["status"] == "withheld"
    assert withheld["publishable_candidate"] is False


def test_foreground_arm_is_canonicalized_frame_relative_without_anatomical_side():
    result = shadow._canonical_foreground_element({
        "body_region": "arm",
        "extension": "outstretched",
        "frame_region": "lower_frame_left",
        "salience": "large",
        "composer_text": "visible arm in lower left",
    })
    assert result is not None
    assert result["anatomical_side"] is None
    assert result["composer_text"] == (
        "an outstretched arm fills much of the lower frame-left foreground"
    )


def test_ordinary_portrait_does_not_promote_shoulder_depth_just_because_it_is_clear():
    policy = {
        "image_key": "synthetic-00035",
        "visibility": {"broad_pose_supported": False},
    }
    framing = {
        "anatomical_span": {
            "upper_anchor": "head",
            "lower_anchor": "shoulders",
        }
    }
    observer_record = {
        "observation": {
            "capture_style": {
                "label": "portrait_like",
                "basis": ["ordinary close portrait"],
            },
            "foreground_body_elements": [],
        }
    }

    out = shadow.evaluate(
        policy,
        framing,
        observer_record,
        _arrays(),
        _dwpose_shoulders(),
    )

    assert out["nearer_shoulder"]["status"] == "clear"
    assert out["nearer_shoulder"]["publishable_after_governor"] is False
    assert out["hypothesis"]["compositionally_selfie_like"] is False
    assert out["hypothesis"]["package_candidate"] is False


def test_selfie_or_foreground_cue_can_make_clear_shoulder_depth_compositionally_useful():
    policy = {
        "image_key": "synthetic-selfie",
        "visibility": {"broad_pose_supported": False},
    }
    framing = {
        "anatomical_span": {
            "upper_anchor": "head",
            "lower_anchor": "shoulders",
        }
    }
    observer_record = {
        "observation": {
            "capture_style": {
                "label": "selfie_style",
                "basis": ["arm's-length composition"],
            },
            "foreground_body_elements": [
                {
                    "body_region": "arm",
                    "extension": "outstretched",
                    "frame_region": "lower_frame_left",
                    "salience": "large",
                    "composer_text": "an outstretched arm fills the lower frame-left foreground",
                }
            ],
        }
    }

    out = shadow.evaluate(
        policy,
        framing,
        observer_record,
        _arrays(),
        _dwpose_shoulders(),
    )

    assert out["foreground_body_composition"]["present"] is True
    assert out["nearer_shoulder"]["publishable_after_governor"] is True
    assert out["hypothesis"]["compositionally_selfie_like"] is True
    assert out["hypothesis"]["package_candidate"] is True


def test_observer_normalization_never_accepts_anatomical_side_field():
    normalized, unexpected = observer._normalize_payload({
        "capture_style": {
            "label": "selfie_style",
            "basis": ["visible arm's-length composition"],
        },
        "foreground_body_elements": [
            {
                "body_region": "arm",
                "extension": "outstretched",
                "frame_region": "lower_frame_right",
                "salience": "large",
                "composer_text": "an outstretched arm fills the lower frame-right foreground",
                "anatomical_side": "left",
            }
        ],
    })
    element = normalized["foreground_body_elements"][0]
    assert element["anatomical_side"] is None
    assert "foreground_body_elements[0].anatomical_side" in unexpected
