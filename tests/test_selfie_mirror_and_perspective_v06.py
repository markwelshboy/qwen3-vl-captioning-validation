import numpy as np

from qwen_caption_validate import selfie_evidence_shadow_v06 as v06
from qwen_caption_validate import selfie_perspective_diagnostic_v01 as diag


def _gestalt(**fields):
    return {"acquisition": fields}


def test_explicit_neutral_mirror_selfie_is_strong_and_publishable():
    evidence = v06._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="mirror selfie in an elevator",
            expression_action=["holding a phone near the face"],
            objects=["black smartphone", "mirror surface"],
        )
    )

    assert evidence["grade"] == "strong"
    assert evidence["supported"] is True
    assert evidence["explicit_mirror_selfie_text"]


def test_generic_selfie_plus_independent_mirror_semantic_is_strong():
    evidence = v06._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="casual selfie indoors",
            expression_action=["selfie-style photo"],
            objects=["black smartphone", "large mirror"],
        )
    )

    assert evidence["grade"] == "strong"
    assert evidence["supported"] is True
    assert evidence["generic_selfie_text"]
    assert evidence["mirror_text"]


def test_mirror_plus_phone_without_selfie_semantic_is_not_publishable():
    evidence = v06._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="person standing in front of a mirror",
            expression_action=["holding a phone with both hands"],
            objects=["smartphone", "mirror"],
        )
    )

    assert evidence["grade"] == "moderate"
    assert evidence["supported"] is False


def test_phone_without_mirror_or_selfie_is_not_mirror_selfie():
    evidence = v06._neutral_mirror_selfie_semantic(
        _gestalt(
            gestalt="person looking at a phone",
            expression_action=["holding phone with both hands"],
            objects=["smartphone"],
        )
    )

    assert evidence["grade"] == "none"
    assert evidence["supported"] is False


def test_mirror_surface_policy_exposes_only_depicted_frame_coordinates():
    policy = v06._mirror_surface_policy(True)

    assert policy["active"] is True
    assert policy["spatial_surface_mode"] == "depicted_frame"
    assert policy["directional_reference_system"] == "frame_relative_only"
    assert policy["composer_anatomical_laterality"] == "withhold"
    assert "frame left" in policy["allowed_directional_terms"]
    assert "frame right" in policy["allowed_directional_terms"]


def test_mirror_capture_style_overrides_direct_selfie_subtype():
    capture = v06._capture_style(
        {"supported": True},
        {"publishable_selfie": True},
    )

    assert capture["family"] == "selfie"
    assert capture["subtype"] == "mirror_selfie"
    assert capture["spatial_surface_mode"] == "depicted_frame"


def test_camera_distance_geometry_is_scale_free_in_shoulder_widths():
    points = np.zeros((70, 3), dtype=np.float64)
    points[5] = [-0.2, 0.0, 0.0]
    points[6] = [0.2, 0.0, 0.0]

    arrays = {
        "pred_keypoints_3d": points,
        "pred_cam_t": np.array([0.0, 0.0, 2.0], dtype=np.float64),
    }

    out = diag._camera_distance_geometry(arrays)

    assert out["status"] == "available"
    assert abs(out["shoulder_width_model_units"] - 0.4) < 1e-6
    assert abs(out["camera_to_shoulder_midpoint_shoulder_widths"] - 5.0) < 1e-6
    assert abs(out["shoulder_midpoint_optical_depth_shoulder_widths"] - 5.0) < 1e-6


def test_projection_geometry_recovers_scale_normalized_focal_length():
    width, height = 1000, 800
    focal = 1200.0
    cam_t = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    k3 = np.array(
        [
            [-0.2, -0.1, 0.2],
            [0.3, 0.15, 0.4],
            [0.1, -0.25, 0.3],
        ],
        dtype=np.float64,
    )
    cam = k3 + cam_t
    k2 = np.column_stack(
        [
            cam[:, 0] * focal / cam[:, 2] + width / 2,
            cam[:, 1] * focal / cam[:, 2] + height / 2,
        ]
    )

    arrays = {
        "pred_keypoints_3d": k3,
        "pred_keypoints_2d": k2,
        "pred_cam_t": cam_t,
    }

    out = diag._projection_geometry(arrays, width, height)

    assert out["status"] == "available"
    assert abs(out["focal_length_px"] - focal) < 1e-6
    assert abs(out["focal_over_image_width"] - 1.2) < 1e-6
