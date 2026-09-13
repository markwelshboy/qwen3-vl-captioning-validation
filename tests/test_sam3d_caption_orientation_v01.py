from __future__ import annotations

import numpy as np

from qwen_caption_validate.sam3d_caption_orientation_v01 import (
    _caption_summary,
    build_caption_orientation,
)


def _arrays(*, cam_x: float = 0.0, shoulder_yaw_45: bool = False) -> dict[str, np.ndarray]:
    points = np.zeros((70, 3), dtype=np.float64)
    if shoulder_yaw_45:
        s = 2 ** -0.5
        points[5] = [s, -1.0, -s]
        points[6] = [-s, -1.0, s]
    else:
        points[5] = [1.0, -1.0, 0.0]
        points[6] = [-1.0, -1.0, 0.0]
    points[9] = [0.5, -0.5, 0.0]
    points[10] = [-0.5, -0.5, 0.0]
    return {
        "global_rot": np.zeros(3, dtype=np.float64),
        "pred_cam_t": np.array([cam_x, 0.0, 3.0], dtype=np.float64),
        "pred_keypoints_3d": points,
    }


def test_camera_center_reference_differs_from_optical_axis_for_off_axis_subject():
    out = build_caption_orientation(_arrays(cam_x=1.0))
    root = out["body_root_orientation"]
    assert root["optical_axis_yaw_deg"] == 0.0
    assert 18.0 < root["yaw_magnitude_deg"] < 19.0
    assert out["policy"]["caption_reference_is_physical_camera_center"] is True


def test_upper_torso_plane_can_preserve_twist_separate_from_root():
    out = build_caption_orientation(_arrays(shoulder_yaw_45=True))
    root = out["body_root_orientation"]
    upper = out["upper_torso_orientation"]
    summary = out["caption_orientation"]
    assert root["orientation_band"] == "frontal"
    assert 44.0 < upper["yaw_magnitude_deg"] < 46.0
    assert upper["orientation_band"] == "three_quarter"
    assert summary["mode"] == "articulated"
    assert summary["meaningful_twist"] is True


def test_small_bucket_boundary_difference_does_not_invent_articulation():
    root = {"orientation_band": "slightly_angled", "yaw_deg": 34.0, "approx_yaw_deg": 35}
    upper = {"orientation_band": "three_quarter", "yaw_deg": 36.0, "approx_yaw_deg": 35}
    summary = _caption_summary(root, upper)
    assert summary["mode"] == "upper_torso_dominant"
    assert summary["meaningful_twist"] is False
    assert summary["preferred_orientation_band"] == "three_quarter"
