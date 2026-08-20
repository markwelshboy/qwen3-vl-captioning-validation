from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.sam3d_probe import (
    _dwpose_bbox_pixels,
    _orientation_metrics,
    _out_of_image_plane_angle_deg,
)


class Sam3DProbeTests(unittest.TestCase):
    def test_out_of_image_plane_angle(self) -> None:
        self.assertEqual(_out_of_image_plane_angle_deg(np.array([1.0, 0.0, 0.0])), 0.0)
        self.assertEqual(_out_of_image_plane_angle_deg(np.array([0.0, 0.0, 1.0])), 90.0)
        self.assertEqual(_out_of_image_plane_angle_deg(np.array([1.0, 0.0, 1.0])), 45.0)

    def test_orientation_metrics_separate_depth_rotation_from_torso_tilt(self) -> None:
        names = ["left-shoulder", "right-shoulder", "left-hip", "right-hip"]
        keypoints = np.array(
            [
                [-0.5, 1.0, -0.5],
                [0.5, 1.0, 0.5],
                [-0.5, 0.0, -0.5],
                [0.5, 0.0, 0.5],
            ],
            dtype=np.float64,
        )
        metrics = _orientation_metrics(keypoints, names)
        self.assertEqual(metrics["shoulder_out_of_image_plane_deg"], 45.0)
        self.assertEqual(metrics["hip_out_of_image_plane_deg"], 45.0)
        self.assertEqual(metrics["torso_depth_rotation_proxy_deg"], 45.0)
        self.assertEqual(metrics["torso_depth_tilt_deg"], 0.0)

    def test_dwpose_bbox_padding_and_clamp(self) -> None:
        record = {
            "image_width": 1000,
            "image_height": 500,
            "derived": {
                "target": {
                    "keypoint_bbox": {
                        "x0": 0.1,
                        "y0": 0.2,
                        "x1": 0.9,
                        "y1": 0.8,
                    }
                }
            },
        }
        bbox, meta = _dwpose_bbox_pixels(record, 0.20)
        self.assertEqual(bbox.shape, (1, 4))
        # Raw bbox is [100,100]-[900,400]. Twenty percent padding would exceed
        # the image on left/right, so x clamps to the image boundaries.
        self.assertAlmostEqual(float(bbox[0, 0]), 0.0)
        self.assertAlmostEqual(float(bbox[0, 1]), 40.0)
        self.assertAlmostEqual(float(bbox[0, 2]), 999.0)
        self.assertAlmostEqual(float(bbox[0, 3]), 460.0)
        self.assertEqual(meta["source"], "dwpose_target_keypoint_bbox")


if __name__ == "__main__":
    unittest.main()
