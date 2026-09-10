import unittest

import numpy as np

from qwen_caption_validate.caption_refiner_pose_ablation import (
    _extract_card_mode,
    _fit_mesh_to_image,
)


class CaptionRefinerPoseAblationTests(unittest.TestCase):
    def test_extract_front_only_mode(self):
        mode, argv = _extract_card_mode([
            "tool",
            "run-dir",
            "--card-mode",
            "front-only",
            "--captions-dir",
            "captions",
        ])
        self.assertEqual(mode, "front-only")
        self.assertNotIn("--card-mode", argv)
        self.assertEqual(argv[1:], ["run-dir", "--captions-dir", "captions"])

    def test_extract_camera_only_mode(self):
        mode, argv = _extract_card_mode([
            "tool",
            "run-dir",
            "--card-mode",
            "camera-only",
            "--captions-dir",
            "captions",
        ])
        self.assertEqual(mode, "camera-only")
        self.assertNotIn("--card-mode", argv)
        self.assertEqual(argv[1:], ["run-dir", "--captions-dir", "captions"])

    def test_extract_crop_mesh_only_equals_form(self):
        mode, argv = _extract_card_mode([
            "tool",
            "run-dir",
            "--card-mode=crop-mesh-only",
            "--captions-dir",
            "captions",
        ])
        self.assertEqual(mode, "crop-mesh-only")
        self.assertEqual(argv[1:], ["run-dir", "--captions-dir", "captions"])

    def test_projection_fit_recovers_linear_camera_map(self):
        key3d = np.array([
            [-1.0, -1.0, -0.4],
            [-1.0,  1.0,  0.2],
            [ 1.0, -1.0,  0.3],
            [ 1.0,  1.0, -0.1],
            [ 0.0, -0.5,  0.7],
            [ 0.0,  0.5, -0.6],
            [ 0.5,  0.0,  0.4],
            [-0.5,  0.0, -0.3],
        ], dtype=np.float64)
        camera = np.array([
            [80.0,  5.0],
            [ 4.0, 70.0],
            [ 8.0, -6.0],
            [160.0, 120.0],
        ], dtype=np.float64)
        key2d = np.column_stack([key3d, np.ones(len(key3d))]) @ camera
        vertices = np.array([
            [-0.25, -0.25, 0.1],
            [ 0.25,  0.25, -0.2],
        ], dtype=np.float64)
        expected = np.column_stack([vertices, np.ones(len(vertices))]) @ camera
        projected, meta = _fit_mesh_to_image(
            vertices,
            {
                "pred_keypoints_3d": key3d,
                "pred_keypoints_2d": key2d,
            },
            320,
            240,
        )
        np.testing.assert_allclose(projected, expected, atol=1e-8)
        self.assertEqual(meta["projection_fit_kind"], "linear_xyz")
        self.assertLess(meta["projection_fit_rms_px"], 1e-6)


if __name__ == "__main__":
    unittest.main()
