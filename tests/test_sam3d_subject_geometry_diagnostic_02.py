from __future__ import annotations

import math
import unittest

import numpy as np

from qwen_caption_validate import sam3d_subject_geometry_diagnostic as v01
from qwen_caption_validate.sam3d_subject_geometry_diagnostic_02 import build_subject_geometry


class Sam3dSubjectGeometryDiagnostic02Tests(unittest.TestCase):
    def _canonical_points(self) -> np.ndarray:
        points = np.zeros((70, 3), dtype=np.float64)
        points[0] = [0.0, 1.70, 0.12]
        points[1] = [0.04, 1.74, 0.08]
        points[2] = [-0.04, 1.74, 0.08]
        points[3] = [0.08, 1.70, 0.00]
        points[4] = [-0.08, 1.70, 0.00]
        points[5] = [0.20, 1.40, 0.00]
        points[6] = [-0.20, 1.40, 0.00]
        points[9] = [0.12, 0.90, 0.00]
        points[10] = [-0.12, 0.90, 0.00]
        points[11] = [0.10, 0.50, 0.00]
        points[12] = [-0.10, 0.50, 0.00]
        points[13] = [0.08, 0.10, 0.00]
        points[14] = [-0.08, 0.10, 0.00]
        points[69] = [0.0, 1.50, 0.00]
        return points

    def _arrays(self, *, yaw_deg: float, camera_center_body=(0.0, 1.7, 2.0)) -> dict[str, np.ndarray]:
        euler = np.array([0.0, math.radians(yaw_deg), 0.0], dtype=np.float64)
        body_to_camera = v01.CAMERA_SYSTEM_FLIP @ v01.euler_zyx_to_rotmat(euler)
        canonical = self._canonical_points()
        keypoints = (body_to_camera @ canonical.T).T
        center = np.asarray(camera_center_body, dtype=np.float64)
        return {
            "global_rot": euler,
            "pred_cam_t": -(body_to_camera @ center),
            "pred_keypoints_3d": keypoints,
        }

    def _dwpose(self) -> dict:
        return {
            "derived": {
                "target": {
                    "visible_body_landmarks": [
                        "left_shoulder",
                        "right_shoulder",
                        "nose",
                        "left_eye",
                        "right_eye",
                        "left_ear",
                        "right_ear",
                    ]
                }
            }
        }

    def test_25_degrees_is_slightly_angled(self) -> None:
        out = build_subject_geometry(self._arrays(yaw_deg=25.0), self._dwpose())
        self.assertEqual(out["schema_version"], "sam3d-subject-geometry-diagnostic-0.2")
        self.assertEqual(out["body_camera_relation"]["orientation_band"], "slightly_angled")
        self.assertEqual(out["body_camera_relation"]["faces_frame"], "right")

    def test_53_degrees_is_three_quarter(self) -> None:
        out = build_subject_geometry(self._arrays(yaw_deg=53.0), self._dwpose())
        self.assertEqual(out["body_camera_relation"]["orientation_band"], "three_quarter")

    def test_82_degrees_is_side_on(self) -> None:
        out = build_subject_geometry(self._arrays(yaw_deg=-82.0), self._dwpose())
        self.assertEqual(out["body_camera_relation"]["orientation_band"], "side_on")
        self.assertEqual(out["body_camera_relation"]["faces_frame"], "left")

    def test_camera_pattern_name_is_capture_mode_neutral(self) -> None:
        out = build_subject_geometry(
            self._arrays(yaw_deg=0.0, camera_center_body=(0.0, 2.0, 0.6)),
            self._dwpose(),
        )
        camera = out["camera_relative_subject"]
        self.assertIn("camera_pose_pattern", camera)
        self.assertNotIn("selfie_like_geometry", camera)
        self.assertEqual(camera["camera_pose_pattern"], "camera_above_subject_aimed_down")


if __name__ == "__main__":
    unittest.main()
