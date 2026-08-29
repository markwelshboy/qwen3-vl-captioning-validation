from __future__ import annotations

import math
import unittest

import numpy as np

from qwen_caption_validate.sam3d_subject_geometry_diagnostic import (
    CAMERA_SYSTEM_FLIP,
    build_subject_geometry,
    euler_zyx_to_rotmat,
)


class Sam3dSubjectGeometryDiagnosticTests(unittest.TestCase):
    def _canonical_points(self) -> np.ndarray:
        points = np.zeros((70, 3), dtype=np.float64)
        points[0] = [0.0, 1.70, 0.12]  # nose
        points[1] = [0.04, 1.74, 0.08]  # left eye (+X is subject-left)
        points[2] = [-0.04, 1.74, 0.08]
        points[3] = [0.08, 1.70, 0.00]  # left ear
        points[4] = [-0.08, 1.70, 0.00]
        points[5] = [0.20, 1.40, 0.00]  # left shoulder
        points[6] = [-0.20, 1.40, 0.00]
        points[9] = [0.12, 0.90, 0.00]
        points[10] = [-0.12, 0.90, 0.00]
        points[11] = [0.10, 0.50, 0.00]
        points[12] = [-0.10, 0.50, 0.00]
        points[13] = [0.08, 0.10, 0.00]
        points[14] = [-0.08, 0.10, 0.00]
        points[69] = [0.0, 1.50, 0.00]
        return points

    def _arrays(self, *, yaw_deg: float = 0.0, camera_center_body=(0.0, 1.7, 2.0), face_camera_forward=False):
        euler = np.array([0.0, math.radians(yaw_deg), 0.0], dtype=np.float64)
        body_to_camera = CAMERA_SYSTEM_FLIP @ euler_zyx_to_rotmat(euler)
        canonical = self._canonical_points()
        keypoints = (body_to_camera @ canonical.T).T

        if face_camera_forward:
            # Deliberately make the reconstructed face point toward the camera
            # independently of root/body yaw: ear midpoint -> nose = -Z.
            keypoints[3] = [-0.08, -1.70, 0.10]
            keypoints[4] = [0.08, -1.70, 0.10]
            keypoints[1] = [-0.04, -1.74, 0.05]
            keypoints[2] = [0.04, -1.74, 0.05]
            keypoints[0] = [0.0, -1.70, 0.00]

        center = np.asarray(camera_center_body, dtype=np.float64)
        cam_t = -(body_to_camera @ center)
        return {
            "global_rot": euler,
            "pred_cam_t": cam_t,
            "pred_keypoints_3d": keypoints,
        }

    def _dwpose(self, *, face=True):
        visible = ["left_shoulder", "right_shoulder"]
        if face:
            visible += ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        return {
            "derived": {
                "target": {
                    "visible_body_landmarks": visible,
                    "pose_extent_hint": "upper_body",
                }
            }
        }

    def test_frontal_root_rotation_reports_frontal(self) -> None:
        out = build_subject_geometry(self._arrays(yaw_deg=0.0), self._dwpose())
        body = out["body_camera_relation"]
        self.assertAlmostEqual(body["yaw_deg"], 0.0, places=3)
        self.assertEqual(body["orientation_band"], "frontal")
        self.assertIsNone(body["faces_frame"])

    def test_positive_yaw_reports_side_on_facing_frame_right(self) -> None:
        out = build_subject_geometry(self._arrays(yaw_deg=90.0), self._dwpose())
        body = out["body_camera_relation"]
        self.assertAlmostEqual(body["yaw_deg"], 90.0, places=3)
        self.assertEqual(body["orientation_band"], "side_on")
        self.assertEqual(body["faces_frame"], "right")

    def test_side_on_body_with_frontal_face_produces_compound_hint(self) -> None:
        out = build_subject_geometry(
            self._arrays(yaw_deg=-80.0, face_camera_forward=True),
            self._dwpose(face=True),
        )
        compound = out["compound_pose_hint"]
        self.assertIsNotNone(compound)
        self.assertEqual(compound["body_orientation"], "side_on")
        self.assertEqual(compound["body_faces_frame"], "left")
        self.assertEqual(compound["head_relation"], "turned_toward_camera")
        self.assertEqual(compound["face_orientation"], "toward_camera")

    def test_camera_center_is_recovered_in_body_coordinates(self) -> None:
        expected = np.array([-0.45, 1.90, 0.55])
        out = build_subject_geometry(
            self._arrays(yaw_deg=25.0, camera_center_body=expected),
            self._dwpose(),
        )
        recovered = np.array(out["camera_relative_subject"]["center_body_xyz"])
        np.testing.assert_allclose(recovered, expected, atol=1e-6)
        self.assertEqual(out["camera_relative_subject"]["side"], "subject_right")
        self.assertEqual(out["camera_relative_subject"]["vertical_band"], "above_eye_level")

    def test_face_semantics_are_not_observation_qualified_without_dwpose_face(self) -> None:
        out = build_subject_geometry(
            self._arrays(yaw_deg=-80.0, face_camera_forward=True),
            self._dwpose(face=False),
        )
        self.assertFalse(out["dwpose_visibility_gate"]["face_yaw_observation_gate"])
        self.assertIsNone(out["compound_pose_hint"])


if __name__ == "__main__":
    unittest.main()
