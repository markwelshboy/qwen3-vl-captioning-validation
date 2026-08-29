from __future__ import annotations

import unittest

from qwen_caption_validate.sam3d_camera_diagnostic import build_camera_diagnostic


class Sam3dCameraDiagnostic02Tests(unittest.TestCase):
    def _record(
        self,
        *,
        upper_z: float,
        hip_z: float,
        ankle_z: float | None,
        cam_t=(0.0, 0.5, 0.5),
    ) -> dict:
        selected = {
            "left_shoulder": [-0.2, 0.5, upper_z],
            "right_shoulder": [0.2, 0.5, upper_z],
            "left_hip": [-0.15, 0.0, hip_z],
            "right_hip": [0.15, 0.0, hip_z],
            "neck": [0.0, 0.65, upper_z],
        }
        if ankle_z is not None:
            selected["left_ankle"] = [-0.1, -0.8, ankle_z]
            selected["right_ankle"] = [0.1, -0.8, ankle_z]
        return {
            "schema_version": "sam3d-geometry-probe-0.1",
            "metrics": {"selected_keypoints_xyz": selected},
            "camera": {
                "pred_cam_t": list(cam_t),
                "focal_length": 1200.0,
            },
        }

    def _dwpose(self, visible: list[str]) -> dict:
        return {
            "schema_version": "dwpose-profile-1.0",
            "derived": {
                "target": {
                    "visible_body_landmarks": visible,
                    "pose_extent_hint": "full_length",
                }
            },
        }

    def test_body_axis_metric_uses_camera_translation_and_reports_signed_fraction(self) -> None:
        out = build_camera_diagnostic(
            self._record(upper_z=0.4, hip_z=0.0, ankle_z=-0.2),
            self._dwpose([
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_ankle",
                "right_ankle",
            ]),
        )
        torso = out["body_axis_camera_position"]["torso_hip_to_shoulder"]
        self.assertLess(torso["camera_longitudinal_fraction"], -0.25)
        self.assertLess(torso["camera_longitudinal_angle_deg"], 0.0)

    def test_strong_negative_low_signature_with_visible_torso_can_qualify_low(self) -> None:
        out = build_camera_diagnostic(
            self._record(upper_z=0.4, hip_z=0.0, ankle_z=-0.2),
            self._dwpose([
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]),
        )
        support = out["low_angle_support"]
        self.assertEqual(support["action"], "qualified")
        self.assertEqual(support["qualified_elevation"], "low")
        self.assertEqual(support["confidence_band"], "strong")
        self.assertTrue(out["dwpose_visibility_gate"]["torso_axis_visibility_qualified"])

    def test_same_sam3d_low_signature_without_visible_hips_is_supporting_only(self) -> None:
        out = build_camera_diagnostic(
            self._record(upper_z=0.4, hip_z=0.0, ankle_z=-0.2),
            self._dwpose(["left_shoulder", "right_shoulder"]),
        )
        support = out["low_angle_support"]
        self.assertEqual(support["candidate_elevation"], "low")
        self.assertEqual(support["action"], "supporting")
        self.assertIsNone(support["qualified_elevation"])
        self.assertFalse(out["dwpose_visibility_gate"]["torso_axis_visibility_qualified"])

    def test_positive_or_non_low_geometry_never_auto_qualifies_high(self) -> None:
        out = build_camera_diagnostic(
            self._record(upper_z=-0.4, hip_z=0.0, ankle_z=0.2),
            self._dwpose([
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_ankle",
                "right_ankle",
            ]),
        )
        support = out["low_angle_support"]
        self.assertEqual(support["action"], "withheld")
        self.assertIsNone(support["qualified_elevation"])
        self.assertTrue(out["interpretation_policy"]["categorical_high_disabled"])


if __name__ == "__main__":
    unittest.main()
