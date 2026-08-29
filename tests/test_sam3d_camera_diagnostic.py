from __future__ import annotations

import unittest

from qwen_caption_validate.sam3d_camera_diagnostic import build_camera_diagnostic


class Sam3dCameraDiagnosticTests(unittest.TestCase):
    def _record(self, *, upper_z: float, hip_z: float, ankle_z: float | None, cam_t=(0.0, 0.0, 4.0)):
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
            "metrics": {
                "selected_keypoints_xyz": selected,
                "signed_depth_fraction_diagnostics": {
                    "hip_mid_to_shoulder_mid": 0.0,
                },
            },
            "camera": {
                "pred_cam_t": list(cam_t),
                "focal_length": 1200.0,
            },
        }

    def test_positive_depth_delta_means_lower_body_is_nearer(self) -> None:
        out = build_camera_diagnostic(self._record(upper_z=0.8, hip_z=0.3, ankle_z=0.0))
        ordering = out["vertical_depth_ordering"]
        self.assertGreater(ordering["shoulder_minus_hip_depth"], 0.0)
        self.assertGreater(ordering["shoulder_minus_ankle_depth"], 0.0)
        self.assertGreater(ordering["ankle_to_shoulder_signed_depth_fraction"], 0.0)

    def test_negative_depth_delta_means_upper_body_is_nearer(self) -> None:
        out = build_camera_diagnostic(self._record(upper_z=0.0, hip_z=0.3, ankle_z=0.8))
        ordering = out["vertical_depth_ordering"]
        self.assertLess(ordering["shoulder_minus_hip_depth"], 0.0)
        self.assertLess(ordering["shoulder_minus_ankle_depth"], 0.0)
        self.assertLess(ordering["ankle_to_shoulder_signed_depth_fraction"], 0.0)

    def test_translation_changes_ray_angles_but_not_relative_depth_order(self) -> None:
        a = build_camera_diagnostic(self._record(upper_z=0.8, hip_z=0.3, ankle_z=0.0, cam_t=(0.0, 0.0, 4.0)))
        b = build_camera_diagnostic(self._record(upper_z=0.8, hip_z=0.3, ankle_z=0.0, cam_t=(0.0, 1.0, 8.0)))
        self.assertEqual(
            a["vertical_depth_ordering"]["shoulder_minus_ankle_depth"],
            b["vertical_depth_ordering"]["shoulder_minus_ankle_depth"],
        )
        self.assertNotEqual(
            a["camera_ray_elevation_deg"]["shoulder_mid"],
            b["camera_ray_elevation_deg"]["shoulder_mid"],
        )

    def test_missing_ankles_remains_unknown_instead_of_using_reconstructed_category(self) -> None:
        out = build_camera_diagnostic(self._record(upper_z=0.5, hip_z=0.2, ankle_z=None))
        ordering = out["vertical_depth_ordering"]
        self.assertIsNone(ordering["shoulder_minus_ankle_depth"])
        self.assertIsNone(ordering["ankle_to_shoulder_signed_depth_fraction"])
        self.assertTrue(out["interpretation_policy"]["categorical_low_high_disabled"])


if __name__ == "__main__":
    unittest.main()
