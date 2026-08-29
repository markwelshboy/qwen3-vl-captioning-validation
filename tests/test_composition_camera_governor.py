from __future__ import annotations

import unittest

from qwen_caption_validate.composition_camera_governor import govern_camera_elevation


class CompositionCameraGovernorTests(unittest.TestCase):
    def _composition(self, elevation: str, pitch: str = "level", confidence: float = 0.9) -> dict:
        return {
            "schema_valid": True,
            "gestalt": {
                "camera": {
                    "elevation": elevation,
                    "pitch": pitch,
                    "confidence": confidence,
                }
            },
        }

    def _diag(
        self,
        action: str,
        *,
        candidate: str | None = "low",
        qualified: str | None = None,
        band: str = "strong",
    ) -> dict:
        return {
            "low_angle_support": {
                "action": action,
                "candidate_elevation": candidate,
                "qualified_elevation": qualified,
                "confidence_band": band,
                "authority": "test_geometry",
                "reasons": ["test reason"],
                "limitations": [],
            },
            "body_axis_camera_position": {"test": True},
            "vertical_depth_ordering": {"test": True},
            "dwpose_visibility_gate": {"test": True},
        }

    def test_qualified_low_overrides_eye_level(self) -> None:
        out = govern_camera_elevation(
            self._composition("eye_level", "level"),
            self._diag("qualified", qualified="low"),
        )
        self.assertEqual(out["governed_camera"]["elevation"], "low")
        self.assertEqual(out["governed_camera"]["pitch"], "level")
        self.assertEqual(out["action"], "vlm_elevation_overridden_by_geometry")
        self.assertEqual(out["authority"], "dwpose_visible_torso_plus_sam3d_camera_geometry")

    def test_qualified_low_does_not_downgrade_vlm_very_low(self) -> None:
        out = govern_camera_elevation(
            self._composition("very_low", "upward"),
            self._diag("qualified", qualified="low"),
        )
        self.assertEqual(out["governed_camera"]["elevation"], "very_low")
        self.assertEqual(out["action"], "vlm_low_corroborated_by_geometry")

    def test_supporting_low_can_corrob_vlm_low_but_not_create_it(self) -> None:
        low = govern_camera_elevation(
            self._composition("very_low", "upward"),
            self._diag("supporting", candidate="low", band="weak"),
        )
        self.assertEqual(low["governed_camera"]["elevation"], "very_low")
        self.assertEqual(low["action"], "vlm_low_supported_by_non_authoritative_geometry")

        eye = govern_camera_elevation(
            self._composition("eye_level", "level"),
            self._diag("supporting", candidate="low", band="weak"),
        )
        self.assertEqual(eye["governed_camera"]["elevation"], "eye_level")
        self.assertEqual(eye["action"], "vlm_preserved_geometry_support_insufficient_to_override")

    def test_withheld_geometry_preserves_vlm(self) -> None:
        out = govern_camera_elevation(
            self._composition("high", "downward"),
            self._diag("withheld", candidate=None, qualified=None, band="withheld"),
        )
        self.assertEqual(out["governed_camera"]["elevation"], "high")
        self.assertEqual(out["governed_camera"]["pitch"], "downward")
        self.assertEqual(out["action"], "vlm_preserved_geometry_withheld")

    def test_geometry_never_creates_very_low_or_changes_pitch(self) -> None:
        out = govern_camera_elevation(
            self._composition("eye_level", "downward"),
            self._diag("qualified", qualified="low"),
        )
        self.assertEqual(out["governed_camera"]["elevation"], "low")
        self.assertEqual(out["governed_camera"]["pitch"], "downward")
        self.assertNotEqual(out["governed_camera"]["elevation"], "very_low")
        self.assertTrue(out["policy"]["pitch_is_not_governed"])


if __name__ == "__main__":
    unittest.main()
