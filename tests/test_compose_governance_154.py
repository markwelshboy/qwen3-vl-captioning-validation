from __future__ import annotations

import unittest

from qwen_caption_validate.compose_governance_154 import _projection_ok, _render_prompt


class ComposeGovernance154Tests(unittest.TestCase):
    def test_projection_guard_accepts_only_154(self) -> None:
        self.assertTrue(_projection_ok({"projection_revision": "1.5.4"}))
        self.assertFalse(_projection_ok({"projection_revision": "1.5.3"}))
        self.assertFalse(_projection_ok(None))

    def test_prompt_forbids_turning_face_orientation_into_gaze(self) -> None:
        evidence = {
            "projection_revision": "1.5.4",
            "pose_orientation": {
                "subject_geometry_orientation": {
                    "body_orientation": {"orientation": "side_on", "faces_frame": "left"},
                    "face_orientation": {"orientation": "toward_camera"},
                    "head_body_relation": {"relation": "turned_toward_camera"},
                }
            },
            "required_claims": [],
        }
        prompt = _render_prompt(
            "Evidence: {{CAPTION_EVIDENCE_JSON}}",
            evidence,
            "sH1Vx",
            "balanced",
        )
        self.assertIn('"projection_revision": "1.5.4"', prompt)
        self.assertIn("face/head orientation relative to the camera, not eye gaze", prompt)
        self.assertIn("Do not turn `toward_camera` into \"looking at the camera\"", prompt)
        self.assertIn("ONE compact relation", prompt)
        self.assertIn("Do not resurrect `signed_shoulder_nearer_relation`", prompt)
        self.assertIn("Subject-relative camera-center geometry remains audit-only", prompt)


if __name__ == "__main__":
    unittest.main()
