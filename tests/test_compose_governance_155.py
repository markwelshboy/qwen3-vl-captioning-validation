from __future__ import annotations

import unittest

from qwen_caption_validate.compose_governance_155 import _projection_ok, _render_prompt


class ComposeGovernance155Tests(unittest.TestCase):
    def test_projection_guard_accepts_only_155(self) -> None:
        self.assertTrue(_projection_ok({"projection_revision": "1.5.5"}))
        self.assertFalse(_projection_ok({"projection_revision": "1.5.4"}))
        self.assertFalse(_projection_ok(None))

    def test_prompt_scopes_face_geometry_to_yaw_and_preserves_head_pitch(self) -> None:
        evidence = {
            "projection_revision": "1.5.5",
            "pose_orientation": {
                "semantic_pose": {"posture": None, "gestures": []},
                "semantic_orientation": {
                    "head_pitch": {"direction": "down", "magnitude": "moderate"},
                },
                "subject_geometry_orientation": {
                    "body_orientation": {"orientation": "side_on", "faces_frame": "left"},
                    "head_body_relation": {
                        "relation": "turned_toward_camera",
                        "face_yaw_band": "near_frontal",
                        "relation_scope": "compensating_horizontal_yaw_relative_to_body",
                    },
                },
            },
            "required_claims": [],
        }
        prompt = _render_prompt("Evidence: {{CAPTION_EVIDENCE_JSON}}", evidence, "sH1Vx", "balanced")
        self.assertIn("HORIZONTAL HEAD/FACE YAW ONLY", prompt)
        self.assertIn("must not be erased or contradicted", prompt)
        self.assertIn("Near-frontal face yaw is deliberately omitted", prompt)
        self.assertIn("an \"upright posture\"", prompt)
        self.assertIn("relative-YAW fact", prompt)
        self.assertIn("does not establish eye gaze", prompt)


if __name__ == "__main__":
    unittest.main()
