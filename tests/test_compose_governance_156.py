from __future__ import annotations

import unittest

from qwen_caption_validate.compose_governance_156 import _projection_ok, _render_prompt


class ComposeGovernance156Tests(unittest.TestCase):
    def test_projection_guard_accepts_only_156(self) -> None:
        self.assertTrue(_projection_ok({"projection_revision": "1.5.6"}))
        self.assertFalse(_projection_ok({"projection_revision": "1.5.5"}))
        self.assertFalse(_projection_ok(None))

    def test_prompt_makes_absent_body_orientation_a_hard_negative(self) -> None:
        evidence = {
            "projection_revision": "1.5.6",
            "pose_orientation": {
                "semantic_pose": {"posture": None, "gestures": []},
                "semantic_orientation": {},
                "subject_geometry_orientation": {
                    "schema_version": "caption-subject-geometry-orientation-1.1",
                },
            },
            "required_claims": [],
        }
        prompt = _render_prompt("Evidence: {{CAPTION_EVIDENCE_JSON}}", evidence, "sH1Vx", "balanced")
        self.assertIn("ABSENCE of `body_orientation` is authoritative", prompt)
        self.assertIn("do NOT say the body/torso is slightly angled", prompt)
        self.assertIn("Do not infer body yaw from photographic framing", prompt)
        self.assertIn("Do not enumerate left/right shoulders", prompt)
        self.assertIn("semantic gesture, or support relation", prompt)


if __name__ == "__main__":
    unittest.main()
