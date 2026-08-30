from __future__ import annotations

import unittest

from qwen_caption_validate.compose_governance_157 import _projection_ok, _render_prompt


class ComposeGovernance157Tests(unittest.TestCase):
    def test_projection_guard_accepts_only_157(self) -> None:
        self.assertTrue(_projection_ok({"projection_revision": "1.5.7"}))
        self.assertFalse(_projection_ok({"projection_revision": "1.5.6"}))
        self.assertFalse(_projection_ok(None))

    def test_prompt_forbids_depth_metrics_and_visibility_checklists(self) -> None:
        evidence = {
            "projection_revision": "1.5.7",
            "pose_orientation": {
                "semantic_pose": {"posture": None, "gestures": []},
                "semantic_orientation": {},
                "subject_geometry_orientation": {},
            },
            "visibility_constraints": {"not_visible": ["left_hip", "right_hip"]},
            "required_claims": [],
        }
        prompt = _render_prompt("Evidence: {{CAPTION_EVIDENCE_JSON}}", evidence, "sH1Vx", "balanced")
        self.assertIn("Shoulder-girdle, pelvis, and torso depth-rotation measurements", prompt)
        self.assertIn("Positive/partial landmark visibility is audit-only", prompt)
        self.assertIn("Do not enumerate left/right shoulders", prompt)
        self.assertIn("ABSENCE of this field is authoritative", prompt)


if __name__ == "__main__":
    unittest.main()
