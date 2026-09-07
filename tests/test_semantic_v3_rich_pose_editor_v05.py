from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor_v05 import (
    _pose_conflict_leaks,
    quality_audit,
)


class RichPoseEditorV05Tests(unittest.TestCase):
    def test_strongly_turned_sideways_rejects_weak_body_turn(self) -> None:
        pose = {
            "caption_ready_phrases": ["Standing with the torso strongly turned sideways to the camera."],
            "components": {"relations": []},
        }
        leaks = _pose_conflict_leaks(
            "She is angled slightly toward the camera while standing with her torso strongly turned sideways to the camera.",
            pose,
        )
        self.assertEqual([value.lower() for value in leaks], ["angled slightly toward the camera"])

    def test_partly_turned_sideways_rejects_weak_body_turn(self) -> None:
        pose = {
            "caption_ready_phrases": ["Torso partly turned sideways to the camera."],
            "components": {"relations": []},
        }
        leaks = _pose_conflict_leaks(
            "Her body is slightly angled toward the camera, with the torso partly turned sideways to the camera.",
            pose,
        )
        self.assertEqual([value.lower() for value in leaks], ["slightly angled toward the camera"])

    def test_nearly_side_on_v04_behavior_is_preserved(self) -> None:
        pose = {
            "caption_ready_phrases": ["Upper body nearly side-on to the camera."],
            "components": {"relations": []},
        }
        leaks = _pose_conflict_leaks(
            "She is turned slightly toward the camera, with her upper body nearly side-on to the camera.",
            pose,
        )
        self.assertEqual([value.lower() for value in leaks], ["turned slightly toward the camera"])

    def test_head_turn_is_still_allowed(self) -> None:
        pose = {
            "caption_ready_phrases": ["Torso strongly turned sideways to the camera."],
            "components": {"relations": []},
        }
        leaks = _pose_conflict_leaks(
            "Her torso is strongly turned sideways to the camera, with her head turned slightly toward the camera.",
            pose,
        )
        self.assertEqual(leaks, [])

    def test_strong_sideways_conflict_fails_quality_gate(self) -> None:
        pose = {
            "caption_ready_phrases": ["Torso strongly turned sideways to the camera."],
            "components": {"relations": []},
        }
        audit = quality_audit(
            "She stands in a kitchen wearing a blue shirt.",
            "She is angled slightly toward the camera while her torso is strongly turned sideways to the camera.",
            pose,
        )
        self.assertIn("conflicting_pose_wording", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])


if __name__ == "__main__":
    unittest.main()
