from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor_v04 import (
    _generic_identity_paraphrase_leaks,
    _pose_conflict_leaks,
    quality_audit,
)


class RichPoseEditorV04Tests(unittest.TestCase):
    def test_v03_00018_generic_hair_and_skin_escape_is_detected(self) -> None:
        text = (
            "A man has hair styled with volume, featuring a natural texture and length that falls "
            "to the upper back. The hand shows a natural skin tone."
        )
        leaks = [value.lower() for value in _generic_identity_paraphrase_leaks(text)]
        self.assertTrue(any("natural texture" in value for value in leaks))
        self.assertTrue(any("length that falls to the upper back" in value for value in leaks))
        self.assertTrue(any("natural skin tone" in value for value in leaks))

    def test_hair_slight_wave_is_detected_as_stable_texture(self) -> None:
        leaks = [
            value.lower()
            for value in _generic_identity_paraphrase_leaks(
                "Her hair falls naturally around her face and neck, with a slight wave."
            )
        ]
        self.assertTrue(any("slight wave" in value for value in leaks))

    def test_transient_hair_falling_forward_is_allowed(self) -> None:
        leaks = _generic_identity_paraphrase_leaks(
            "Her hair falls forward, partially obscuring her face."
        )
        self.assertEqual(leaks, [])

    def test_side_on_pose_rejects_weak_body_turn_wording(self) -> None:
        pose = {
            "caption_ready_phrases": ["Upper body nearly side-on to the camera."],
            "components": {"relations": []},
        }
        leaks = _pose_conflict_leaks(
            "She is turned slightly toward the camera, with her upper body nearly side-on to the camera.",
            pose,
        )
        self.assertEqual([value.lower() for value in leaks], ["turned slightly toward the camera"])

    def test_side_on_pose_allows_head_turn_toward_camera(self) -> None:
        pose = {
            "caption_ready_phrases": ["Upper body nearly side-on to the camera."],
            "components": {"relations": []},
        }
        leaks = _pose_conflict_leaks(
            "Her upper body is nearly side-on to the camera, with her head turned slightly toward the camera.",
            pose,
        )
        self.assertEqual(leaks, [])

    def test_v03_style_output_fails_v04_gate(self) -> None:
        pose = {
            "caption_ready_phrases": ["Upper body nearly side-on to the camera."],
            "components": {"relations": []},
        }
        audit = quality_audit(
            "draft",
            "She is turned slightly toward the camera, with her upper body nearly side-on to the camera. "
            "Her hair falls naturally around her face and neck, with a slight wave.",
            pose,
        )
        self.assertIn("generic_identity_paraphrase", audit["warnings"])
        self.assertIn("conflicting_pose_wording", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])

    def test_lighting_highlights_on_hair_remain_allowed(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        audit = quality_audit(
            "draft",
            "Soft window light casts gentle highlights on her hair and the side of her face.",
            pose,
        )
        self.assertEqual(audit["hair_dye_detail_leaks"], [])
        self.assertNotIn("intrinsic_identity_leakage", audit["warnings"])


if __name__ == "__main__":
    unittest.main()
