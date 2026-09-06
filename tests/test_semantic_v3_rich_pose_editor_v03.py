from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor_v03 import (
    _mandatory_redactions,
    _transient_hair_mentions,
    build_editor_input,
    quality_audit,
)


class RichPoseEditorV03Tests(unittest.TestCase):
    def test_bare_hair_color_tokens_are_deduped_when_context_phrase_exists(self) -> None:
        pose = {"components": {"relations": []}}
        redactions = _mandatory_redactions(
            "Her shoulder-length, light brown hair falls forward.", pose
        )
        hair = [value.lower() for value in redactions["protected_hair_mentions"]]
        self.assertIn("light brown hair", hair)
        self.assertIn("shoulder-length", hair)
        self.assertNotIn("light", hair)
        self.assertNotIn("brown", hair)

    def test_contextual_hair_redactions_do_not_target_scene_colors(self) -> None:
        rich = {
            "caption": (
                "A woman with light brown hair stands on light-toned flooring beside a brown cabinet."
            )
        }
        pose = {"caption_ready_phrases": [], "conditional_hints": [], "components": {"relations": []}}
        template = (
            "DRAFT={{RICH_DRAFT}}\nPOSE={{POSE_CORRECTIONS}}\nREDACT={{MANDATORY_REDACTIONS}}"
        )
        built = build_editor_input(rich, pose, template)
        prompt = built["editor_prompt"]
        self.assertIn("HAIR-ONLY", prompt)
        self.assertIn("light brown hair", prompt)
        self.assertIn("light-toned flooring", prompt)
        self.assertIn("brown cabinet", prompt)

    def test_hair_falls_forward_obscuring_face_is_transient(self) -> None:
        values = [
            value.lower()
            for value in _transient_hair_mentions(
                "Her shoulder-length light brown hair falls forward, partially obscuring her face."
            )
        ]
        self.assertTrue(any("falls forward" in value and "obscuring her face" in value for value in values))

    def test_bare_highlights_and_roots_fail_audit_when_still_in_hair_dye_context(self) -> None:
        pose = {"components": {"relations": []}}
        audit = quality_audit(
            "He has hair with lighter highlights and darker roots.",
            "He has hair styled with volume, featuring highlights and roots.",
            pose,
        )
        self.assertIn("highlights", [value.lower() for value in audit["hair_dye_detail_leaks"]])
        self.assertIn("roots", [value.lower() for value in audit["hair_dye_detail_leaks"]])
        self.assertIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])

    def test_lighting_highlights_on_hair_are_allowed(self) -> None:
        pose = {"components": {"relations": []}}
        edited = (
            "Soft window light casts gentle highlights on her hair and the side of her face."
        )
        audit = quality_audit(edited, edited, pose)
        self.assertEqual(audit["hair_dye_detail_leaks"], [])
        self.assertNotIn("intrinsic_identity_leakage", audit["warnings"])

    def test_nonhair_roots_do_not_fail_audit(self) -> None:
        pose = {"components": {"relations": []}}
        audit = quality_audit(
            "A person stands near a tree.",
            "A person stands near exposed tree roots beside a brown wall.",
            pose,
        )
        self.assertEqual(audit["hair_dye_detail_leaks"], [])

    def test_prompt_requires_pose_replacement_not_side_by_side_conflict(self) -> None:
        rich = {"caption": "She is turned slightly toward the camera in a park."}
        pose = {
            "caption_ready_phrases": ["Upper body nearly side-on to the camera."],
            "conditional_hints": [],
            "components": {"relations": []},
        }
        template = (
            "Replace conflicting geometry; do not leave both formulations side by side.\n"
            "DRAFT={{RICH_DRAFT}}\nPOSE={{POSE_CORRECTIONS}}\nREDACT={{MANDATORY_REDACTIONS}}"
        )
        built = build_editor_input(rich, pose, template)
        self.assertIn("Upper body nearly side-on to the camera.", built["editor_prompt"])
        self.assertIn("do not leave both formulations side by side", built["editor_prompt"])


if __name__ == "__main__":
    unittest.main()
