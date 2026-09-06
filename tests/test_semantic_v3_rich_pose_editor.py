from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor import build_editor_input, quality_audit


PROMPT = "DRAFT={{RICH_DRAFT}}\nPOSE={{POSE_CORRECTIONS}}"


class SemanticV3RichPoseEditorTests(unittest.TestCase):
    def test_editor_uses_only_caption_ready_pose(self):
        rich = {"caption": "A detailed draft with a large window and a cup."}
        pose = {
            "caption_ready_phrases": ["Head resting on the left fist."],
            "conditional_hints": [{"phrase": "leaning slightly forward", "requires": "semantic_corroboration"}],
            "components": {
                "relations": [
                    {"phrase": "head resting on the left fist", "side": "left"}
                ]
            },
            "injection_priority": "high",
        }
        result = build_editor_input(rich, pose, PROMPT)
        self.assertIn("Head resting on the left fist.", result["editor_prompt"])
        self.assertNotIn("leaning slightly forward", result["editor_prompt"])
        self.assertEqual(len(result["pose_conditional_hints_withheld"]), 1)

    def test_no_pose_correction_does_not_inject_reconstruction(self):
        rich = {"caption": "A detailed portrait with a tapestry in the background."}
        pose = {
            "caption_ready_phrases": [],
            "conditional_hints": [{"phrase": "standing"}],
            "components": {},
        }
        result = build_editor_input(rich, pose, PROMPT)
        self.assertIn("None", result["editor_prompt"])
        self.assertNotIn("standing", result["editor_prompt"])

    def test_quality_audit_flags_identity_leakage(self):
        pose = {"components": {}}
        audit = quality_audit(
            "A detailed caption about a woman by a window with a laptop and cup.",
            "A blonde woman with shoulder-length hair sits by a window with a laptop and cup.",
            pose,
        )
        self.assertIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertTrue(audit["identity_leaks"])

    def test_quality_audit_rejects_unprotected_laterality(self):
        pose = {"components": {}}
        audit = quality_audit(
            "A woman raises a hand under her chin beside a large window.",
            "A woman raises her right hand under her chin beside a large window.",
            pose,
        )
        self.assertIn("unsupported_anatomical_laterality", audit["warnings"])
        self.assertIn("right hand", [value.lower() for value in audit["unauthorized_anatomical_laterality"]])

    def test_quality_audit_allows_governed_fist_side(self):
        pose = {
            "components": {
                "relations": [
                    {"phrase": "head resting on the left fist", "side": "left"}
                ]
            }
        }
        audit = quality_audit(
            "A detailed close portrait beside a large gridded window with a red car outside.",
            "A detailed close portrait with her head resting on the left fist beside a large gridded window with a red car outside.",
            pose,
        )
        self.assertNotIn("unsupported_anatomical_laterality", audit["warnings"])

    def test_quality_audit_flags_overcompression(self):
        pose = {"components": {}}
        draft = " ".join(["detail"] * 100)
        edited = " ".join(["detail"] * 50)
        audit = quality_audit(draft, edited, pose)
        self.assertIn("edited_caption_overcompressed", audit["warnings"])
        self.assertEqual(audit["word_ratio_vs_draft"], 0.5)


if __name__ == "__main__":
    unittest.main()
