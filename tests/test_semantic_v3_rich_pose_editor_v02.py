from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor_v02 import (
    _identity_mentions,
    _mandatory_redactions,
    _unsupported_laterality_mentions,
    build_editor_input,
    quality_audit,
)


class RichPoseEditorV02Tests(unittest.TestCase):
    def test_hair_length_after_noun_is_detected(self) -> None:
        leaks = _identity_mentions("Her hair is shoulder-length and layered around the face.")
        self.assertTrue(any("shoulder-length" in value.lower() for value in leaks))

    def test_hair_color_roots_and_tan_are_detected(self) -> None:
        text = "He has dark hair with lighter highlights and darker roots. His hand appears tanned."
        leaks = [value.lower() for value in _identity_mentions(text)]
        self.assertTrue(any("dark hair" in value for value in leaks))
        self.assertTrue(any("lighter highlights" in value for value in leaks))
        self.assertTrue(any("darker roots" in value for value in leaks))
        self.assertTrue(any("appears tanned" in value for value in leaks))

    def test_unsupported_body_laterality_becomes_mandatory_redaction(self) -> None:
        pose = {"components": {"relations": []}}
        values = _unsupported_laterality_mentions(
            "A mug is held in his right hand while his left hand rests on the keyboard.",
            pose,
        )
        self.assertEqual({value.lower() for value in values}, {"right hand", "left hand"})

    def test_governed_left_fist_is_not_redacted(self) -> None:
        pose = {
            "components": {
                "relations": [
                    {"side": "left", "phrase": "Head resting on the left fist."}
                ]
            }
        }
        values = _unsupported_laterality_mentions("Her head rests on the left fist.", pose)
        self.assertEqual(values, [])

    def test_editor_input_contains_explicit_redaction_directives(self) -> None:
        rich = {
            "caption": "A woman with blonde hair sits beside a window. Her left wrist has a watch."
        }
        pose = {
            "caption_ready_phrases": ["Seated."],
            "conditional_hints": [],
            "components": {"relations": []},
        }
        template = (
            "DRAFT={{RICH_DRAFT}}\nPOSE={{POSE_CORRECTIONS}}\nREDACT={{MANDATORY_REDACTIONS}}"
        )
        built = build_editor_input(rich, pose, template)
        redactions = built["mandatory_redactions"]
        self.assertTrue(redactions["protected_identity_mentions"])
        self.assertEqual(
            [value.lower() for value in redactions["unsupported_anatomical_laterality"]],
            ["left wrist"],
        )
        self.assertNotIn("{{MANDATORY_REDACTIONS}}", built["editor_prompt"])

    def test_meta_redaction_language_fails_gate(self) -> None:
        pose = {"components": {"relations": []}}
        audit = quality_audit(
            "A woman with blonde hair is near a window.",
            "A woman is near a window, with no specific hair color emphasized.",
            pose,
        )
        self.assertIn("meta_redaction_language", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])

    def test_00144_style_output_no_longer_false_passes(self) -> None:
        pose = {
            "components": {
                "relations": [
                    {"side": "left", "phrase": "Head resting on the left fist."}
                ]
            }
        }
        edited = (
            "A woman is tightly framed around the head and upper torso. "
            "Her head rests on the left fist. Her hair is shoulder-length, layered, "
            "with no specific color emphasized. A large window fills the background."
        )
        audit = quality_audit("draft", edited, pose)
        self.assertIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertIn("meta_redaction_language", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])


if __name__ == "__main__":
    unittest.main()
