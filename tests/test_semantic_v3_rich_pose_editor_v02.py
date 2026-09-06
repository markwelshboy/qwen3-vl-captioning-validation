from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor_v02 import (
    _identity_mentions,
    _mandatory_redactions,
    _transient_hair_mentions,
    _unsupported_laterality_mentions,
    build_editor_input,
    quality_audit,
)


class RichPoseEditorV02Tests(unittest.TestCase):
    def test_hair_length_after_noun_is_detected(self) -> None:
        leaks = _identity_mentions("Her hair is shoulder-length and layered around the face.")
        lowered = [value.lower() for value in leaks]
        self.assertTrue(any("shoulder-length" in value for value in lowered))
        self.assertIn("layered", lowered)

    def test_hair_color_roots_and_tan_are_detected(self) -> None:
        text = "He has dark hair with lighter highlights and darker roots. His hand appears tanned."
        leaks = [value.lower() for value in _identity_mentions(text)]
        self.assertTrue(any("dark hair" in value for value in leaks))
        self.assertTrue(any("lighter highlights" in value for value in leaks))
        self.assertTrue(any("darker roots" in value for value in leaks))
        self.assertTrue(any("appears tanned" in value for value in leaks))

    def test_stable_texture_is_protected_but_transient_arrangement_is_preserved(self) -> None:
        text = "Her shoulder-length blonde wavy hair is pulled back over her ears."
        leaks = [value.lower() for value in _identity_mentions(text)]
        transient = [value.lower() for value in _transient_hair_mentions(text)]
        self.assertTrue(any("shoulder-length" in value for value in leaks))
        self.assertTrue(any("blonde" in value for value in leaks))
        self.assertTrue(any("wavy" in value for value in leaks))
        self.assertTrue(any("hair is pulled back over her ears" in value for value in transient))

    def test_transient_hair_states_are_not_identity_leaks(self) -> None:
        text = "Her hair is messy and damp, with loose strands falling across her forehead."
        self.assertEqual(_identity_mentions(text), [])
        transient = [value.lower() for value in _transient_hair_mentions(text)]
        self.assertTrue(any("hair is messy" in value for value in transient))
        self.assertTrue(any("loose strands" in value for value in transient))

    def test_bun_is_preserved_while_color_and_length_are_protected(self) -> None:
        text = "Her long dark hair is tied into a loose bun."
        leaks = [value.lower() for value in _identity_mentions(text)]
        transient = [value.lower() for value in _transient_hair_mentions(text)]
        self.assertTrue(any("dark hair" in value for value in leaks))
        self.assertTrue(any("long" in value for value in leaks))
        self.assertTrue(any("hair is tied into a" in value and "bun" in value for value in transient))

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

    def test_editor_input_contains_redactions_and_hair_preservation_hints(self) -> None:
        rich = {
            "caption": (
                "A woman with shoulder-length blonde hair pulled back over her ears sits beside a window. "
                "Her left wrist has a watch."
            )
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
        self.assertTrue(redactions["transient_hair_mentions_to_preserve"])
        self.assertEqual(
            [value.lower() for value in redactions["unsupported_anatomical_laterality"]],
            ["left wrist"],
        )
        prompt = built["editor_prompt"].lower()
        self.assertIn("transient/image-specific hair state", prompt)
        self.assertIn("pulled back over her ears", prompt)
        self.assertNotIn("{{mandatory_redactions}}", prompt)

    def test_transient_hair_only_can_pass_gate(self) -> None:
        pose = {"components": {"relations": []}}
        edited = "A woman sits by the window with her hair pulled back over her ears, slightly messy and damp."
        audit = quality_audit("draft", edited, pose)
        self.assertNotIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertTrue(audit["transient_hair_mentions"])

    def test_stable_texture_still_fails_gate_when_transient_state_is_present(self) -> None:
        pose = {"components": {"relations": []}}
        edited = "A woman sits by the window with wavy hair pulled back over her ears."
        audit = quality_audit("draft", edited, pose)
        self.assertIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])

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
