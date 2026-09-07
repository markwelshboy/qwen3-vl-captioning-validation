from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor_v07 import (
    _additional_primary_hair_mentions,
    _filter_primary_mentions,
    _revised_redactions,
    quality_audit,
)


class RichPoseEditorV07Tests(unittest.TestCase):
    def test_secondary_background_child_hair_is_scene_content(self) -> None:
        text = (
            "A woman smiles in an airplane cabin. Further in the background, another child with light hair "
            "is visible wearing a patterned top."
        )
        self.assertEqual(_filter_primary_mentions(text, ["light hair"]), [])
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        audit = quality_audit(text, text, pose)
        self.assertNotIn("intrinsic_identity_leakage", audit["warnings"])

    def test_background_person_dark_hair_is_scene_content(self) -> None:
        text = (
            "A woman smiles near a bright window. In the background, slightly out of focus, another person "
            "with dark hair is visible behind her shoulder."
        )
        self.assertEqual(_filter_primary_mentions(text, ["dark hair"]), [])

    def test_primary_subject_hair_is_still_protected(self) -> None:
        text = "A woman with light hair smiles at the camera while wearing clear-framed glasses."
        self.assertEqual(_filter_primary_mentions(text, ["light hair"]), ["light hair"])

    def test_comma_cut_length_is_protected_without_contaminating_transient_state(self) -> None:
        draft = (
            "Her blonde hair, cut to shoulder length, is swept back from her face, with a few strands framing "
            "her cheeks."
        )
        base = {
            "protected_hair_mentions": ["blonde hair"],
            "protected_other_identity_mentions": [],
            "transient_hair_mentions_to_preserve": [
                "hair, cut to shoulder length, is swept back from her face",
            ],
            "unsupported_anatomical_laterality": [],
        }
        redactions = _revised_redactions(draft, base)
        protected = [value.lower() for value in redactions["protected_hair_mentions"]]
        transient = [value.lower() for value in redactions["transient_hair_mentions_to_preserve"]]
        self.assertIn("cut to shoulder length", protected)
        self.assertFalse(any("shoulder length" in value for value in transient), transient)
        self.assertTrue(any("swept back" in value for value in transient), transient)
        self.assertTrue(any("strands framing" in value for value in transient), transient)

    def test_soft_layered_bob_is_reported_as_context_rich_phrase(self) -> None:
        text = "A woman with hair styled in a soft, layered bob with loose strands framing her face smiles."
        extra = [value.lower() for value in _additional_primary_hair_mentions(text)]
        self.assertIn("soft, layered bob", extra)

        base = {
            "protected_hair_mentions": ["layered", "bob"],
            "protected_other_identity_mentions": [],
            "transient_hair_mentions_to_preserve": ["loose strands"],
            "unsupported_anatomical_laterality": [],
        }
        redactions = _revised_redactions(text, base)
        protected = [value.lower() for value in redactions["protected_hair_mentions"]]
        self.assertIn("soft, layered bob", protected)
        self.assertNotIn("layered", protected)
        self.assertNotIn("bob", protected)

    def test_soft_layered_bob_fails_audit(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A woman wears glasses and a dark top against a gray background. Her hair has loose strands around "
            "her face."
        )
        edited = (
            "A woman with hair styled in a soft, layered bob with loose strands around her face wears glasses "
            "and a dark top against a gray background."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("soft, layered bob", [value.lower() for value in audit["identity_leaks"]])
        self.assertIn("intrinsic_identity_leakage", audit["warnings"])


if __name__ == "__main__":
    unittest.main()
