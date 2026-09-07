from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_editor_v06 import (
    _additional_protected_hair_mentions,
    _revised_redactions,
    quality_audit,
)


class RichPoseEditorV06Tests(unittest.TestCase):
    def test_light_colored_hair_replaces_phantom_red_token(self) -> None:
        draft = (
            "She has shoulder-length, light-colored hair with soft waves, some strands falling "
            "around her face and tucked behind her ears."
        )
        base = {
            "protected_hair_mentions": ["shoulder-length", "red"],
            "protected_other_identity_mentions": [],
            "transient_hair_mentions_to_preserve": [],
            "unsupported_anatomical_laterality": [],
        }
        redactions = _revised_redactions(draft, base)
        lowered = [value.lower() for value in redactions["protected_hair_mentions"]]
        self.assertIn("light-colored hair", lowered)
        self.assertIn("soft waves", lowered)
        self.assertNotIn("red", lowered)

    def test_light_toned_layers_are_protected(self) -> None:
        text = "A woman with light-toned hair styled with soft layers and loose strands framing her face smiles."
        leaks = [value.lower() for value in _additional_protected_hair_mentions(text)]
        self.assertIn("light-toned hair", leaks)
        self.assertIn("soft layers", leaks)

    def test_medium_to_long_dark_brown_and_texture_are_protected(self) -> None:
        text = (
            "Their hair is medium to long, dark brown, with loose strands around the face. "
            "The hair appears dry and naturally tousled, with slight volume and texture."
        )
        leaks = [value.lower() for value in _additional_protected_hair_mentions(text)]
        self.assertIn("medium to long", leaks)
        self.assertIn("dark brown", leaks)
        self.assertTrue(any("texture" in value for value in leaks))

    def test_bob_waves_curls_and_explicit_length_are_protected(self) -> None:
        samples = {
            "Her hair is styled in a short bob, with loose wisps framing her face.": "short bob",
            "Her hair is styled with a side part and soft waves, with strands tucked behind one ear.": "soft waves",
            "His hair falls in soft curls around his face and shoulders.": "soft curls",
            "Her hair is cut to shoulder length and swept back from her face.": "shoulder length",
            "Her hair falls to her shoulders, partially covering her face.": "falls to her shoulders",
        }
        for text, expected in samples.items():
            with self.subTest(text=text):
                leaks = [value.lower() for value in _additional_protected_hair_mentions(text)]
                self.assertTrue(any(expected in value for value in leaks), leaks)

    def test_tattoo_portrait_long_hair_is_not_primary_subject_identity(self) -> None:
        draft = (
            "A man with curly, tousled dark hair has a portrait tattoo depicting a bearded figure "
            "with long hair on his upper arm."
        )
        base = {
            "protected_hair_mentions": ["dark hair", "curly", "long hair", "long"],
            "protected_other_identity_mentions": [],
            "transient_hair_mentions_to_preserve": [],
            "unsupported_anatomical_laterality": [],
        }
        redactions = _revised_redactions(draft, base)
        lowered = [value.lower() for value in redactions["protected_hair_mentions"]]
        self.assertIn("dark hair", lowered)
        self.assertIn("curly", lowered)
        self.assertNotIn("long hair", lowered)
        self.assertNotIn("long", lowered)

    def test_artwork_age_description_is_not_primary_subject_identity(self) -> None:
        draft = (
            "A woman sits at a table. Behind her is a large black-and-white portrait of an older man "
            "wearing sunglasses and a dark coat."
        )
        base = {
            "protected_hair_mentions": [],
            "protected_other_identity_mentions": ["older man"],
            "transient_hair_mentions_to_preserve": [],
            "unsupported_anatomical_laterality": [],
        }
        redactions = _revised_redactions(draft, base)
        self.assertEqual(redactions["protected_other_identity_mentions"], [])

    def test_audit_allows_depicted_long_hair_but_rejects_subject_light_colored_hair(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A man is seated outdoors wearing a black shirt. A portrait tattoo on his arm depicts a bearded "
            "figure with long hair. Green plants and a wooden bench fill the background."
        )
        tattoo_only = quality_audit(draft, draft, pose)
        self.assertNotIn("intrinsic_identity_leakage", tattoo_only["warnings"])

        edited = (
            "A woman with light-colored hair sits at a table wearing glasses and a dark cardigan. "
            "A yellow mug and puzzle pieces are visible in front of her."
        )
        subject_hair = quality_audit(edited, edited, pose)
        self.assertIn("intrinsic_identity_leakage", subject_hair["warnings"])
        self.assertFalse(subject_hair["passes_basic_gate"])

    def test_v05_sideways_pose_conflict_gate_remains_active(self) -> None:
        pose = {
            "caption_ready_phrases": ["Torso strongly turned sideways to the camera."],
            "components": {"relations": []},
        }
        draft = (
            "She stands in a kitchen angled slightly toward the camera. A window, counter, kettle, bowl, "
            "and several cabinets remain clearly visible behind her."
        )
        edited = (
            "She stands in a kitchen angled slightly toward the camera, with her torso strongly turned sideways "
            "to the camera. A window, counter, kettle, bowl, and several cabinets remain clearly visible behind her."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("conflicting_pose_wording", audit["warnings"])


if __name__ == "__main__":
    unittest.main()
