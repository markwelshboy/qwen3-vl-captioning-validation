from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_compact_renderer_semantic_repair_v02 import (
    deterministic_local_repair,
)


class CompactRendererSemanticRepairV02Tests(unittest.TestCase):
    def test_hair_length_location_is_removed_but_transient_state_survives(self) -> None:
        caption = (
            "V3SUBJ reclines on a bed. Their tousled hair falls loosely around their shoulders, "
            "partially framing their face. They gaze calmly at the camera."
        )
        audit = {
            "semantic_policy_audit": {
                "identity_leaks": ["falls loosely around their shoulders"],
                "unauthorized_anatomical_laterality": [],
            }
        }
        result = deterministic_local_repair(caption=caption, audit=audit)
        self.assertTrue(result["changed"])
        self.assertNotIn("around their shoulders", result["caption"])
        self.assertIn("Their tousled hair", result["caption"])
        self.assertIn("strands partially framing their face", result["caption"])
        self.assertEqual(result["edits"][0]["kind"], "remove_hair_length_location")

    def test_unsupported_right_hand_is_neutralized_without_touching_left_fist(self) -> None:
        caption = (
            "V3SUBJ sits at a table with their head resting on their left fist. "
            "Their right hand holds a yellow mug."
        )
        audit = {
            "semantic_policy_audit": {
                "identity_leaks": [],
                "unauthorized_anatomical_laterality": ["right hand"],
            }
        }
        result = deterministic_local_repair(caption=caption, audit=audit)
        self.assertIn("left fist", result["caption"])
        self.assertNotIn("right hand", result["caption"])
        self.assertIn("Their hand holds a yellow mug", result["caption"])
        self.assertEqual(result["edits"][0]["kind"], "neutralize_unsupported_laterality")

    def test_unknown_identity_leak_is_not_guessed_at(self) -> None:
        caption = "V3SUBJ smiles at the camera with a protected identity phrase."
        audit = {
            "semantic_policy_audit": {
                "identity_leaks": ["protected identity phrase"],
                "unauthorized_anatomical_laterality": [],
            }
        }
        result = deterministic_local_repair(caption=caption, audit=audit)
        self.assertFalse(result["changed"])
        self.assertEqual(result["caption"], caption)
        self.assertEqual(result["edits"], [])

    def test_only_audited_side_phrase_is_changed(self) -> None:
        caption = (
            "V3SUBJ rests on their left hand while their right hand holds a mug and "
            "their right knee is visible."
        )
        audit = {
            "semantic_policy_audit": {
                "identity_leaks": [],
                "unauthorized_anatomical_laterality": ["right hand"],
            }
        }
        result = deterministic_local_repair(caption=caption, audit=audit)
        self.assertIn("left hand", result["caption"])
        self.assertIn("right knee", result["caption"])
        self.assertNotIn("right hand", result["caption"])


if __name__ == "__main__":
    unittest.main()
