from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_repair import _repairable
from qwen_caption_validate.semantic_v3_rich_pose_repair_v05 import quality_audit


class RichPoseRepairV05Tests(unittest.TestCase):
    def test_secondary_background_hair_does_not_trigger_repair(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        text = (
            "A woman smiles in an airplane cabin. Further in the background, another child with light hair "
            "is visible wearing a patterned top."
        )
        audit = quality_audit(text, text, pose)
        self.assertTrue(audit["passes_basic_gate"])
        self.assertFalse(_repairable(audit))

    def test_soft_layered_bob_is_context_rich_repair_candidate(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A woman wears glasses and a dark sleeveless top against a gray background. Loose strands frame "
            "her face and sweep slightly to the side."
        )
        edited = (
            "A woman with hair styled in a soft, layered bob with loose strands framing her face wears glasses "
            "and a dark sleeveless top against a gray background."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("soft, layered bob", [value.lower() for value in audit["identity_leaks"]])
        self.assertIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertTrue(_repairable(audit))

    def test_comma_cut_to_shoulder_length_is_repair_candidate(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A woman stands in a bright terminal wearing a black top and holding a purple phone. Her hair is "
            "swept back from her face."
        )
        edited = (
            "A woman stands in a bright terminal wearing a black top and holding a purple phone. Her hair, cut "
            "to shoulder length, is swept back from her face."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("cut to shoulder length", [value.lower() for value in audit["identity_leaks"]])
        self.assertTrue(_repairable(audit))

    def test_fist_to_hand_authority_remains(self) -> None:
        pose = {
            "caption_ready_phrases": ["Head resting on the left fist."],
            "components": {
                "relations": [
                    {
                        "relation": "head_supported_by_fist",
                        "phrase": "head resting on the left fist",
                        "side": "left",
                    }
                ]
            },
        }
        draft = "A woman sits near a window with her head supported by a fist and a red car outside."
        edited = "Her left hand supports her head with the fist under the jaw while a red car is visible outside."
        audit = quality_audit(draft, edited, pose)
        self.assertNotIn("unsupported_anatomical_laterality", audit["warnings"])


if __name__ == "__main__":
    unittest.main()
