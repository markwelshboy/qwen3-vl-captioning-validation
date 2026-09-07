from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_repair import _repairable
from qwen_caption_validate.semantic_v3_rich_pose_repair_v04 import (
    _local_attachment_leaks,
    quality_audit,
)


class RichPoseRepairV04Tests(unittest.TestCase):
    def test_facial_hair_sentence_is_not_bad_attachment(self) -> None:
        text = (
            "Facial hair is present as a light stubble along the jawline and upper lip, "
            "with a faint mustache and goatee."
        )
        self.assertEqual(_local_attachment_leaks(text), [])

    def test_hair_featuring_beard_still_is_bad_attachment(self) -> None:
        text = (
            "A man has hair styled with volume, featuring a short, well-groomed beard and mustache."
        )
        self.assertTrue(_local_attachment_leaks(text))

    def test_light_colored_hair_is_repair_candidate(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A woman with shoulder-length, light-colored hair sits at a table wearing glasses and a cardigan. "
            "A mug, saucer, puzzle pieces, and window remain visible around her."
        )
        edited = (
            "A woman with light-colored hair sits at a table wearing glasses and a cardigan. "
            "A mug, saucer, puzzle pieces, and window remain visible around her."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertTrue(_repairable(audit))

    def test_tattoo_portrait_long_hair_passes(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        text = (
            "A man sits outdoors wearing a black tank top. A portrait tattoo on his arm depicts a bearded figure "
            "with long hair. Green foliage and a wooden bench fill the background."
        )
        audit = quality_audit(text, text, pose)
        self.assertNotIn("intrinsic_identity_leakage", audit["warnings"])
        self.assertTrue(audit["passes_basic_gate"])

    def test_governed_fist_still_authorizes_same_side_hand(self) -> None:
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
        draft = (
            "A woman sits near a large window wearing a patterned top and rests her head on a fist. "
            "A red car is blurred outside."
        )
        edited = (
            "A woman sits near a large window wearing a patterned top. Her left hand supports her head, "
            "with the fist under the jaw. A red car is blurred outside."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertNotIn("unsupported_anatomical_laterality", audit["warnings"])


if __name__ == "__main__":
    unittest.main()
