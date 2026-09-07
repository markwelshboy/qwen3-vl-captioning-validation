from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_repair import _repairable
from qwen_caption_validate.semantic_v3_rich_pose_repair_v03 import quality_audit


class RichPoseRepairV03Tests(unittest.TestCase):
    def test_strong_sideways_conflict_is_repair_candidate(self) -> None:
        pose = {
            "caption_ready_phrases": ["Torso strongly turned sideways to the camera."],
            "components": {"relations": []},
        }
        draft = (
            "She stands in a kitchen wearing a blue shirt while angled slightly toward the camera. "
            "A window and counter are visible behind her. A metal kettle sits beside a bowl on the counter, "
            "and daylight enters through the window."
        )
        edited = (
            "She stands in a kitchen wearing a blue shirt while angled slightly toward the camera, "
            "with her torso strongly turned sideways to the camera. A window and counter are visible behind her. "
            "A metal kettle sits beside a bowl on the counter, and daylight enters through the window."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("conflicting_pose_wording", audit["warnings"])
        self.assertTrue(_repairable(audit))

    def test_partly_sideways_conflict_is_repair_candidate(self) -> None:
        pose = {
            "caption_ready_phrases": ["Torso partly turned sideways to the camera."],
            "components": {"relations": []},
        }
        draft = (
            "She is slightly angled toward the camera in a detailed indoor scene with shelves and a lamp. "
            "Books, a framed print, and a small plant remain clearly visible across the background."
        )
        edited = (
            "She is slightly angled toward the camera, with her torso partly turned sideways to the camera, "
            "in a detailed indoor scene with shelves and a lamp. Books, a framed print, and a small plant remain "
            "clearly visible across the background."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("conflicting_pose_wording", audit["warnings"])
        self.assertTrue(_repairable(audit))

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
        draft = "She rests her head on a fist beside a large window while wearing a patterned top."
        edited = "Her left hand supports her head, with the fist under the jaw, beside a large window while she wears a patterned top."
        audit = quality_audit(draft, edited, pose)
        self.assertNotIn("unsupported_anatomical_laterality", audit["warnings"])

    def test_local_hair_to_beard_attachment_still_fails(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A man has shoulder-length wavy hair. He has a short beard and wears a black shirt near a beige wall."
        )
        edited = (
            "A man has hair styled with volume, featuring a short beard, and wears a black shirt near a beige wall."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("bad_local_attachment", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])


if __name__ == "__main__":
    unittest.main()
