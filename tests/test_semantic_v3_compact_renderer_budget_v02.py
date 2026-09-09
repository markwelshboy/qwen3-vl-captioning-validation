from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_compact_renderer_budget_v02 import (
    _select_budget_sentences,
)


class CompactRendererBudgetV02Tests(unittest.TestCase):
    def test_00044_shape_keeps_clothing_and_scene_by_shaving_only_scene_tail(self) -> None:
        caption = (
            "V3SUBJ, in a tight crop around the head and upper torso, rests their head on the left fist, "
            "gazing directly at the camera with a calm, thoughtful expression and subtle smile. "
            "They wear clear, round-framed glasses with thin translucent rims and small metallic hinges, "
            "along with a dark brown textured knit cardigan over a light beige ribbed crewneck top. "
            "Small dangling earrings with off-white or pale gold beads adorn their ears. "
            "Behind them, a large multi-paned window reveals a softly blurred outdoor scene with a red vehicle "
            "and indistinct buildings under overcast light, while a wooden surface appears in the lower right foreground."
        )
        result = _select_budget_sentences(caption, min_words=60, max_words=90)
        self.assertLessEqual(len(result["caption"].split()), 90)
        self.assertIn("round-framed glasses", result["caption"])
        self.assertIn("dark brown textured knit cardigan", result["caption"])
        self.assertIn("multi-paned window", result["caption"])
        self.assertIn("red vehicle", result["caption"])
        self.assertNotIn("Small dangling earrings", result["caption"])
        self.assertNotIn("wooden surface appears", result["caption"])
        self.assertIn(3, result["compacted_sentence_indices"])

    def test_new_family_coverage_prevents_repeated_pose_detail_from_beating_clothing(self) -> None:
        caption = (
            "V3SUBJ, seated between two moss-covered trees in woodland, has their torso turned sideways while gazing upward. "
            "They wear a dark navy jacket, matching trousers, and brown sneakers. "
            "Their hands are clasped in their lap, showing a ring. "
            "Tousled hair falls around the face while the head remains tilted. "
            "The trees flank them tightly in a medium-full shot under soft light."
        )
        result = _select_budget_sentences(caption, min_words=45, max_words=58)
        self.assertIn("dark navy jacket", result["caption"])
        self.assertIn("hands are clasped", result["caption"])
        self.assertNotIn("Tousled hair falls", result["caption"])

    def test_held_object_outweighs_generic_background(self) -> None:
        caption = (
            "V3SUBJ, in a close-up portrait, sits at a table with their head resting on the left fist. "
            "They wear round glasses and a chunky brown cardigan over a cream top. "
            "V3SUBJ holds a large yellow ceramic mug with a blue handle. "
            "A plain gray wall and smooth floor form the background under soft lighting."
        )
        result = _select_budget_sentences(caption, min_words=35, max_words=50)
        self.assertIn("yellow ceramic mug", result["caption"])
        self.assertNotIn("plain gray wall", result["caption"])

    def test_distinctive_scene_anchor_beats_minor_accessory_inventory(self) -> None:
        caption = (
            "V3SUBJ, in a medium close-up selfie, sits sideways while one arm extends toward the lens holding the device. "
            "They wear a dark navy fleece pullover and light gray pants. "
            "Their hair is pulled back with loose strands around the face. "
            "White wireless earbuds are visible, with a small packaged item on their lap. "
            "Behind them are a paved path, green hedge, tree, chain-link fence, and basketball hoop."
        )
        result = _select_budget_sentences(caption, min_words=55, max_words=72)
        self.assertIn("dark navy fleece pullover", result["caption"])
        self.assertIn("basketball hoop", result["caption"])
        self.assertNotIn("wireless earbuds", result["caption"])

    def test_crop_specific_framing_beats_plain_wall_fill(self) -> None:
        caption = (
            "V3SUBJ, mid-squat indoors, leans with their torso sideways and knees deeply bent. "
            "They wear a white athletic t-shirt and black shorts with a blue resistance band above the knees. "
            "Their arms are bent with hands clasped near the chest. "
            "The camera captures them from waist to below the knees at a slight low angle. "
            "A plain light wall and smooth floor frame the scene."
        )
        result = _select_budget_sentences(caption, min_words=45, max_words=62)
        self.assertIn("from waist to below the knees", result["caption"])
        self.assertNotIn("plain light wall", result["caption"])

    def test_first_sentence_is_never_compacted(self) -> None:
        first = (
            "V3SUBJ, seated sideways, rests their chin on a fist while their head turns toward the camera, "
            "while the lower body remains outside the crop."
        )
        caption = first + " Behind them, a window shows a red car, while a table edge appears in the foreground."
        result = _select_budget_sentences(caption, min_words=20, max_words=40)
        self.assertTrue(result["caption"].startswith(first))
        self.assertNotIn(0, result["compacted_sentence_indices"])


if __name__ == "__main__":
    unittest.main()
