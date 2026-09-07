from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_compact_renderer_budget_v01 import (
    _select_budget_sentences,
)


class CompactRendererBudgetV01Tests(unittest.TestCase):
    def test_compact_selects_complete_sentences_under_hard_budget(self) -> None:
        caption = (
            "V3SUBJ, in a tight crop around the head and upper torso, rests their head on the left fist, "
            "gazing directly at the camera with a calm, thoughtful expression and subtle smile. "
            "They wear clear, round-framed glasses with thin translucent rims and small metallic hinges, "
            "along with a dark brown textured knit cardigan over a light beige ribbed crewneck top. "
            "Small dangling earrings with off-white beads adorn their ears. "
            "Behind them, a multi-paned window reveals a softly blurred outdoor scene with a red vehicle "
            "under overcast light, while a wooden surface appears in the lower right foreground."
        )
        result = _select_budget_sentences(caption, min_words=60, max_words=90)
        words = len(result["caption"].split())
        self.assertGreaterEqual(words, 60)
        self.assertLessEqual(words, 90)
        self.assertTrue(result["caption"].startswith("V3SUBJ,"))
        self.assertIn("multi-paned window", result["caption"])
        self.assertNotIn("Small dangling earrings", result["caption"])
        self.assertTrue(result["caption"].endswith("."))

    def test_scene_and_clothing_outrank_low_value_accessory_inventory(self) -> None:
        caption = (
            "V3SUBJ, in a medium close-up selfie, smiles warmly at the camera while seated with the upper body strongly turned sideways. "
            "They wear a dark navy fleece pullover with a high collar and a small brand patch, paired with light gray pants. "
            "One arm extends toward the lens, revealing detailed tattoos on the forearm and wrist. "
            "White wireless earbuds are in their ears, and a small packaged food item rests on their lap. "
            "The background shows a paved path, green hedge, tree with fresh leaves, and a distant fence under a partly cloudy sky."
        )
        result = _select_budget_sentences(caption, min_words=60, max_words=90)
        self.assertIn("dark navy fleece pullover", result["caption"])
        self.assertIn("The background shows", result["caption"])
        self.assertLessEqual(len(result["caption"].split()), 90)

    def test_medium_can_passthrough_all_sentences_when_already_inside_budget(self) -> None:
        caption = (
            "V3SUBJ, mid-squat indoors, leans with their torso partly turned sideways to the camera. "
            "They wear a white athletic t-shirt and black shorts with a light blue resistance band above the knees. "
            "Their arms are bent and raised in front of the chest with hands clasped. "
            "The camera frames them from the waist to just below the knees at a slight low angle. "
            "A plain light wall and smooth floor form the background."
        )
        result = _select_budget_sentences(caption, min_words=60, max_words=140)
        self.assertEqual(result["caption"], caption)

    def test_first_sentence_is_never_dropped(self) -> None:
        caption = (
            "V3SUBJ reclines on a bed with strong foreshortening and their legs toward the foreground. "
            "They wear dark jeans and a gray shirt. "
            "A lamp and window are visible behind the bed. "
            "Soft light fills the room."
        )
        result = _select_budget_sentences(caption, min_words=20, max_words=40)
        self.assertTrue(result["caption"].startswith("V3SUBJ reclines on a bed"))

    def test_selection_never_truncates_sentence_text(self) -> None:
        caption = (
            "V3SUBJ stands on a beach with their torso turned sideways to the camera. "
            "They wear a striped top and dark jeans. "
            "Gentle waves and beach houses appear behind them under a pale sky."
        )
        result = _select_budget_sentences(caption, min_words=20, max_words=30)
        selected = result["caption"]
        self.assertTrue(selected.endswith("."))
        source_sentences = [value.strip() for value in caption.split(". ")]
        for fragment in selected.split(". "):
            clean = fragment.strip()
            if clean and not clean.endswith("."):
                clean += "."
            normalized_source = [
                value if value.endswith(".") else value + "." for value in source_sentences
            ]
            self.assertIn(clean, normalized_source)


if __name__ == "__main__":
    unittest.main()
