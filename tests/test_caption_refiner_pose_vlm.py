import unittest

from qwen_caption_validate.caption_refiner_pose_vlm import (
    _caption_from_payload,
    _spatial_terms,
    _split_sentences,
)
from qwen_caption_validate.caption_refiner_server import _page


class CaptionRefinerPoseVlmTests(unittest.TestCase):
    def test_split_sentences_keeps_two_caption_ready_sentences(self):
        text = "The subject is seated at a three-quarter angle. The frame crops around mid-torso."
        self.assertEqual(
            _split_sentences(text),
            [
                "The subject is seated at a three-quarter angle.",
                "The frame crops around mid-torso.",
            ],
        )

    def test_explicit_caption_field_wins(self):
        payload = {
            "caption": "Wrong fallback caption with enough words to be selected.",
            "result": {"caption": "The current Fizgig JSON-derived caption is selected here."},
        }
        text, field = _caption_from_payload(payload, "result.caption")
        self.assertEqual(text, "The current Fizgig JSON-derived caption is selected here.")
        self.assertEqual(field, "result.caption")

    def test_default_caption_field_prefers_final_caption(self):
        payload = {
            "caption": "This is a lower priority caption value for the test.",
            "final_caption": "This is the final caption value that should win here.",
        }
        text, field = _caption_from_payload(payload, None)
        self.assertEqual(text, payload["final_caption"])
        self.assertEqual(field, "final_caption")

    def test_spatial_review_terms_are_only_highlights(self):
        text = "A bag is on frame left behind the subject while a lamp sits on the right."
        terms = [item.lower() for item in _spatial_terms(text)]
        self.assertIn("frame left", terms)
        self.assertIn("behind", terms)
        self.assertIn("right", terms)

    def test_review_page_contains_replacement_controls(self):
        page = _page({
            "records": [{
                "image_key": "example",
                "image_asset": "assets/example.png",
                "pose_card_asset": "assets/example.pose_card.webp",
                "caption_field": "caption",
                "existing_caption": "Sentence one. Sentence two. Sentence three.",
                "spatial_review_terms": [],
                "pose_vlm": {
                    "text": "Replacement one. Replacement two.",
                    "exactly_two_sentences": True,
                },
                "existing_pose_language": None,
            }]
        })
        self.assertIn("Replace first 2 sentences", page)
        self.assertIn("Existing JSON-derived caption", page)
        self.assertIn("Pose + framing reference", page)


if __name__ == "__main__":
    unittest.main()
