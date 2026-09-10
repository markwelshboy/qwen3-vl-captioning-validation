import unittest

from qwen_caption_validate.caption_refiner_delta import (
    _format_laterality_facts,
    _frame_side,
    _render_prompt,
)


class CaptionRefinerDeltaTests(unittest.TestCase):
    def test_frame_side_bands(self):
        self.assertEqual(_frame_side(10, 100), "frame-left")
        self.assertEqual(_frame_side(50, 100), "near frame-center")
        self.assertEqual(_frame_side(90, 100), "frame-right")

    def test_laterality_format_keeps_subject_and_frame_sides_distinct(self):
        text = _format_laterality_facts([
            {
                "joint": "wrist",
                "anatomical_side": "subject-left",
                "inside_frame": True,
                "frame_side": "frame-right",
                "source": "dwpose",
            },
            {
                "joint": "wrist",
                "anatomical_side": "subject-right",
                "inside_frame": False,
                "frame_side": "outside photograph",
                "source": "sam3d",
            },
        ])
        self.assertIn("subject-left wrist: frame-right, inside photograph", text)
        self.assertIn("subject-right wrist: outside photograph", text)

    def test_render_prompt_injects_caption_and_laterality(self):
        template = "CAP={{CURRENT_CAPTION}}\nLAT={{LATERALITY_FACTS}}"
        rendered = _render_prompt(template, "A current caption.", "- subject-left wrist: frame-right")
        self.assertEqual(
            rendered,
            "CAP=A current caption.\nLAT=- subject-left wrist: frame-right",
        )


if __name__ == "__main__":
    unittest.main()
