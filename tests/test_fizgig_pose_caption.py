from __future__ import annotations

import unittest

from qwen_caption_validate.fizgig_pose_caption import _render_prompt, validate_caption


class FizgigPoseCaptionTests(unittest.TestCase):
    def test_renders_feminine_variables(self) -> None:
        template = (
            '[TRIGGER] [GENDER_GRAMMAR] [SUBJECT_PRONOUN] [OBJECT_PRONOUN] '
            '[POSSESSIVE_PRONOUN] [REFLEXIVE_PRONOUN] [PROTECTED_TRAITS]'
        )
        text, variables = _render_prompt(
            template,
            trigger="sH1VX",
            grammar="feminine",
            protected_traits=["tattoo", "nose ring"],
        )
        self.assertEqual(text, "sH1VX feminine she her her herself tattoo, nose ring")
        self.assertEqual(variables["TRIGGER"], "sH1VX")

    def test_valid_caption(self) -> None:
        result = validate_caption(
            "sH1VX sits angled toward the camera. She wears a dark jacket.",
            "sH1VX",
        )
        self.assertTrue(result["valid"])

    def test_rejects_missing_trigger(self) -> None:
        result = validate_caption("She sits angled toward the camera.", "sH1VX")
        self.assertFalse(result["valid"])
        self.assertTrue(any("missing" in value.lower() for value in result["errors"]))

    def test_rejects_repeated_trigger(self) -> None:
        result = validate_caption(
            "sH1VX sits angled toward the camera. sH1VX wears a dark jacket.",
            "sH1VX",
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any("exactly once" in value.lower() for value in result["errors"]))

    def test_rejects_generic_replacement_subject(self) -> None:
        result = validate_caption("sH1VX, a woman sits near a window.", "sH1VX")
        self.assertFalse(result["valid"])
        self.assertTrue(any("generic replacement" in value.lower() for value in result["errors"]))


if __name__ == "__main__":
    unittest.main()
