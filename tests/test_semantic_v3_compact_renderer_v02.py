from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_compact_renderer_v02 import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    _PROFILE_RULES,
    _trigger_audit,
    build_renderer_input,
)


PROMPT = """HARD {{MIN_WORDS}} {{MAX_WORDS}} {{PROFILE_RULES}} {{TRIGGER}} {{GRAMMAR_PROFILE}} {{SUBJECT_PRONOUN}} {{OBJECT_PRONOUN}} {{POSSESSIVE_PRONOUN}} {{REFLEXIVE_PRONOUN}}\n{{POSE_CORRECTIONS}}\n{{SEMANTIC_CAPTION}}"""
POSE = {
    "caption_ready_phrases": ["Torso partly turned sideways to the camera while leaning noticeably."],
    "components": {"relations": []},
}


class SemanticV3CompactRendererV02Tests(unittest.TestCase):
    def test_default_generation_ceiling_is_lower_than_v01(self) -> None:
        self.assertEqual(DEFAULT_MAX_TOKENS, 220)

    def test_compact_profile_is_explicitly_selective_and_hard_capped(self) -> None:
        value = build_renderer_input(
            semantic_caption="A woman stands near a window wearing a dark jacket.",
            pose=POSE,
            trigger="V3SUBJ",
            grammar_profile="neutral",
            profile="compact",
            prompt_template=PROMPT,
        )
        prompt = value["renderer_prompt"]
        self.assertIn("Aim for about 70–80 words and NEVER exceed 90", prompt)
        self.assertIn("Prefer 2–3 sentences", prompt)
        self.assertNotIn("{{PROFILE_RULES}}", prompt)
        self.assertIn("Torso partly turned sideways", prompt)

    def test_medium_profile_is_selective_but_richer(self) -> None:
        self.assertIn("Aim for about 115–125 words and NEVER exceed 140", _PROFILE_RULES["medium"])
        self.assertIn("Prefer 3–5 sentences", _PROFILE_RULES["medium"])

    def test_actual_prompt_says_word_ceiling_is_hard_and_omission_is_expected(self) -> None:
        text = DEFAULT_PROMPT.read_text(encoding="utf-8")
        self.assertIn("HARD MAXIMUM", text)
        self.assertIn("SELECTIVE training caption", text)
        self.assertIn("expected to omit many true but lower-priority details", text)
        self.assertIn("DROP lower-priority details", text)

    def test_natural_trigger_appositive_is_valid(self) -> None:
        audit = _trigger_audit(
            "V3SUBJ, in a close-up portrait, faces the camera while wearing a dark jacket.",
            "V3SUBJ",
        )
        self.assertEqual(audit["warnings"], [])
        self.assertTrue(audit["starts_with_exact_trigger_as_direct_token"])

    def test_detached_generic_identity_after_trigger_remains_invalid(self) -> None:
        audit = _trigger_audit("V3SUBJ, a woman in a dark jacket, faces the camera.", "V3SUBJ")
        self.assertIn("generic_primary_subject_after_trigger", audit["warnings"])

    def test_trigger_may_repeat_for_ownership(self) -> None:
        audit = _trigger_audit(
            "V3SUBJ sits beside another person. V3SUBJ's legs extend into the foreground.",
            "V3SUBJ",
        )
        self.assertEqual(audit["exact_trigger_count"], 2)
        self.assertEqual(audit["warnings"], [])


if __name__ == "__main__":
    unittest.main()
