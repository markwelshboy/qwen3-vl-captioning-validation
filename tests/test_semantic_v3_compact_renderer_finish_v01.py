from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_compact_renderer_finish_v01 import (
    PROFILE_PREFERRED_RANGES,
    _failure_lines,
    _filtered_semantic_audit,
    build_finish_input,
    quality_audit,
)


POSE_NONE = {"caption_ready_phrases": [], "components": {"relations": []}}
POSE_LEFT_FIST = {
    "caption_ready_phrases": ["Head resting on the left fist."],
    "components": {
        "relations": [
            {"side": "left", "phrase": "Head resting on the left fist."},
        ]
    },
}

PROMPT = """{{PROFILE}} {{MIN_WORDS}} {{MAX_WORDS}} {{PREFERRED_MIN_WORDS}} {{PREFERRED_MAX_WORDS}}
{{TRIGGER}} {{SUBJECT_PRONOUN}} {{OBJECT_PRONOUN}} {{POSSESSIVE_PRONOUN}} {{REFLEXIVE_PRONOUN}}
{{POSE_CORRECTIONS}}
{{FAILURES}}
{{CURRENT_CAPTION}}
{{SEMANTIC_CAPTION}}"""


def _caption_with_words(target: int, first: str = "V3SUBJ faces the camera wearing a dark jacket.") -> str:
    words = first.split()
    while len(words) < target:
        words.append("nearby")
    return " ".join(words[:target])


class SemanticV3CompactRendererFinishV01Tests(unittest.TestCase):
    def test_preferred_ranges_leave_headroom(self) -> None:
        self.assertEqual(PROFILE_PREFERRED_RANGES["compact"], (70, 82))
        self.assertEqual(PROFILE_PREFERRED_RANGES["medium"], (112, 128))

    def test_black_headband_is_not_treated_as_black_hair(self) -> None:
        text = (
            "V3SUBJ faces the camera. Their damp hair is slicked back with a thin black headband, "
            "and they hold a black ceramic mug."
        )
        audit = _filtered_semantic_audit(text, text, POSE_NONE)
        self.assertNotIn("black", [str(v).lower() for v in audit.get("identity_leaks") or []])
        self.assertNotIn("intrinsic_identity_leakage", audit.get("warnings") or [])

    def test_explicit_black_hair_remains_identity_leak(self) -> None:
        text = "V3SUBJ faces the camera with black hair pulled back from the face."
        audit = _filtered_semantic_audit(text, text, POSE_NONE)
        self.assertTrue(audit.get("identity_leaks"), audit)
        self.assertIn("intrinsic_identity_leakage", audit.get("warnings") or [])

    def test_unsupported_right_hand_is_named_for_local_neutralization(self) -> None:
        text = _caption_with_words(70, "V3SUBJ sits at a cafe table with their head resting on the left fist while their right hand holds a yellow mug.")
        audit = quality_audit(
            semantic_caption=text,
            rendered_caption=text,
            pose=POSE_LEFT_FIST,
            trigger="V3SUBJ",
            profile="compact",
        )
        self.assertIn("unsupported_anatomical_laterality", audit["warnings"])
        lines = "\n".join(_failure_lines(audit)).lower()
        self.assertIn("right hand", lines)
        self.assertIn("remove only the unsupported side word", lines)

    def test_overlength_finish_input_targets_preferred_range(self) -> None:
        current = _caption_with_words(108)
        semantic = _caption_with_words(180)
        record = {
            "profile": "compact",
            "trigger": "V3SUBJ",
            "grammar_profile": "neutral",
            "semantic_caption": semantic,
            "rendered_caption": current,
        }
        value = build_finish_input(renderer_record=record, pose=POSE_NONE, prompt_template=PROMPT)
        self.assertFalse(value["initial_quality_audit"]["passes_basic_gate"])
        self.assertIn("70", value["finish_prompt"])
        self.assertIn("82", value["finish_prompt"])
        self.assertTrue(any("Overlength" in line for line in value["hard_failures"]))

    def test_in_range_caption_needs_no_hard_failure(self) -> None:
        text = _caption_with_words(78)
        audit = quality_audit(
            semantic_caption=text,
            rendered_caption=text,
            pose=POSE_NONE,
            trigger="V3SUBJ",
            profile="compact",
        )
        self.assertTrue(audit["passes_basic_gate"], audit["warnings"])
        self.assertEqual(audit["warnings"], [])


if __name__ == "__main__":
    unittest.main()
