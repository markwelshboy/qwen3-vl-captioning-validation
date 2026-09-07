from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.semantic_v3_compact_renderer_v01 import (
    PROFILE_BUDGETS,
    _ownership_review,
    _semantic_caption_for_key,
    _trigger_audit,
    _word_count,
    build_renderer_input,
    quality_audit,
)


PROMPT = """{{MIN_WORDS}} {{MAX_WORDS}} {{TRIGGER}} {{GRAMMAR_PROFILE}} {{SUBJECT_PRONOUN}} {{OBJECT_PRONOUN}} {{POSSESSIVE_PRONOUN}} {{REFLEXIVE_PRONOUN}}\n{{POSE_CORRECTIONS}}\n{{SEMANTIC_CAPTION}}"""
POSE = {
    "caption_ready_phrases": ["Upper body nearly side-on to the camera."],
    "components": {"relations": []},
}


def _safe_text_words(target: int) -> str:
    text = "V3SUBJ stands near a large window wearing a dark jacket and glasses while smiling toward the camera."
    filler = " Soft daylight fills the room around nearby furniture."
    while _word_count(text + filler) <= target:
        text += filler
    remaining = target - _word_count(text)
    if remaining:
        text += " " + " ".join(["nearby"] * remaining)
    return text


class SemanticV3CompactRendererV01Tests(unittest.TestCase):
    def test_profile_budgets_are_calibration_targets(self) -> None:
        self.assertEqual(PROFILE_BUDGETS["compact"], (60, 90))
        self.assertEqual(PROFILE_BUDGETS["medium"], (100, 140))

    def test_renderer_input_binds_trigger_and_governed_pose(self) -> None:
        value = build_renderer_input(
            semantic_caption="A woman stands near a window with her upper body turned sideways.",
            pose=POSE,
            trigger="sH1Vx",
            grammar_profile="feminine",
            profile="compact",
            prompt_template=PROMPT,
        )
        self.assertEqual(value["target_word_range"], {"min": 60, "max": 90})
        self.assertEqual(value["trigger"], "sH1Vx")
        self.assertEqual(value["grammar"]["subject_pronoun"], "she")
        self.assertIn("Upper body nearly side-on to the camera.", value["renderer_prompt"])
        self.assertIn("A woman stands near a window", value["renderer_prompt"])

    def test_trigger_can_repeat_for_ownership(self) -> None:
        text = (
            "sH1Vx stands beside another woman near a window. The other woman looks away while sH1Vx's legs "
            "extend toward the foreground and her hands remain close to her lap."
        )
        audit = _trigger_audit(text, "sH1Vx")
        self.assertEqual(audit["exact_trigger_count"], 2)
        self.assertEqual(audit["warnings"], [])

    def test_trigger_must_be_direct_first_subject_and_exact_case(self) -> None:
        audit = _trigger_audit("A portrait shows sh1vx near a window.", "sH1Vx")
        self.assertIn("trigger_missing", audit["warnings"])
        self.assertIn("trigger_case_changed", audit["warnings"])
        self.assertIn("trigger_not_direct_first_subject", audit["warnings"])

    def test_generic_subject_immediately_after_trigger_fails(self) -> None:
        audit = _trigger_audit("sH1Vx, a woman wearing glasses, stands near a window.", "sH1Vx")
        self.assertIn("generic_primary_subject_after_trigger", audit["warnings"])

    def test_ownership_review_flags_single_trigger_after_secondary_person(self) -> None:
        text = (
            "sH1Vx sits beside another woman at a cafe table. The other woman looks toward the window. "
            "The legs extend into the foreground. Soft daylight fills the room."
        )
        review = _ownership_review(text, 1)
        self.assertIn("single_trigger_despite_secondary_person_context", review["review_warnings"])
        self.assertIn("possible_detached_body_ownership", review["review_warnings"])

    def test_compact_quality_gate_accepts_target_length_and_multiple_trigger(self) -> None:
        rendered = _safe_text_words(70)
        rendered += " V3SUBJ's hands remain visible near the jacket."
        # Keep the complete sample within compact range after the ownership re-anchor.
        self.assertGreaterEqual(_word_count(rendered), 60)
        self.assertLessEqual(_word_count(rendered), 90)
        audit = quality_audit(
            semantic_caption=rendered,
            rendered_caption=rendered,
            pose={"caption_ready_phrases": [], "components": {"relations": []}},
            trigger="V3SUBJ",
            profile="compact",
        )
        self.assertTrue(audit["passes_basic_gate"], audit["warnings"])
        self.assertEqual(audit["trigger"]["exact_trigger_count"], 2)

    def test_age_proxy_still_fails_semantic_policy_after_rendering(self) -> None:
        rendered = _safe_text_words(65) + " Visible fine lines remain around the eyes."
        audit = quality_audit(
            semantic_caption=rendered,
            rendered_caption=rendered,
            pose={"caption_ready_phrases": [], "components": {"relations": []}},
            trigger="V3SUBJ",
            profile="compact",
        )
        self.assertIn("age_proxy_identity_leakage", audit["warnings"])

    def test_repair_artifact_wins_over_editor_as_frozen_semantic_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            editor = root / "item.edited_caption.json"
            repair_dir = root / "repair"
            repair_dir.mkdir()
            editor.write_text(json.dumps({"edited_caption": "editor text"}), encoding="utf-8")
            repaired = repair_dir / "item.repaired_caption.json"
            repaired.write_text(json.dumps({"repaired_caption": "repaired text"}), encoding="utf-8")
            caption, path, stage = _semantic_caption_for_key(
                "item",
                edited_path=editor,
                repair_dir=repair_dir,
            )
            self.assertEqual(caption, "repaired text")
            self.assertEqual(path, repaired)
            self.assertEqual(stage, "repair-v0.7")


if __name__ == "__main__":
    unittest.main()
