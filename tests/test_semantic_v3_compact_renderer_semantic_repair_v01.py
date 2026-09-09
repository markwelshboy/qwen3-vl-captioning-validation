from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_compact_renderer_semantic_repair_v01 import (
    _semantic_failure_lines,
    _semantic_warnings,
    build_semantic_repair_input,
)


class CompactRendererSemanticRepairV01Tests(unittest.TestCase):
    def test_length_warning_alone_never_becomes_semantic_repair_candidate(self) -> None:
        audit = {
            "warnings": ["rendered_caption_above_target_words"],
            "target_word_range": {"min": 60, "max": 90},
            "preferred_word_range": {"min": 70, "max": 82},
            "rendered_word_count": 96,
            "trigger": {"warnings": []},
            "semantic_policy_audit": {},
        }
        self.assertEqual(_semantic_warnings(audit), [])
        self.assertEqual(_semantic_failure_lines(audit), [])

    def test_unsupported_laterality_gets_local_neutralization_instruction(self) -> None:
        audit = {
            "warnings": ["unsupported_anatomical_laterality"],
            "trigger": {"warnings": []},
            "semantic_policy_audit": {
                "identity_leaks": [],
                "generic_identity_paraphrase_leaks": [],
                "hair_dye_detail_leaks": [],
                "age_proxy_identity_leaks": [],
                "unauthorized_anatomical_laterality": ["right hand"],
                "conflicting_pose_wording": [],
                "awkward_haircut_residue": [],
                "bad_local_attachment": [],
            },
        }
        lines = _semantic_failure_lines(audit)
        self.assertEqual(len(lines), 1)
        self.assertIn("right hand", lines[0])
        self.assertIn("Remove only the unsupported side word", lines[0])

    def test_identity_leak_gets_identity_only_instruction(self) -> None:
        audit = {
            "warnings": ["intrinsic_identity_leakage"],
            "trigger": {"warnings": []},
            "semantic_policy_audit": {
                "identity_leaks": ["falls loosely around their shoulders"],
                "generic_identity_paraphrase_leaks": [],
                "hair_dye_detail_leaks": [],
                "age_proxy_identity_leaks": [],
                "unauthorized_anatomical_laterality": [],
                "conflicting_pose_wording": [],
                "awkward_haircut_residue": [],
                "bad_local_attachment": [],
            },
        }
        lines = _semantic_failure_lines(audit)
        self.assertEqual(len(lines), 1)
        self.assertIn("falls loosely around their shoulders", lines[0])
        self.assertIn("Remove only that identity meaning", lines[0])

    def test_repair_prompt_uses_budgeted_caption_not_long_semantic_source(self) -> None:
        budget_record = {
            "profile": "compact",
            "trigger": "V3SUBJ",
            "grammar_profile": "neutral",
            "semantic_caption": (
                "V3SUBJ reclines on a bed. Their tousled hair falls loosely around their shoulders. "
                "A UNIQUE_LONG_SOURCE_ONLY_LAMP_DETAIL appears behind the bed."
            ),
            "budgeted_caption": (
                "V3SUBJ reclines on a bed. Their tousled hair falls loosely around their shoulders."
            ),
        }
        prompt_template = (
            "{{PROFILE}} {{MIN_WORDS}} {{MAX_WORDS}} {{TRIGGER}} "
            "{{SUBJECT_PRONOUN}} {{OBJECT_PRONOUN}} {{POSSESSIVE_PRONOUN}} "
            "{{REFLEXIVE_PRONOUN}}\n{{POSE_CORRECTIONS}}\n{{FAILURES}}\n{{CURRENT_CAPTION}}"
        )
        value = build_semantic_repair_input(
            budget_record=budget_record,
            pose={"caption_ready_phrases": []},
            prompt_template=prompt_template,
        )
        self.assertIn("falls loosely around their shoulders", value["repair_prompt"])
        self.assertNotIn("UNIQUE_LONG_SOURCE_ONLY_LAMP_DETAIL", value["repair_prompt"])

    def test_build_input_flags_right_hand_without_using_semantic_source_as_prompt(self) -> None:
        budget_record = {
            "profile": "compact",
            "trigger": "V3SUBJ",
            "grammar_profile": "neutral",
            "semantic_caption": (
                "V3SUBJ sits at a table. V3SUBJ holds a mug. "
                "A UNIQUE_SOURCE_ONLY_WINDOW_DETAIL appears behind them."
            ),
            "budgeted_caption": (
                "V3SUBJ sits at a table facing the camera. In their right hand, they hold a yellow mug."
            ),
        }
        prompt_template = (
            "{{PROFILE}} {{MIN_WORDS}} {{MAX_WORDS}} {{TRIGGER}} "
            "{{SUBJECT_PRONOUN}} {{OBJECT_PRONOUN}} {{POSSESSIVE_PRONOUN}} "
            "{{REFLEXIVE_PRONOUN}}\n{{POSE_CORRECTIONS}}\n{{FAILURES}}\n{{CURRENT_CAPTION}}"
        )
        value = build_semantic_repair_input(
            budget_record=budget_record,
            pose={"caption_ready_phrases": []},
            prompt_template=prompt_template,
        )
        self.assertIn("unsupported_anatomical_laterality", value["semantic_warnings"])
        self.assertIn("right hand", "\n".join(value["semantic_failure_lines"]))
        self.assertNotIn("UNIQUE_SOURCE_ONLY_WINDOW_DETAIL", value["repair_prompt"])


if __name__ == "__main__":
    unittest.main()
