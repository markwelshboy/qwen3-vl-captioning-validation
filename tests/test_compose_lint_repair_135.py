from __future__ import annotations

import unittest

from qwen_caption_validate.compose_lint_repair_135 import _needs_repair, _render_repair_prompt


class ComposeLintRepair135Tests(unittest.TestCase):
    def test_violation_always_requires_repair(self) -> None:
        lint = {"violation_count": 1, "warning_count": 0}
        self.assertTrue(_needs_repair(lint, include_warnings=False))

    def test_warning_repair_is_configurable(self) -> None:
        lint = {"violation_count": 0, "warning_count": 1}
        self.assertTrue(_needs_repair(lint, include_warnings=True))
        self.assertFalse(_needs_repair(lint, include_warnings=False))

    def test_clean_caption_is_not_repaired(self) -> None:
        lint = {"violation_count": 0, "warning_count": 0}
        self.assertFalse(_needs_repair(lint, include_warnings=True))

    def test_repair_prompt_contains_original_evidence_and_findings(self) -> None:
        caption = "TOKEN says the right shoulder is visible."
        evidence = {
            "caption_policy": {"trigger_token": "TOKEN"},
            "pose_orientation": {
                "qualified_laterality": [{"side": "left", "body_family": "shoulder"}]
            },
        }
        lint = {
            "violation_count": 1,
            "violations": [
                {
                    "type": "unqualified_anatomical_laterality",
                    "text": "right shoulder",
                }
            ],
        }
        prompt = _render_repair_prompt(caption, evidence, lint)
        self.assertIn(caption, prompt)
        self.assertIn("unqualified_anatomical_laterality", prompt)
        self.assertIn('"side": "left"', prompt)
        self.assertIn("Do not add or reconstruct visual facts", prompt)
        self.assertIn("Never infer the complementary anatomical side", prompt)


if __name__ == "__main__":
    unittest.main()
