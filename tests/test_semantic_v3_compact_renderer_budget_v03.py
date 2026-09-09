from __future__ import annotations

import unittest
from unittest.mock import patch

from qwen_caption_validate import semantic_v3_compact_renderer_budget_v03 as v03


class CompactRendererBudgetV03Tests(unittest.TestCase):
    def _base_result(self, caption: str) -> dict:
        return {
            "budgeted_caption": caption,
            "status": "pass",
            "semantic_repair_required": False,
            "remaining_hard_warnings": [],
        }

    def test_finish_length_requires_renderer_rerun_even_if_budget_caption_is_complete(self) -> None:
        renderer = {"performance": {"finish_reason": "length"}}
        with patch.object(v03.budget_v02, "budget_renderer_record", return_value=self._base_result("V3SUBJ smiles at the camera.")):
            result = v03.budget_renderer_record(renderer_record=renderer, pose={})
        self.assertEqual(result["status"], "renderer_rerender_required")
        self.assertTrue(result["renderer_rerender_required"])
        self.assertIn("source_renderer_truncated", result["renderer_rerender_warnings"])
        self.assertFalse(result["semantic_repair_required"])

    def test_incomplete_budgeted_sentence_requires_renderer_rerun(self) -> None:
        renderer = {"performance": {"finish_reason": "stop"}}
        with patch.object(v03.budget_v02, "budget_renderer_record", return_value=self._base_result("V3SUBJ wears a patterned top with green and brown on a")):
            result = v03.budget_renderer_record(renderer_record=renderer, pose={})
        self.assertEqual(result["status"], "renderer_rerender_required")
        self.assertIn("budgeted_caption_incomplete_sentence", result["renderer_rerender_warnings"])

    def test_complete_stop_source_preserves_v02_pass(self) -> None:
        renderer = {"performance": {"finish_reason": "stop"}}
        with patch.object(v03.budget_v02, "budget_renderer_record", return_value=self._base_result("V3SUBJ smiles at the camera.")):
            result = v03.budget_renderer_record(renderer_record=renderer, pose={})
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["renderer_rerender_required"])
        self.assertEqual(result["renderer_rerender_warnings"], [])

    def test_terminal_quotes_and_parentheses_are_allowed(self) -> None:
        self.assertTrue(v03._caption_ends_complete_sentence('V3SUBJ wears a shirt (dark).'))
        self.assertTrue(v03._caption_ends_complete_sentence('V3SUBJ smiles."'))


if __name__ == "__main__":
    unittest.main()
