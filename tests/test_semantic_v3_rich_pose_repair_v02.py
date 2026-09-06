from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_repair_v02 import (
    _local_attachment_leaks,
    quality_audit,
)


class RichPoseRepairV02Tests(unittest.TestCase):
    def test_hair_featuring_beard_is_bad_local_attachment(self) -> None:
        text = (
            "A medium close-up captures a man with hair styled with volume, "
            "featuring a short, well-groomed beard and mustache."
        )
        leaks = _local_attachment_leaks(text)
        self.assertTrue(leaks)

    def test_person_with_hair_then_separate_beard_sentence_is_clean(self) -> None:
        text = (
            "A medium close-up captures a man with hair styled with volume. "
            "He has a short, well-groomed beard and mustache."
        )
        self.assertEqual(_local_attachment_leaks(text), [])

    def test_attachment_failure_blocks_final_gate(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A medium close-up captures a man with shoulder-length wavy hair. "
            "He has a short, well-groomed beard and mustache. He wears a black shirt."
        )
        edited = (
            "A medium close-up captures a man with hair styled with volume, "
            "featuring a short, well-groomed beard and mustache. He wears a black shirt."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("bad_local_attachment", audit["warnings"])
        self.assertFalse(audit["passes_basic_gate"])

    def test_clean_repair_still_passes(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A medium close-up captures a man with shoulder-length wavy hair. "
            "He has a short, well-groomed beard and mustache. He wears a black shirt."
        )
        edited = (
            "A medium close-up captures a man with hair styled with volume. "
            "He has a short, well-groomed beard and mustache. He wears a black shirt."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertNotIn("bad_local_attachment", audit["warnings"])


if __name__ == "__main__":
    unittest.main()
