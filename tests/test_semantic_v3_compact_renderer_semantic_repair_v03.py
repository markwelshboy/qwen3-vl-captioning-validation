from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_compact_renderer_semantic_repair_v03 import (
    deterministic_local_repair,
)


class CompactRendererSemanticRepairV03Tests(unittest.TestCase):
    def test_compact_hair_length_repair_produces_finite_verb(self) -> None:
        caption = (
            "V3SUBJ reclines on a bed. Their tousled hair falls loosely around their shoulders, "
            "partially framing their face. They gaze calmly at the camera."
        )
        audit = {
            "semantic_policy_audit": {
                "identity_leaks": ["falls loosely around their shoulders"],
                "unauthorized_anatomical_laterality": [],
            }
        }
        result = deterministic_local_repair(caption=caption, audit=audit)
        self.assertIn("Their tousled hair partially frames their face.", result["caption"])
        self.assertNotIn("around their shoulders", result["caption"])

    def test_medium_hair_length_repair_keeps_following_while_clause(self) -> None:
        caption = (
            "Their tousled hair falls loosely around their shoulders, partially framing their face, "
            "while their head rests on a white pillow."
        )
        audit = {
            "semantic_policy_audit": {
                "identity_leaks": ["falls loosely around their shoulders"],
                "unauthorized_anatomical_laterality": [],
            }
        }
        result = deterministic_local_repair(caption=caption, audit=audit)
        self.assertEqual(
            result["caption"],
            "Their tousled hair partially frames their face, while their head rests on a white pillow.",
        )

    def test_laterality_rule_from_v02_is_retained(self) -> None:
        caption = (
            "V3SUBJ sits with their head resting on their left fist while their right hand holds a mug."
        )
        audit = {
            "semantic_policy_audit": {
                "identity_leaks": [],
                "unauthorized_anatomical_laterality": ["right hand"],
            }
        }
        result = deterministic_local_repair(caption=caption, audit=audit)
        self.assertIn("left fist", result["caption"])
        self.assertIn("their hand holds a mug", result["caption"])
        self.assertNotIn("right hand", result["caption"])


if __name__ == "__main__":
    unittest.main()
