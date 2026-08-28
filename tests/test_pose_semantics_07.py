from __future__ import annotations

import unittest

from qwen_caption_validate.pose_semantics_07 import (
    _corroboration_audit,
    _enrich_supported_lean_labels,
    _semantic_economy_v07,
)


class PoseSemantics07Tests(unittest.TestCase):
    def test_contextual_probe_seat_completion_cannot_self_confirm(self) -> None:
        result = {
            "support_graph": {
                "body_support_edges": [
                    {
                        "actor_side": None,
                        "actor_part": "torso",
                        "target_class": "seat",
                        "target": "chair",
                        "relation": "support",
                        "confidence": 0.80,
                        "authority": "pose_gestalt_contextual_support",
                        "source": "pose_gestalt_v1.support_configuration",
                    }
                ],
                "head_hand_edges": [],
                "support_chains": [],
            }
        }
        audit = _corroboration_audit(result)
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["bottom_up_support_count"], 0)
        self.assertEqual(audit["probe_observed_support_count"], 0)
        self.assertEqual(audit["cross_source_support_chain_count"], 0)

    def test_probe_observed_surface_support_can_corroborate(self) -> None:
        result = {
            "support_graph": {
                "body_support_edges": [
                    {
                        "actor_side": "left",
                        "actor_part": "forearm",
                        "target_class": "surface",
                        "target": "table",
                        "relation": "support",
                        "confidence": 0.95,
                        "authority": "pose_gestalt_observed_support",
                        "source": "pose_gestalt_v1.support_configuration",
                    }
                ],
                "head_hand_edges": [],
                "support_chains": [],
            }
        }
        audit = _corroboration_audit(result)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["probe_observed_support_count"], 1)

    def test_contextual_forearm_support_can_corroborate_via_independent_head_chain(self) -> None:
        result = {
            "support_graph": {
                "body_support_edges": [
                    {
                        "actor_side": "left",
                        "actor_part": "forearm",
                        "target_class": "surface",
                        "target": "surface",
                        "relation": "support",
                        "confidence": 0.80,
                        "authority": "pose_gestalt_contextual_support",
                        "source": "pose_gestalt_v1.support_configuration",
                    }
                ],
                "head_hand_edges": [
                    {
                        "actor_side": "left",
                        "relation": "head_supported_by_hand",
                        "confidence": 0.89,
                        "authority": "pose_semantics_head_support",
                    }
                ],
                "support_chains": [
                    {
                        "side": "left",
                        "support_part": "forearm",
                        "surface": "surface",
                    }
                ],
            }
        }
        audit = _corroboration_audit(result)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["cross_source_support_chain_count"], 1)

    def test_seated_support_ranking_prefers_observed_forearm_on_named_table(self) -> None:
        result = {
            "posture": {"status": "qualified", "label": "seated"},
            "preferred_pose": {},
            "gestures": [
                {
                    "label": "left hand resting on the surface",
                    "support_score": 0.90,
                    "caption_preferred": True,
                    "support": ["governed_fusion_body_surface_relation"],
                    "limitations": [],
                    "details": {"class": "surface_support", "actor_side": "left", "part": "hand", "surface": "surface"},
                },
                {
                    "label": "left forearm resting on the table",
                    "support_score": 0.90,
                    "caption_preferred": False,
                    "support": ["pose_gestalt_observed_support"],
                    "limitations": ["redundant same-surface support evidence subsumed by seated posture"],
                    "details": {"class": "surface_support", "actor_side": "left", "part": "forearm", "surface": "table"},
                },
                {
                    "label": "right hand resting on the surface",
                    "support_score": 0.90,
                    "caption_preferred": False,
                    "support": ["analyze_observed_body_surface_relation"],
                    "limitations": ["redundant same-surface support evidence subsumed by seated posture"],
                    "details": {"class": "surface_support", "actor_side": "right", "part": "hand", "surface": "surface"},
                },
            ],
        }
        _semantic_economy_v07(result)
        preferred = [item["label"] for item in result["gestures"] if item.get("caption_preferred")]
        self.assertEqual(preferred, ["left forearm resting on the table"])
        self.assertEqual(result["preferred_pose"]["gestures"], preferred)

    def test_supported_lean_preserves_table_or_desk_specificity(self) -> None:
        result = {
            "support_graph": {
                "body_support_edges": [
                    {
                        "actor_side": "left",
                        "actor_part": "forearm",
                        "target_class": "surface",
                        "target": "surface",
                        "relation": "support",
                        "confidence": 0.80,
                        "source": "pose_gestalt_v1.support_configuration",
                        "source_text": {"target": "surface (likely table or desk)"},
                    }
                ],
                "support_chains": [
                    {
                        "side": "left",
                        "support_part": "forearm",
                        "surface": "surface",
                    }
                ],
            },
            "gestures": [
                {
                    "label": "leaning on the left arm at a surface, with the chin resting on the hand",
                    "caption_preferred": True,
                    "details": {"class": "supported_lean", "actor_side": "left", "surface": "surface"},
                }
            ],
        }
        _enrich_supported_lean_labels(result)
        self.assertEqual(
            result["gestures"][0]["label"],
            "leaning on the left arm at a table or desk, with the chin resting on the hand",
        )
        self.assertEqual(result["gestures"][0]["details"]["surface_phrase"], "table or desk")


if __name__ == "__main__":
    unittest.main()
