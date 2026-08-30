from __future__ import annotations

import unittest

from qwen_caption_validate.pose_semantics_12 import _apply_support_target_firewall


class PoseSemantics12Tests(unittest.TestCase):
    def test_surface_support_restored_by_v11_is_withheld(self) -> None:
        result = {
            "preferred_pose": {"gestures": ["left forearm resting on the table"]},
            "gestures": [
                {
                    "label": "left forearm resting on the table",
                    "caption_preferred": True,
                    "support_score": 0.9,
                    "details": {
                        "class": "surface_support",
                        "actor_side": "left",
                        "part": "forearm",
                        "surface": "table",
                        "semantic_authority": "image_conditioned_pose_gestalt_with_observed_supporting_segment",
                    },
                    "support": [
                        "pose_gestalt_observed_support",
                        "v11_dwpose_supporting_segment_observation_gate",
                    ],
                }
            ],
        }
        _apply_support_target_firewall(result)
        self.assertEqual(result["preferred_pose"]["gestures"], [])
        self.assertFalse(result["gestures"][0]["caption_preferred"])
        self.assertEqual(
            result["support_target_firewall_v12"]["withheld_surface_support"],
            ["left forearm resting on the table"],
        )

    def test_supported_lean_becomes_target_neutral(self) -> None:
        result = {
            "preferred_pose": {
                "gestures": [
                    "leaning on the left arm at a table or desk, with the chin resting on the hand"
                ]
            },
            "gestures": [
                {
                    "label": "leaning on the left arm at a table or desk, with the chin resting on the hand",
                    "caption_preferred": True,
                    "support_score": 0.8,
                    "confidence_band": "strong",
                    "details": {
                        "class": "supported_lean",
                        "actor_side": "left",
                        "surface": "surface",
                        "semantic_authority": "image_conditioned_pose_gestalt_with_observed_supporting_segment",
                    },
                    "support": [
                        "head/chin is supported by the left hand",
                        "left forearm is supported by surface (pose_gestalt_contextual_support)",
                        "v11_dwpose_supporting_segment_observation_gate",
                    ],
                }
            ],
        }
        _apply_support_target_firewall(result)
        self.assertEqual(
            result["preferred_pose"]["gestures"],
            ["chin resting on the left hand, with the left forearm held beneath it"],
        )
        gesture = next(item for item in result["gestures"] if item.get("caption_preferred"))
        self.assertEqual((gesture.get("details") or {}).get("surface_target_status"), "withheld_unverified")
        self.assertNotIn("table", gesture["label"])
        self.assertNotIn("desk", gesture["label"])

    def test_non_v11_gesture_is_unchanged(self) -> None:
        result = {
            "preferred_pose": {"gestures": ["right hand resting on the hip"]},
            "gestures": [
                {
                    "label": "right hand resting on the hip",
                    "caption_preferred": True,
                    "details": {"class": "hand_on_hip", "actor_side": "right"},
                }
            ],
        }
        _apply_support_target_firewall(result)
        self.assertEqual(result["preferred_pose"]["gestures"], ["right hand resting on the hip"])


if __name__ == "__main__":
    unittest.main()
