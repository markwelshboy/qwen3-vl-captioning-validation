from __future__ import annotations

import unittest

from qwen_caption_validate.pose_semantics_11 import _restore_cropped_seated_semantics


class PoseSemantics11Tests(unittest.TestCase):
    def _v10(self, label=None) -> dict:
        posture = {"status": "withheld", "label": None}
        if label:
            posture = {"status": "qualified", "label": label, "caption_preferred": True}
        return {
            "posture": posture,
            "preferred_pose": {"posture": label, "gestures": []},
            "gestures": [],
            "pose_gestalt_probe": {},
            "pose_gestalt_corroboration": {"valid": False, "route": "withheld"},
        }

    def _v07(self) -> dict:
        return {
            "posture": {
                "status": "qualified",
                "label": "seated",
                "caption_preferred": True,
                "support_score": 0.85,
                "limitations": [],
            },
            "preferred_pose": {
                "posture": "seated",
                "gestures": [
                    "leaning on the left arm at a table, with the chin resting on the hand"
                ],
            },
            "pose_gestalt_probe": {
                "posture": "seated",
                "contradicted_by_existing_geometric_posture": False,
            },
            "pose_gestalt_corroboration": {"valid": True},
            "support_graph": {
                "support_chains": [
                    {"side": "left", "support_part": "forearm", "surface": "table"}
                ]
            },
            "gestures": [
                {
                    "id": "gesture_supported_lean_left_table",
                    "label": "leaning on the left arm at a table, with the chin resting on the hand",
                    "caption_preferred": True,
                    "support_score": 0.8,
                    "support": ["pose_gestalt_contextual_support"],
                    "details": {
                        "class": "supported_lean",
                        "actor_side": "left",
                        "surface": "table",
                    },
                }
            ],
        }

    def _gestalt(self) -> dict:
        return {
            "gestalt": {
                "posture": "seated",
                "posture_basis": "contextual",
                "posture_confidence": 0.85,
            }
        }

    def _dwpose(self, with_forearm=True) -> dict:
        visible = ["left_elbow"]
        if with_forearm:
            visible.append("left_wrist")
        return {"derived": {"target": {"visible_body_landmarks": visible}}}

    def test_recovers_cropped_seated_and_support_when_segment_observed(self) -> None:
        result = self._v10()
        _restore_cropped_seated_semantics(result, self._v07(), self._dwpose(True), self._gestalt())
        self.assertEqual(result["posture"]["label"], "seated")
        self.assertEqual(result["posture"]["authority"], "v11_dedicated_image_pose_gestalt_plus_non_circular_support_validation")
        labels = result["preferred_pose"]["gestures"]
        self.assertIn("leaning on the left arm at a table, with the chin resting on the hand", labels)
        audit = result["top_down_cropped_seated_semantics_v11"]
        self.assertTrue(audit["eligible"])
        self.assertTrue(audit["promoted"])

    def test_support_detail_remains_withheld_without_observed_segment(self) -> None:
        result = self._v10()
        _restore_cropped_seated_semantics(result, self._v07(), self._dwpose(False), self._gestalt())
        self.assertEqual(result["posture"]["label"], "seated")
        self.assertEqual(result["preferred_pose"]["gestures"], [])
        self.assertEqual(result["top_down_cropped_seated_semantics_v11"]["restored_support_gestures"], [])

    def test_does_not_override_conflicting_qualified_v10_posture(self) -> None:
        result = self._v10("standing")
        _restore_cropped_seated_semantics(result, self._v07(), self._dwpose(True), self._gestalt())
        self.assertEqual(result["posture"]["label"], "standing")
        self.assertFalse(result["top_down_cropped_seated_semantics_v11"]["eligible"])


if __name__ == "__main__":
    unittest.main()
