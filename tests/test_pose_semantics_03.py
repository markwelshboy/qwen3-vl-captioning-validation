from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate import pose_semantics as base
from qwen_caption_validate.pose_semantics_03 import build_pose_semantics


IDX = base.IDX


class PoseSemantics03Tests(unittest.TestCase):
    def _dwpose(self, points: dict[str, tuple[float, float]]) -> dict:
        person = np.full((18, 2), -1.0, dtype=float)
        for name, xy in points.items():
            person[IDX[name]] = xy

        def chain(names: list[str]) -> dict:
            visible = [
                name for name in names
                if name in points and 0 <= points[name][0] <= 1 and 0 <= points[name][1] <= 1
            ]
            return {
                "landmarks": names,
                "visible": visible,
                "visible_count": len(visible),
                "complete": len(visible) == len(names),
            }

        visible = [name for name, xy in points.items() if 0 <= xy[0] <= 1 and 0 <= xy[1] <= 1]
        return {
            "raw_pose": {"bodies": {"candidate": person.tolist()}},
            "derived": {
                "target_person_index": 0,
                "target": {
                    "visible_body_landmarks": visible,
                    "pose_extent_hint": "close_or_medium_close",
                    "connectivity": {
                        "right_arm": chain(["right_shoulder", "right_elbow", "right_wrist"]),
                        "left_arm": chain(["left_shoulder", "left_elbow", "left_wrist"]),
                        "right_leg": chain(["right_hip", "right_knee", "right_ankle"]),
                        "left_leg": chain(["left_hip", "left_knee", "left_ankle"]),
                    },
                },
            },
        }

    def _two_arms_down(self) -> dict[str, tuple[float, float]]:
        return {
            "nose": (0.50, 0.10),
            "neck": (0.50, 0.20),
            "right_shoulder": (0.40, 0.25),
            "right_elbow": (0.40, 0.45),
            "right_wrist": (0.40, 0.65),
            "left_shoulder": (0.60, 0.25),
            "left_elbow": (0.60, 0.45),
            "left_wrist": (0.60, 0.65),
        }

    def test_dwpose_only_arm_down_is_diagnostic_not_preferred(self) -> None:
        result = build_pose_semantics(self._dwpose(self._two_arms_down()), {"fusion": {}}, {"analysis": {}})
        arm_down = [item for item in result["gestures"] if (item.get("details") or {}).get("class") == "arm_down"]
        self.assertEqual(len(arm_down), 2)
        self.assertTrue(all(item["confidence_band"] == "weak" for item in arm_down))
        self.assertTrue(all(item["caption_preferred"] is False for item in arm_down))
        self.assertEqual(result["preferred_pose"]["gestures"], [])

    def test_explicit_bilateral_object_interaction_consumes_both_arm_fallbacks(self) -> None:
        fusion = {
            "fusion": {
                "qualified_interactions": [{
                    "type": "holding",
                    "actor_part": "both hands",
                    "target": "smartphone",
                    "confidence": 0.95,
                    "fusion_v2": {"selection_usable": True},
                }]
            }
        }
        result = build_pose_semantics(self._dwpose(self._two_arms_down()), fusion, {"analysis": {}})
        labels = [item["label"] for item in result["gestures"] if item["caption_preferred"]]
        self.assertIn("both hands holding smartphone", labels)
        self.assertNotIn("left arm hanging at the side", labels)
        self.assertNotIn("right arm hanging at the side", labels)
        classes = [(item.get("details") or {}).get("class") for item in result["gestures"]]
        self.assertNotIn("arm_down", classes)

    def test_governed_arm_down_agreement_can_still_promote(self) -> None:
        fusion = {
            "fusion": {
                "qualified_body_parts": [{
                    "part": "left arm",
                    "anatomical_side": "left",
                    "geometry": "arm hangs down at side",
                    "fusion_v2": {
                        "selection_usable": True,
                        "laterality_selection_usable": True,
                        "qualified_anatomical_side": "left",
                    },
                }]
            }
        }
        result = build_pose_semantics(self._dwpose(self._two_arms_down()), fusion, {"analysis": {}})
        preferred = [item for item in result["gestures"] if item["caption_preferred"]]
        labels = [item["label"] for item in preferred]
        self.assertIn("left arm hanging at the side", labels)
        left = next(item for item in preferred if item["label"] == "left arm hanging at the side")
        self.assertGreaterEqual(left["support_score"], 0.80)


if __name__ == "__main__":
    unittest.main()
