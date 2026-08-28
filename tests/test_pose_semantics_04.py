from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.pose_semantics import IDX
from qwen_caption_validate.pose_semantics_04 import build_pose_semantics


class PoseSemantics04Tests(unittest.TestCase):
    def _dwpose(self, points: dict[str, tuple[float, float]]) -> dict:
        person = np.full((18, 2), -1.0, dtype=float)
        for name, xy in points.items():
            person[IDX[name]] = xy

        def chain(names: list[str]) -> dict:
            visible = [name for name in names if name in points]
            return {
                "landmarks": names,
                "visible": visible,
                "visible_count": len(visible),
                "complete": len(visible) == len(names),
            }

        return {
            "raw_pose": {"bodies": {"candidate": person.tolist()}},
            "derived": {
                "target_person_index": 0,
                "target": {
                    "visible_body_landmarks": list(points),
                    "pose_extent_hint": "full_length",
                    "connectivity": {
                        "right_arm": chain(["right_shoulder", "right_elbow", "right_wrist"]),
                        "left_arm": chain(["left_shoulder", "left_elbow", "left_wrist"]),
                        "right_leg": chain(["right_hip", "right_knee", "right_ankle"]),
                        "left_leg": chain(["left_hip", "left_knee", "left_ankle"]),
                    },
                },
            },
        }

    def _asymmetric_standing_points(self) -> dict[str, tuple[float, float]]:
        # Right leg is essentially straight. Left knee has mild flexion while the
        # thigh still descends vertically; this remains an ordinary standing pose.
        return {
            "nose": (0.50, 0.08),
            "neck": (0.50, 0.18),
            "right_shoulder": (0.42, 0.22),
            "left_shoulder": (0.58, 0.22),
            "right_elbow": (0.44, 0.38),
            "left_elbow": (0.56, 0.38),
            "right_wrist": (0.46, 0.50),
            "left_wrist": (0.54, 0.50),
            "right_hip": (0.46, 0.48),
            "left_hip": (0.54, 0.48),
            "right_knee": (0.47, 0.70),
            "right_ankle": (0.48, 0.91),
            "left_knee": (0.55, 0.70),
            "left_ankle": (0.61, 0.90),
        }

    def test_asymmetric_mild_knee_flexion_promotes_standing(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._asymmetric_standing_points()),
            {"fusion": {}},
            {"analysis": {"image_summary": "The subject is standing indoors while holding an item."}},
        )
        self.assertEqual(result["posture"]["status"], "qualified")
        self.assertEqual(result["posture"]["label"], "standing")
        self.assertEqual(result["posture"]["confidence_band"], "strong")
        self.assertEqual(result["posture"]["authority"], "top_down_weight_bearing_stance_plus_analyze_semantics")
        self.assertEqual(result["preferred_pose"]["posture"], "standing")
        self.assertTrue(result["human_summary"].startswith("Standing;"))

    def test_same_geometry_without_semantic_standing_does_not_promote(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._asymmetric_standing_points()),
            {"fusion": {}},
            {"analysis": {"image_summary": "The subject is indoors while holding an item."}},
        )
        self.assertNotEqual(result["posture"].get("authority"), "top_down_weight_bearing_stance_plus_analyze_semantics")

    def test_deeply_flexed_knee_does_not_use_asymmetric_standing_rule(self) -> None:
        points = self._asymmetric_standing_points()
        # Pull the left ankle far back/up to create a much smaller knee angle.
        points["left_ankle"] = (0.40, 0.74)
        result = build_pose_semantics(
            self._dwpose(points),
            {"fusion": {}},
            {"analysis": {"image_summary": "The subject is standing."}},
        )
        self.assertNotEqual(result["posture"].get("authority"), "top_down_weight_bearing_stance_plus_analyze_semantics")


if __name__ == "__main__":
    unittest.main()
