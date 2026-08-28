from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate import pose_semantics as base
from qwen_caption_validate.pose_semantics_02 import build_pose_semantics

IDX = base.IDX


class PoseSemantics02Tests(unittest.TestCase):
    def _dwpose(self, points: dict[str, tuple[float, float]], extent: str = "close_or_medium_close") -> dict:
        person = np.full((18, 2), -1.0, dtype=float)
        for name, xy in points.items():
            person[IDX[name]] = xy

        def chain(names: list[str]) -> dict:
            visible = [name for name in names if name in points and 0 <= points[name][0] <= 1 and 0 <= points[name][1] <= 1]
            return {"landmarks": names, "visible": visible, "visible_count": len(visible), "complete": len(visible) == len(names)}

        return {
            "raw_pose": {"bodies": {"candidate": person.tolist()}},
            "derived": {
                "target_person_index": 0,
                "target": {
                    "pose_extent_hint": extent,
                    "connectivity": {
                        "right_arm": chain(["right_shoulder", "right_elbow", "right_wrist"]),
                        "left_arm": chain(["left_shoulder", "left_elbow", "left_wrist"]),
                        "right_leg": chain(["right_hip", "right_knee", "right_ankle"]),
                        "left_leg": chain(["left_hip", "left_knee", "left_ankle"]),
                    },
                },
            },
        }

    def test_out_of_frame_wrist_is_not_observed_for_semantics(self) -> None:
        points = {
            "nose": (0.5, 0.1), "neck": (0.5, 0.2),
            "left_shoulder": (0.6, 0.25), "left_elbow": (0.62, 0.6), "left_wrist": (0.62, 1.20),
            "right_shoulder": (0.4, 0.25), "right_elbow": (0.38, 0.6), "right_wrist": (0.38, 1.15),
        }
        result = build_pose_semantics(self._dwpose(points), {"fusion": {}}, {"analysis": {}})
        self.assertNotIn("left_wrist", result["geometry_features"]["visible_joints"])
        self.assertNotIn("right_wrist", result["geometry_features"]["visible_joints"])
        labels = [item["label"] for item in result["gestures"]]
        self.assertNotIn("left arm hanging at the side", labels)
        self.assertNotIn("right arm hanging at the side", labels)

    def test_medium_close_and_dwpose_close_veto_mid_thigh_extent(self) -> None:
        points = {
            "nose": (0.5, 0.1), "neck": (0.5, 0.2),
            "left_shoulder": (0.6, 0.25), "right_shoulder": (0.4, 0.25),
        }
        analysis = {
            "analysis": {
                "framing": {
                    "shot_scale": "medium_close_up",
                    "subject_extent": "Upper body from mid-thighs to top of head",
                }
            }
        }
        result = build_pose_semantics(self._dwpose(points), {"fusion": {}}, analysis)
        self.assertEqual(result["framing"]["label"], "medium close-up")
        self.assertEqual(result["framing"]["arbitration"]["status"], "resolved_conflict")
        self.assertEqual(result["preferred_pose"]["framing"], "medium close-up")

    def test_explicit_both_hands_actor_is_preserved(self) -> None:
        points = {
            "nose": (0.5, 0.1), "neck": (0.5, 0.2),
            "left_shoulder": (0.6, 0.25), "left_elbow": (0.58, 0.45), "left_wrist": (0.53, 0.55),
            "right_shoulder": (0.4, 0.25), "right_elbow": (0.42, 0.45), "right_wrist": (0.47, 0.55),
        }
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
        result = build_pose_semantics(self._dwpose(points), fusion, {"analysis": {}})
        labels = [item["label"] for item in result["gestures"] if item["caption_preferred"]]
        self.assertIn("both hands holding smartphone", labels)
        self.assertNotIn("hand holding smartphone", labels)

    def test_hand_on_hip_support_neutralizes_raw_target_side(self) -> None:
        points = {
            "nose": (0.5, 0.1), "neck": (0.5, 0.2),
            "right_shoulder": (0.4, 0.25), "right_elbow": (0.3, 0.42), "right_wrist": (0.45, 0.52),
            "left_shoulder": (0.6, 0.25),
            "right_hip": (0.45, 0.52), "left_hip": (0.55, 0.52),
        }
        fusion = {
            "fusion": {
                "qualified_interactions": [{
                    "type": "contact",
                    "actor_part": "right hand",
                    "actor_anatomical_side": "right",
                    "target": "left hip",
                    "confidence": 0.95,
                    "fusion_v2": {"selection_usable": True, "qualified_actor_anatomical_side": "right"},
                }]
            }
        }
        result = build_pose_semantics(self._dwpose(points, "waist_or_upper_body"), fusion, {"analysis": {}})
        gesture = next(item for item in result["gestures"] if item["details"].get("class") == "hand_on_hip")
        self.assertEqual(gesture["label"], "right hand resting on the hip")
        self.assertTrue(all("left hip" not in support.lower() for support in gesture["support"]))


if __name__ == "__main__":
    unittest.main()
