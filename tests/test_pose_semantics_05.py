from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.pose_semantics import IDX
from qwen_caption_validate.pose_semantics_05 import build_pose_semantics


class PoseSemantics05Tests(unittest.TestCase):
    def _dwpose(self, points: dict[str, tuple[float, float]], extent: str = "close_or_medium_close") -> dict:
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

    def _portrait_points(self) -> dict[str, tuple[float, float]]:
        return {
            "nose": (0.50, 0.12),
            "neck": (0.50, 0.26),
            "right_shoulder": (0.38, 0.30),
            "right_elbow": (0.36, 0.52),
            "right_wrist": (0.38, 0.72),
            "left_shoulder": (0.62, 0.30),
            "left_elbow": (0.67, 0.55),
            "left_wrist": (0.57, 0.38),
        }

    def _support_fusion(self) -> dict:
        return {
            "fusion": {
                "qualified_interactions": [
                    {
                        "type": "support",
                        "actor_part": "left hand",
                        "actor_anatomical_side": "left",
                        "target": "chin",
                        "confidence": 0.95,
                        "fusion_v2": {
                            "selection_usable": True,
                            "qualified_actor_anatomical_side": "left",
                        },
                    },
                    {
                        "type": "support",
                        "actor_part": "left forearm",
                        "actor_anatomical_side": "left",
                        "target": "table",
                        "confidence": 0.92,
                        "fusion_v2": {
                            "selection_usable": True,
                            "qualified_actor_anatomical_side": "left",
                        },
                    },
                ]
            }
        }

    def _analysis(self, summary: str) -> dict:
        return {
            "analysis": {
                "image_summary": summary,
                "framing": {
                    "shot_scale": "medium_close_up",
                    "subject_extent": "head and upper torso",
                },
                "target_subject": {
                    "visible_body_parts": [],
                    "interactions": [],
                },
                "non_target_entities": [
                    {"description": "table in front of the subject"}
                ],
            }
        }

    def test_head_hand_arm_table_chain_compresses_to_supported_lean(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._support_fusion(),
            self._analysis("A close portrait of the subject."),
        )
        chains = result["support_graph"]["support_chains"]
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0]["side"], "left")
        labels = [item["label"] for item in result["gestures"] if item["caption_preferred"]]
        self.assertIn("leaning on the left arm at a table, with the chin resting on the hand", labels)
        self.assertNotIn("chin/head resting on the left hand", labels)

    def test_explicit_seated_plus_support_chain_promotes_contextual_seated(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._support_fusion(),
            self._analysis("The subject is seated at a table in a close portrait."),
        )
        self.assertEqual(result["posture"]["status"], "qualified")
        self.assertEqual(result["posture"]["label"], "seated")
        self.assertEqual(result["posture"]["authority"], "contextual_support_configuration")
        self.assertTrue(result["contextual_posture"]["caption_preferred"])
        self.assertIn("Seated;", result["human_summary"])
        self.assertIn("leaning on the left arm at a table", result["human_summary"])

    def test_table_support_chain_without_seated_semantics_does_not_invent_seated(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._support_fusion(),
            self._analysis("The subject leans on a table in a close portrait."),
        )
        self.assertNotEqual(result["preferred_pose"].get("posture"), "seated")
        self.assertFalse(result["contextual_posture"]["caption_preferred"])
        self.assertIn("leaning on the left arm at a table", result["human_summary"])

    def test_direct_body_to_seat_support_can_promote_without_visible_legs(self) -> None:
        fusion = {
            "fusion": {
                "qualified_interactions": [
                    {
                        "type": "support",
                        "actor_part": "torso",
                        "target": "chair seat",
                        "confidence": 0.95,
                        "fusion_v2": {"selection_usable": True},
                    }
                ]
            }
        }
        analysis = self._analysis("A close portrait of the subject.")
        analysis["analysis"]["non_target_entities"] = [{"description": "chair"}]
        result = build_pose_semantics(self._dwpose(self._portrait_points()), fusion, analysis)
        self.assertEqual(result["posture"]["label"], "seated")
        self.assertEqual(result["posture"]["authority"], "contextual_support_configuration")

    def test_existing_geometric_standing_is_not_overwritten(self) -> None:
        points = self._portrait_points()
        points.update({
            "right_hip": (0.45, 0.48),
            "right_knee": (0.46, 0.70),
            "right_ankle": (0.47, 0.92),
            "left_hip": (0.55, 0.48),
            "left_knee": (0.56, 0.70),
            "left_ankle": (0.57, 0.92),
        })
        analysis = self._analysis("The subject is standing while leaning on a table.")
        result = build_pose_semantics(self._dwpose(points, extent="full_length"), self._support_fusion(), analysis)
        self.assertEqual(result["posture"]["label"], "standing")
        self.assertTrue(result["contextual_posture"]["contradicted_by_existing_geometric_posture"] or not result["contextual_posture"]["caption_preferred"])


if __name__ == "__main__":
    unittest.main()
