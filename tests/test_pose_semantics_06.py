from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.pose_semantics import IDX
from qwen_caption_validate.pose_semantics_06 import build_pose_semantics


class PoseSemantics06Tests(unittest.TestCase):
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

    def _analysis(self) -> dict:
        return {
            "analysis": {
                "image_summary": "A close portrait of the subject.",
                "framing": {
                    "shot_scale": "medium_close_up",
                    "subject_extent": "head and upper torso",
                },
                "target_subject": {
                    "visible_body_parts": [],
                    "interactions": [],
                },
                "non_target_entities": [],
            }
        }

    def _head_support_fusion(self) -> dict:
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
                    }
                ]
            }
        }

    def _gestalt(self, *, posture: str = "seated", confidence: float = 0.92, supports: list[dict] | None = None) -> dict:
        return {
            "gestalt": {
                "schema_version": "pose-gestalt-1.0",
                "posture": posture,
                "posture_basis": "contextual",
                "posture_confidence": confidence,
                "body_configuration": "seated at a table",
                "support_configuration": supports or [],
                "semantic_pose_summary": "seated at a table, leaning on the left elbow with chin on hand",
                "evidence": ["supported portrait configuration"],
                "counterevidence": [],
            }
        }

    def test_seated_probe_plus_support_graph_promotes(self) -> None:
        fusion = {
            "fusion": {
                "qualified_interactions": [
                    {
                        "type": "support",
                        "actor_part": "right forearm",
                        "actor_anatomical_side": "right",
                        "target": "table",
                        "confidence": 0.95,
                        "fusion_v2": {
                            "selection_usable": True,
                            "qualified_actor_anatomical_side": "right",
                        },
                    }
                ]
            }
        }
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            fusion,
            self._analysis(),
            self._gestalt(),
        )
        self.assertEqual(result["posture"]["status"], "qualified")
        self.assertEqual(result["posture"]["label"], "seated")
        self.assertEqual(result["posture"]["authority"], "top_down_pose_gestalt_plus_support_graph")
        self.assertTrue(result["pose_gestalt_probe"]["caption_preferred"])
        self.assertTrue(result["human_summary"].startswith("Seated;"))

    def test_seated_probe_without_support_does_not_promote(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            {"fusion": {}},
            self._analysis(),
            self._gestalt(),
        )
        self.assertNotEqual(result["preferred_pose"].get("posture"), "seated")
        self.assertFalse(result["pose_gestalt_probe"]["caption_preferred"])

    def test_gestalt_forearm_table_plus_head_hand_builds_supported_lean(self) -> None:
        supports = [
            {
                "body_part": "left forearm",
                "relation": "resting_on",
                "target": "table",
                "evidence_status": "contextual",
                "confidence": 0.90,
            }
        ]
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._head_support_fusion(),
            self._analysis(),
            self._gestalt(supports=supports),
        )
        chains = result["support_graph"]["support_chains"]
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0]["side"], "left")
        labels = [item["label"] for item in result["gestures"] if item.get("caption_preferred")]
        self.assertIn("leaning on the left arm at a table, with the chin resting on the hand", labels)
        self.assertNotIn("chin/head resting on the left hand", labels)
        self.assertEqual(result["preferred_pose"]["posture"], "seated")

    def test_existing_geometric_standing_cannot_be_overwritten_by_probe(self) -> None:
        points = self._portrait_points()
        points.update({
            "right_hip": (0.45, 0.48),
            "right_knee": (0.46, 0.70),
            "right_ankle": (0.47, 0.92),
            "left_hip": (0.55, 0.48),
            "left_knee": (0.56, 0.70),
            "left_ankle": (0.57, 0.92),
        })
        analysis = self._analysis()
        analysis["analysis"]["image_summary"] = "The subject is standing."
        result = build_pose_semantics(
            self._dwpose(points, extent="full_length"),
            {"fusion": {}},
            analysis,
            self._gestalt(
                supports=[
                    {
                        "body_part": "left hand",
                        "relation": "resting_on",
                        "target": "table",
                        "evidence_status": "observed",
                        "confidence": 0.95,
                    }
                ]
            ),
        )
        self.assertEqual(result["posture"]["label"], "standing")
        self.assertTrue(result["pose_gestalt_probe"]["contradicted_by_existing_geometric_posture"])
        self.assertFalse(result["pose_gestalt_probe"]["caption_preferred"])


if __name__ == "__main__":
    unittest.main()
