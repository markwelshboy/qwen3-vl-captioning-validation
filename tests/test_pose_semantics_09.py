from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.pose_semantics import IDX
from qwen_caption_validate.pose_semantics_09 import build_pose_semantics


class PoseSemantics09Tests(unittest.TestCase):
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
            "neck": (0.50, 0.25),
            "right_shoulder": (0.40, 0.30),
            "left_shoulder": (0.60, 0.30),
        }

    def _analysis(
        self,
        summary: str,
        *,
        interactions: list[dict] | None = None,
        body_parts: list[dict] | None = None,
        entities: list[dict] | None = None,
    ) -> dict:
        return {
            "analysis": {
                "image_summary": summary,
                "framing": {"shot_scale": "medium", "subject_extent": "subject visible"},
                "target_subject": {
                    "visible_body_parts": body_parts or [],
                    "interactions": interactions or [],
                },
                "non_target_entities": entities or [],
            }
        }

    def _fusion(self) -> dict:
        return {"fusion": {"qualified_interactions": [], "qualified_body_parts": []}}

    def _gestalt(self, posture: str, confidence: float = 0.95, basis: str = "geometric") -> dict:
        return {
            "gestalt": {
                "schema_version": "pose-gestalt-1.0",
                "posture": posture,
                "posture_basis": basis,
                "posture_confidence": confidence,
                "body_configuration": posture,
                "support_configuration": [],
                "semantic_pose_summary": posture,
                "evidence": [],
                "counterevidence": [],
            }
        }

    def test_reclining_promotes_with_bedlike_support(self) -> None:
        interactions = [{
            "type": "support",
            "actor_part": "upper_torso",
            "actor_ownership": "target",
            "target": "bedding",
            "evidence_status": "observed",
            "confidence": 0.95,
            "notes": "torso resting on bedding",
        }]
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._fusion(),
            self._analysis("The subject is lying on a bed.", interactions=interactions),
            self._gestalt("reclining", 0.95, "contextual"),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "reclining")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "reclining_analyze_gestalt_plus_bedlike_support",
        )

    def test_reclining_without_support_remains_candidate(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._fusion(),
            self._analysis("A close portrait."),
            self._gestalt("reclining", 0.90, "contextual"),
        )
        self.assertIsNone(result["preferred_pose"]["posture"])
        self.assertEqual(result["posture_candidate"]["label"], "reclining")

    def test_standing_promotes_from_direct_weight_bearing_leg_record(self) -> None:
        points = self._portrait_points()
        points.update({
            "right_hip": (0.44, 0.50), "right_knee": (0.45, 0.70),
            "left_hip": (0.56, 0.50), "left_knee": (0.57, 0.70),
        })
        body_parts = [{
            "part": "upper_legs",
            "ownership": "target",
            "visibility": "partial",
            "geometry": "standing upright",
            "support": "standing on floor",
            "contact": None,
            "confidence": 0.90,
        }]
        result = build_pose_semantics(
            self._dwpose(points, "three_quarter_or_long"),
            self._fusion(),
            self._analysis("A mirror selfie in an elevator.", body_parts=body_parts),
            self._gestalt("standing", 0.95, "geometric"),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "standing")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "standing_gestalt_plus_direct_weight_bearing_support",
        )

    def test_weight_bearing_stance_vetoes_false_squat_gestalt(self) -> None:
        points = self._portrait_points()
        points.update({
            "right_hip": (0.44, 0.50), "right_knee": (0.45, 0.70), "right_ankle": (0.46, 0.90),
            "left_hip": (0.56, 0.50), "left_knee": (0.65, 0.58),
        })
        body_parts = [{
            "part": "right_leg",
            "anatomical_side": "right",
            "ownership": "target",
            "visibility": "full",
            "geometry": "extended, foot flat on floor",
            "support": "standing on foot",
            "contact": "foot on floor",
            "confidence": 0.95,
        }]
        result = build_pose_semantics(
            self._dwpose(points, "three_quarter_or_long"),
            self._fusion(),
            self._analysis("The subject bends forward with one knee lifted.", body_parts=body_parts),
            self._gestalt("squatting", 0.95, "geometric"),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "standing")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "standing_weight_bearing_vetoes_squat_gestalt",
        )
        self.assertEqual(result["posture_candidate"]["status"], "vetoed_candidate")

    def test_seated_promotes_with_analyze_gestalt_and_weak_bottom_up_geometry(self) -> None:
        points = self._portrait_points()
        points.update({
            "right_hip": (0.45, 0.55),
            "right_knee": (0.65, 0.56),
        })
        result = build_pose_semantics(
            self._dwpose(points, "three_quarter_or_long"),
            self._fusion(),
            self._analysis("The subject is seated between two trees."),
            self._gestalt("seated", 0.90, "contextual"),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "seated")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "seated_analyze_gestalt_plus_lower_body_geometry",
        )


if __name__ == "__main__":
    unittest.main()
