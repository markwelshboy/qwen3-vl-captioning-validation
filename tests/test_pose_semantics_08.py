from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.pose_semantics import IDX
from qwen_caption_validate.pose_semantics_08 import build_pose_semantics


class PoseSemantics08Tests(unittest.TestCase):
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
        summary: str = "A close portrait of the subject.",
        *,
        interactions: list[dict] | None = None,
        body_parts: list[dict] | None = None,
        entities: list[dict] | None = None,
    ) -> dict:
        return {
            "analysis": {
                "image_summary": summary,
                "framing": {
                    "shot_scale": "close_up",
                    "subject_extent": "head and upper torso",
                },
                "target_subject": {
                    "visible_body_parts": body_parts or [],
                    "interactions": interactions or [],
                },
                "non_target_entities": entities or [],
            }
        }

    def _fusion(self, interactions: list[dict] | None = None) -> dict:
        return {
            "fusion": {
                "qualified_interactions": interactions or [],
                "qualified_body_parts": [],
            }
        }

    def _gestalt(
        self,
        posture: str,
        confidence: float = 0.90,
        basis: str = "contextual",
        supports: list[dict] | None = None,
    ) -> dict:
        return {
            "gestalt": {
                "schema_version": "pose-gestalt-1.0",
                "posture": posture,
                "posture_basis": basis,
                "posture_confidence": confidence,
                "body_configuration": posture,
                "support_configuration": supports or [],
                "semantic_pose_summary": posture,
                "evidence": [],
                "counterevidence": [],
            }
        }

    def _surface_support(self, body_part: str = "left forearm", status: str = "observed") -> dict:
        return {
            "body_part": body_part,
            "relation": "resting_on",
            "target": "table or desk surface",
            "evidence_status": status,
            "confidence": 0.95,
        }

    def test_probe_observed_forearm_without_visible_segment_is_advisory_only(self) -> None:
        points = self._portrait_points()
        points["left_wrist"] = (0.58, 0.60)
        result = build_pose_semantics(
            self._dwpose(points),
            self._fusion(),
            self._analysis(),
            self._gestalt("seated", supports=[self._surface_support()]),
        )
        self.assertIsNone(result["preferred_pose"]["posture"])
        self.assertEqual(result["posture_candidate"]["label"], "seated")
        self.assertFalse(result["pose_gestalt_corroboration"]["valid"])
        probe_gestures = [
            item for item in result["gestures"]
            if "pose_gestalt_" in " ".join(str(v) for v in item.get("support") or [])
        ]
        self.assertTrue(probe_gestures)
        self.assertFalse(any(item.get("caption_preferred") for item in probe_gestures))

    def test_visible_arm_on_table_alone_does_not_establish_seated(self) -> None:
        points = self._portrait_points()
        points.update({
            "left_elbow": (0.65, 0.50),
            "left_wrist": (0.70, 0.70),
        })
        result = build_pose_semantics(
            self._dwpose(points),
            self._fusion(),
            self._analysis("The subject leans forward over a table."),
            self._gestalt("seated", supports=[self._surface_support()]),
        )
        self.assertIsNone(result["preferred_pose"]["posture"])
        self.assertEqual(result["posture_candidate"]["label"], "seated")
        self.assertGreater(result["pose_gestalt_corroboration"]["verified_activity_support_count"], 0)

    def test_analyze_seated_plus_verified_activity_support_promotes(self) -> None:
        points = self._portrait_points()
        points.update({
            "left_elbow": (0.65, 0.50),
            "left_wrist": (0.70, 0.70),
        })
        result = build_pose_semantics(
            self._dwpose(points),
            self._fusion(),
            self._analysis("The subject is seated at a desk using a laptop."),
            self._gestalt("seated", supports=[self._surface_support()]),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "seated")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "seated_analyze_gestalt_plus_verified_activity_support",
        )

    def test_seat_specific_scene_promotes_seated_without_lower_body(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._fusion(),
            self._analysis("The subject is seated in an airplane seat with the headrest behind her."),
            self._gestalt("seated", confidence=0.90),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "seated")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "seated_analyze_gestalt_seat_scene_agreement",
        )

    def test_independent_car_seat_back_support_is_posture_bearing(self) -> None:
        entities = [{
            "description": "dark leather car seat backrest",
            "contact": "subject's back resting on seat",
            "support": "supporting subject's back",
            "confidence": 0.95,
        }]
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._fusion(),
            self._analysis("A close portrait inside a vehicle.", entities=entities),
            self._gestalt("seated", confidence=0.90),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "seated")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "seated_posture_bearing_support",
        )

    def test_standing_promotes_with_analyze_agreement_and_bilateral_hip_knee(self) -> None:
        points = self._portrait_points()
        points.update({
            "right_hip": (0.45, 0.50),
            "right_knee": (0.46, 0.72),
            "left_hip": (0.55, 0.50),
            "left_knee": (0.56, 0.72),
        })
        result = build_pose_semantics(
            self._dwpose(points, extent="three_quarter_or_long"),
            self._fusion(),
            self._analysis("The subject stands on a beach."),
            self._gestalt("standing", confidence=0.95, basis="geometric"),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "standing")
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "standing_analyze_gestalt_plus_bilateral_hip_knee",
        )

    def test_close_portrait_standing_gestalt_remains_candidate(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._fusion(),
            self._analysis("A close portrait at an event."),
            self._gestalt("standing", confidence=0.90),
        )
        self.assertIsNone(result["preferred_pose"]["posture"])
        self.assertEqual(result["posture_candidate"]["label"], "standing")

    def test_squat_beats_seated_like_component_geometry(self) -> None:
        points = self._portrait_points()
        points.update({
            "right_hip": (0.42, 0.52),
            "right_knee": (0.62, 0.52),
            "right_ankle": (0.62, 0.78),
            "left_hip": (0.58, 0.52),
            "left_knee": (0.38, 0.52),
        })
        result = build_pose_semantics(
            self._dwpose(points, extent="three_quarter_or_long"),
            self._fusion(),
            self._analysis("The person is performing a squat exercise."),
            self._gestalt("squatting", confidence=0.95, basis="geometric"),
        )
        self.assertEqual(result["preferred_pose"]["posture"], "squatting")
        self.assertTrue(result["human_summary"].startswith("Squatting;"))
        self.assertEqual(
            result["pose_gestalt_corroboration"]["route"],
            "squatting_analyze_gestalt_plus_flexed_leg_geometry",
        )

    def test_reclining_top_down_only_remains_candidate(self) -> None:
        result = build_pose_semantics(
            self._dwpose(self._portrait_points()),
            self._fusion(),
            self._analysis("A smiling close portrait."),
            self._gestalt("reclining", confidence=0.90),
        )
        self.assertIsNone(result["preferred_pose"]["posture"])
        self.assertEqual(result["posture_candidate"]["label"], "reclining")


if __name__ == "__main__":
    unittest.main()
