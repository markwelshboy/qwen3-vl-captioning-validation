from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.pose_semantics import BODY18, IDX, build_pose_semantics, calculate_angle


class PoseSemanticsTests(unittest.TestCase):
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

        visible = [name for name in BODY18 if name in points]
        connectivity = {
            "right_arm": chain(["right_shoulder", "right_elbow", "right_wrist"]),
            "left_arm": chain(["left_shoulder", "left_elbow", "left_wrist"]),
            "right_leg": chain(["right_hip", "right_knee", "right_ankle"]),
            "left_leg": chain(["left_hip", "left_knee", "left_ankle"]),
        }
        return {
            "raw_pose": {"bodies": {"candidate": person.tolist()}},
            "derived": {
                "target_person_index": 0,
                "target": {
                    "visible_body_landmarks": visible,
                    "pose_extent_hint": "full_length" if any("ankle" in name for name in visible) else "three_quarter_or_long",
                    "connectivity": connectivity,
                },
            },
        }

    def _standing_points(self, *, ankles: bool = True) -> dict[str, tuple[float, float]]:
        points = {
            "nose": (0.50, 0.10),
            "neck": (0.50, 0.20),
            "right_shoulder": (0.40, 0.22),
            "right_elbow": (0.38, 0.38),
            "right_wrist": (0.40, 0.52),
            "left_shoulder": (0.60, 0.22),
            "left_elbow": (0.62, 0.38),
            "left_wrist": (0.60, 0.52),
            "right_hip": (0.45, 0.50),
            "right_knee": (0.45, 0.70),
            "left_hip": (0.55, 0.50),
            "left_knee": (0.55, 0.70),
        }
        if ankles:
            points.update({"right_ankle": (0.45, 0.90), "left_ankle": (0.55, 0.90)})
        return points

    def test_calculate_angle(self) -> None:
        self.assertAlmostEqual(calculate_angle((0, 0), (1, 0), (2, 0)), 180.0)
        self.assertAlmostEqual(calculate_angle((0, 0), (1, 0), (1, 1)), 90.0)
        self.assertIsNone(calculate_angle(None, (1, 0), (1, 1)))

    def test_straight_complete_legs_classify_standing(self) -> None:
        result = build_pose_semantics(self._dwpose(self._standing_points()), {"fusion": {}}, {"analysis": {}})
        self.assertEqual(result["posture"]["status"], "qualified")
        self.assertEqual(result["posture"]["label"], "standing")
        self.assertEqual(result["posture"]["confidence_band"], "strong")
        self.assertGreaterEqual(result["geometry_features"]["angles_deg"]["right_knee"], 175.0)

    def test_seated_right_angle_geometry_classifies_seated(self) -> None:
        points = self._standing_points()
        points.update({
            "right_hip": (0.45, 0.50), "right_knee": (0.25, 0.50), "right_ankle": (0.25, 0.75),
            "left_hip": (0.55, 0.50), "left_knee": (0.75, 0.50), "left_ankle": (0.75, 0.75),
        })
        result = build_pose_semantics(self._dwpose(points), {"fusion": {}}, {"analysis": {}})
        self.assertEqual(result["posture"]["status"], "qualified")
        self.assertEqual(result["posture"]["label"], "seated")
        self.assertGreaterEqual(result["posture"]["support_score"], 0.70)

    def test_incomplete_cropped_legs_do_not_invent_knee_angle_but_can_corrobate_standing(self) -> None:
        dwpose = self._dwpose(self._standing_points(ankles=False))
        body_parts = []
        for side in ("left", "right"):
            body_parts.append({
                "part": f"{side} leg",
                "anatomical_side": side,
                "geometry": "standing, knee slightly bent",
                "support": "standing on sand",
                "fusion_v2": {
                    "selection_usable": True,
                    "laterality_selection_usable": True,
                    "qualified_anatomical_side": side,
                },
            })
        result = build_pose_semantics(
            dwpose,
            {"fusion": {"qualified_body_parts": body_parts}},
            {"analysis": {"image_summary": "A woman stands on a beach."}},
        )
        self.assertEqual(result["posture"]["label"], "standing")
        self.assertEqual(result["posture"]["confidence_band"], "strong")
        self.assertIsNone(result["geometry_features"]["angles_deg"]["right_knee"])
        self.assertIsNone(result["geometry_features"]["angles_deg"]["left_knee"])

    def test_hand_on_hip_interaction_becomes_semantic_gesture(self) -> None:
        points = self._standing_points()
        points["right_wrist"] = (0.46, 0.51)
        fusion = {
            "fusion": {
                "qualified_interactions": [{
                    "type": "contact",
                    "actor_part": "right hand",
                    "actor_anatomical_side": "right",
                    "target": "hip",
                    "confidence": 0.95,
                    "fusion_v2": {"selection_usable": True, "qualified_actor_anatomical_side": "right"},
                }]
            }
        }
        result = build_pose_semantics(self._dwpose(points), fusion, {"analysis": {}})
        labels = [item["label"] for item in result["gestures"] if item["caption_preferred"]]
        self.assertIn("right hand resting on the hip", labels)
        self.assertNotIn("right arm hanging at the side", labels)

    def test_bilateral_object_interactions_compress_to_both_hands(self) -> None:
        interactions = []
        for side in ("left", "right"):
            interactions.append({
                "type": "holding",
                "actor_part": f"{side} hand",
                "actor_anatomical_side": side,
                "target": "smartphone",
                "confidence": 0.95,
                "fusion_v2": {"selection_usable": True, "qualified_actor_anatomical_side": side},
            })
        result = build_pose_semantics(
            self._dwpose(self._standing_points()),
            {"fusion": {"qualified_interactions": interactions}},
            {"analysis": {}},
        )
        labels = [item["label"] for item in result["gestures"] if item["caption_preferred"]]
        self.assertIn("both hands holding smartphone", labels)
        self.assertNotIn("left hand holding smartphone", labels)
        self.assertNotIn("right hand holding smartphone", labels)

    def test_qualified_sam_depth_becomes_three_quarter_with_nearer_side(self) -> None:
        fusion = {
            "fusion": {
                "sam3d_geometry_audit": {
                    "target_provenance": {"context_risk": "no_semantic_multi_subject_risk_detected"},
                    "shoulder_depth_rotation": {"magnitude_deg": 46.0, "authority": "qualified_component_geometry"},
                    "hip_depth_rotation": {"magnitude_deg": 42.0, "authority": "qualified_component_geometry"},
                },
                "signed_depth_authority_audit": {
                    "torso_direction": {"action": "qualified", "nearer_anatomical_side": "right"},
                    "components": {},
                },
            }
        }
        result = build_pose_semantics(self._dwpose(self._standing_points()), fusion, {"analysis": {}})
        torso = result["torso_orientation"]
        self.assertEqual(torso["status"], "qualified")
        self.assertIn("three-quarter", torso["label"])
        self.assertEqual(torso["nearer_anatomical_side"], "right")
        self.assertEqual(torso["confidence_band"], "strong")

    def test_partial_upper_body_keeps_posture_unknown(self) -> None:
        points = {
            "nose": (0.50, 0.10), "neck": (0.50, 0.20),
            "right_shoulder": (0.42, 0.25), "right_elbow": (0.40, 0.45), "right_wrist": (0.40, 0.65),
            "left_shoulder": (0.58, 0.25), "left_elbow": (0.60, 0.45), "left_wrist": (0.60, 0.65),
        }
        result = build_pose_semantics(self._dwpose(points), {"fusion": {}}, {"analysis": {}})
        self.assertEqual(result["posture"]["status"], "withheld")
        self.assertIsNone(result["preferred_pose"]["posture"])


if __name__ == "__main__":
    unittest.main()
