from __future__ import annotations

import unittest

from qwen_caption_validate.contact_orientation_refine_235 import refine_contact_orientation


def _dw(visible: list[str]) -> dict:
    def chain(side: str) -> dict:
        names = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
        seen = [name for name in names if name in visible]
        return {"visible": seen, "visible_count": len(seen), "complete": len(seen) == 3}

    return {
        "derived": {
            "target": {
                "visible_body_landmarks": visible,
                "connectivity": {
                    "left_arm": chain("left"),
                    "right_arm": chain("right"),
                    "left_leg": {"visible": [], "visible_count": 0, "complete": False},
                    "right_leg": {"visible": [], "visible_count": 0, "complete": False},
                },
            },
            "hand_candidates": [],
        }
    }


def _state(side: str = "left") -> dict:
    return {
        "qualified_ownership": "target",
        "qualified_anatomical_side": side,
        "selection_usable": True,
        "laterality_selection_usable": True,
        "reasons": [],
        "laterality_reasons": [],
    }


class ContactOrientation235Tests(unittest.TestCase):
    def test_unobserved_thigh_vetoes_hand_contact_support_and_interaction(self) -> None:
        payload = {
            "fusion": {
                "schema_version": "analysis-fusion-2.3.4",
                "orientation_semantics": {},
                "qualified_body_parts": [
                    {
                        "part": "left arm",
                        "anatomical_side": "left",
                        "geometry": "arm bent at elbow, hand resting near thigh",
                        "contact": "hand resting on thigh",
                        "support": "hand supporting weight on thigh",
                        "fusion_v2": _state("left"),
                    },
                    {
                        "part": "left hand",
                        "anatomical_side": "left",
                        "geometry": "fingers slightly curled, palm down",
                        "contact": "hand resting on thigh",
                        "support": "hand supporting weight on thigh",
                        "fusion_v2": _state("left"),
                    },
                ],
                "qualified_interactions": [
                    {
                        "type": "support",
                        "actor_part": "left_hand",
                        "target": "left_thigh",
                        "fusion_v2": {
                            "qualified_actor_anatomical_side": "left",
                            "laterality_selection_usable": True,
                            "selection_usable": True,
                            "reasons": [],
                        },
                    }
                ],
                "sam3d_geometry_audit": {},
            }
        }
        dw = _dw(["nose", "neck", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder"])
        out = refine_contact_orientation(payload, dw, {"target_subject": {"gaze": {"target": "camera_lens"}}})
        fusion = out["fusion"]
        arm, hand = fusion["qualified_body_parts"]
        self.assertEqual(arm["geometry"], "arm bent at elbow")
        self.assertIsNone(arm["contact"])
        self.assertIsNone(arm["support"])
        self.assertIsNone(hand["contact"])
        self.assertIsNone(hand["support"])
        self.assertFalse(fusion["qualified_interactions"][0]["fusion_v2"]["selection_usable"])
        self.assertGreaterEqual(len(fusion["self_contact_support_audit"]["blocked_body_fields"]), 4)

    def test_observed_hand_to_head_support_survives(self) -> None:
        payload = {
            "fusion": {
                "qualified_body_parts": [
                    {
                        "part": "left hand",
                        "anatomical_side": "left",
                        "contact": "contact with chin",
                        "support": "supporting head",
                        "fusion_v2": _state("left"),
                    }
                ],
                "qualified_interactions": [
                    {
                        "type": "support",
                        "actor_part": "left_hand",
                        "target": "head",
                        "fusion_v2": {
                            "qualified_actor_anatomical_side": "left",
                            "laterality_selection_usable": True,
                            "selection_usable": True,
                            "reasons": [],
                        },
                    }
                ],
                "orientation_semantics": {},
                "sam3d_geometry_audit": {},
            }
        }
        dw = _dw(["nose", "neck", "left_eye", "right_eye", "left_shoulder", "left_elbow", "left_wrist"])
        out = refine_contact_orientation(payload, dw, {"target_subject": {"gaze": {"target": "camera_lens"}}})
        hand = out["fusion"]["qualified_body_parts"][0]
        self.assertEqual(hand["support"], "supporting head")
        self.assertEqual(hand["contact"], "contact with chin")
        self.assertTrue(out["fusion"]["qualified_interactions"][0]["fusion_v2"]["selection_usable"])

    def test_observed_hand_to_hip_contact_survives(self) -> None:
        payload = {
            "fusion": {
                "qualified_body_parts": [],
                "qualified_interactions": [
                    {
                        "type": "contact",
                        "actor_part": "right_hand",
                        "target": "right_hip",
                        "fusion_v2": {
                            "qualified_actor_anatomical_side": "right",
                            "laterality_selection_usable": True,
                            "selection_usable": True,
                            "reasons": [],
                        },
                    }
                ],
                "orientation_semantics": {},
                "sam3d_geometry_audit": {},
            }
        }
        dw = _dw(["right_shoulder", "right_elbow", "right_wrist", "right_hip"])
        out = refine_contact_orientation(payload, dw, {})
        self.assertTrue(out["fusion"]["qualified_interactions"][0]["fusion_v2"]["selection_usable"])

    def test_strong_shoulder_depth_suppresses_weak_frontal_torso_and_derives_head_turn(self) -> None:
        payload = {
            "fusion": {
                "orientation_semantics": {
                    "torso_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                    "head_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                },
                "qualified_body_parts": [],
                "qualified_interactions": [],
                "sam3d_geometry_audit": {
                    "shoulder_depth_rotation": {"authority": "qualified_component_geometry", "magnitude_deg": 80.0},
                    "landmark_visibility": {"head": {"visibility": "visible", "confidence": 0.99}},
                },
            }
        }
        out = refine_contact_orientation(payload, _dw([]), {"target_subject": {"gaze": {"target": "camera_lens"}}})
        fusion = out["fusion"]
        self.assertEqual(fusion["orientation_semantics"]["torso_yaw"]["direction"], "unknown")
        self.assertEqual(fusion["qualified_upper_torso_depth_relation"]["magnitude"], "strong")
        self.assertEqual(fusion["qualified_head_torso_relation"]["camera_relation"], "toward_camera")
        self.assertTrue(fusion["orientation_consistency_audit"]["suppressed_semantic_torso_yaw"])

    def test_moderate_shoulder_depth_does_not_create_relative_head_relation(self) -> None:
        payload = {
            "fusion": {
                "orientation_semantics": {
                    "torso_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                    "head_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                },
                "qualified_body_parts": [],
                "qualified_interactions": [],
                "sam3d_geometry_audit": {
                    "shoulder_depth_rotation": {"authority": "qualified_component_geometry", "magnitude_deg": 30.0},
                    "landmark_visibility": {"head": {"visibility": "visible", "confidence": 0.99}},
                },
            }
        }
        out = refine_contact_orientation(payload, _dw([]), {"target_subject": {"gaze": {"target": "camera_lens"}}})
        self.assertNotIn("qualified_head_torso_relation", out["fusion"])
        self.assertEqual(out["fusion"]["orientation_semantics"]["torso_yaw"]["direction"], "frontal")


if __name__ == "__main__":
    unittest.main()
