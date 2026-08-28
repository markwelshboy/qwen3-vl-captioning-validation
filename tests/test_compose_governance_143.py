from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_143 import (
    _chin_gesture_claim,
    _framing_label_and_description,
    _qualify_cropped_standing,
    _salient_interaction_claims,
    _sanitize_cropped_leg_kinematics,
)
from qwen_caption_validate.compose_governance_143 import _GOVERNANCE_ADDENDUM


class ComposeGovernance143Tests(unittest.TestCase):
    def test_mid_thigh_extent_normalizes_to_three_quarter(self) -> None:
        framing = {
            "shot_scale": "medium_close_up",
            "subject_extent": "Upper body from mid-thighs to top of head",
        }
        label, description, _ = _framing_label_and_description(framing, {})
        self.assertEqual(label, "three_quarter")
        self.assertIn("three-quarter", description)

    def test_explicit_feet_crop_outranks_dwpose_ankle_extent(self) -> None:
        framing = {
            "shot_scale": "full_length",
            "subject_extent": "head to mid-calf, with feet partially cropped",
        }
        fusion = {"deterministic_geometry": {"pose_extent_hint": "full_length"}}
        label, description, _ = _framing_label_and_description(framing, fusion)
        self.assertEqual(label, "near_full_length")
        self.assertIn("cropped", description)

    def test_cropped_leg_kinematics_are_withheld_without_ankle(self) -> None:
        evidence = {
            "pose_orientation": {
                "visible_subject_parts": [
                    {
                        "part": "right leg",
                        "anatomical_side": "right",
                        "geometry": "standing, knee slightly bent",
                        "support": "standing on sand",
                    },
                    {
                        "part": "left leg",
                        "anatomical_side": "left",
                        "geometry": "standing, knee slightly bent",
                        "support": "standing on sand",
                    },
                ]
            }
        }
        fused = {
            "fusion": {
                "deterministic_geometry": {
                    "connectivity": {
                        "right_leg": {"visible": ["right_hip", "right_knee"], "visible_count": 2, "complete": False},
                        "left_leg": {"visible": ["left_hip", "left_knee"], "visible_count": 2, "complete": False},
                    }
                }
            }
        }
        audit = {"blocked": []}
        _sanitize_cropped_leg_kinematics(evidence, fused, audit)
        parts = evidence["pose_orientation"]["visible_subject_parts"]
        self.assertEqual(parts[0]["geometry"], "standing")
        self.assertEqual(parts[1]["geometry"], "standing")
        self.assertIsNone(parts[0]["support"])
        self.assertIsNone(parts[1]["support"])
        reasons = {item["reason"] for item in audit["blocked"]}
        self.assertIn("knee_angle_withheld_without_complete_hip_knee_ankle_chain", reasons)
        self.assertIn("distal_ground_support_withheld_without_visible_ankle_foot_chain", reasons)

    def test_cropped_standing_can_be_qualified_without_ankles(self) -> None:
        evidence = {"pose_orientation": {"whole_body_posture": {"allowed": [], "authority": "direct_visible_support_only", "evidence": []}}}
        fused = {
            "fusion": {
                "qualified_body_parts": [
                    {
                        "part": "right leg",
                        "anatomical_side": "left",
                        "geometry": "standing, knee slightly bent",
                        "support": "standing on sand",
                        "fusion_v2": {"selection_usable": True, "laterality_selection_usable": True, "qualified_anatomical_side": "right"},
                    },
                    {
                        "part": "left leg",
                        "anatomical_side": "right",
                        "geometry": "standing, knee slightly bent",
                        "support": "standing on sand",
                        "fusion_v2": {"selection_usable": True, "laterality_selection_usable": True, "qualified_anatomical_side": "left"},
                    },
                ],
                "deterministic_geometry": {
                    "connectivity": {
                        "right_leg": {"visible_count": 2, "complete": False},
                        "left_leg": {"visible_count": 2, "complete": False},
                    }
                },
            }
        }
        analysis = {"image_summary": "A woman stands on a beach with one hand on her hip."}
        audit = {"allowed": []}
        claim = _qualify_cropped_standing(evidence, fused, analysis, audit)
        self.assertIsNotNone(claim)
        self.assertIn("standing", evidence["pose_orientation"]["whole_body_posture"]["allowed"])
        self.assertEqual(claim["id"], "whole_body_posture_standing")

    def test_high_confidence_hand_on_hip_is_salient(self) -> None:
        evidence = {
            "pose_orientation": {
                "qualified_interactions": [
                    {
                        "type": "contact",
                        "actor_part": "right hand",
                        "actor_anatomical_side": "right",
                        "target": "hip",
                        "confidence": 0.95,
                    }
                ]
            }
        }
        claims = _salient_interaction_claims(evidence)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["description"], "right hand rests on the hip")

    def test_curled_chin_support_compresses_without_inventing_fist(self) -> None:
        evidence = {
            "pose_orientation": {
                "visible_subject_parts": [
                    {
                        "part": "left hand",
                        "anatomical_side": "left",
                        "laterality_qualified": True,
                        "geometry": "fingers curled under chin",
                        "contact": "under chin",
                        "support": "supporting head",
                    }
                ]
            }
        }
        audit = {"allowed": []}
        claim = _chin_gesture_claim(evidence, audit)
        self.assertIsNotNone(claim)
        self.assertEqual(claim["description"], "chin resting on the left hand")
        self.assertNotIn("fist", claim["description"])

    def test_prompt_forbids_reconstructing_cropped_leg_kinematics(self) -> None:
        self.assertIn("do not reconstruct \"bent legs\"", _GOVERNANCE_ADDENDUM)
        self.assertIn("High-confidence interactions", _GOVERNANCE_ADDENDUM)
        self.assertIn("normalized_shot_scale", _GOVERNANCE_ADDENDUM)


if __name__ == "__main__":
    unittest.main()
