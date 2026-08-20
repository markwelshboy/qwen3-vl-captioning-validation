from __future__ import annotations

import unittest

from qwen_caption_validate.fusion_v2 import (
    _camera_audit,
    _framing_audit,
    _qualify_body_parts,
    _qualify_interactions,
)


def _pose_with_supported_hand(side: str = "left") -> dict:
    return {
        "hand_candidates": [
            {
                "supported_by_nearby_visible_target_wrist": True,
                "nearest_visible_target_wrist": side,
                "target_arm_chain_complete": True,
            }
        ]
    }


class FusionV22Tests(unittest.TestCase):
    def test_isolated_unknown_fingers_remain_non_authoritative(self) -> None:
        analysis = {
            "framing": {"photographic_archetype": "candid"},
            "target_subject": {
                "visible_body_parts": [
                    {
                        "part": "fingers",
                        "anatomical_side": "unknown",
                        "ownership": "unknown",
                        "visibility": "fragment",
                        "connectivity_to_target_chain": "disconnected_in_crop",
                    }
                ],
                "interactions": [
                    {
                        "type": "contact",
                        "actor_part": "fingers",
                        "actor_ownership": "unknown",
                        "evidence_status": "observed",
                    }
                ],
            },
        }
        parts, _ = _qualify_body_parts(analysis, {"hand_candidates": []})
        self.assertEqual(parts[0]["fusion_v2"]["qualified_ownership"], "unknown")
        self.assertFalse(parts[0]["fusion_v2"]["selection_usable"])

        interactions, _ = _qualify_interactions(analysis, parts)
        self.assertEqual(interactions[0]["fusion_v2"]["qualified_actor_ownership"], "unknown")
        self.assertFalse(interactions[0]["fusion_v2"]["selection_usable"])

    def test_supported_target_action_survives_laterality_conflict(self) -> None:
        analysis = {
            "framing": {"photographic_archetype": "selfie"},
            "target_subject": {
                "visible_body_parts": [
                    {
                        "part": "fingers",
                        "anatomical_side": "right",
                        "ownership": "target",
                        "visibility": "fragment",
                        "connectivity_to_target_chain": "connected_visible",
                    }
                ],
                "interactions": [
                    {
                        "type": "holding",
                        "actor_part": "right hand fingers",
                        "actor_ownership": "target",
                        "evidence_status": "observed",
                    }
                ],
            },
        }
        parts, _ = _qualify_body_parts(analysis, _pose_with_supported_hand("left"))
        fusion = parts[0]["fusion_v2"]
        self.assertEqual(fusion["qualified_ownership"], "target")
        self.assertTrue(fusion["selection_usable"])
        self.assertEqual(fusion["qualified_anatomical_side"], "unknown")
        self.assertFalse(fusion["laterality_selection_usable"])

        interactions, _ = _qualify_interactions(analysis, parts)
        interaction = interactions[0]["fusion_v2"]
        self.assertEqual(interaction["qualified_actor_ownership"], "target")
        self.assertTrue(interaction["selection_usable"])
        self.assertEqual(interaction["qualified_actor_anatomical_side"], "unknown")
        self.assertFalse(interaction["laterality_selection_usable"])

    def test_full_length_extent_reconciles_semantic_three_quarter(self) -> None:
        analysis = {
            "framing": {
                "shot_scale": "three_quarter",
                "subject_extent": "mid-thighs to head",
            },
            "target_subject": {
                "visible_body_parts": [
                    {"part": "foot", "visible_subparts": ["shoe", "sock"]},
                    {"part": "leg", "visible_subparts": ["thigh", "calf"]},
                ]
            },
        }
        geometry = {
            "pose_extent_hint": "full_length",
            "connectivity": {
                "left_leg": {"complete": True},
                "right_leg": {"complete": True},
            },
        }
        audit = _framing_audit(analysis, geometry)
        self.assertTrue(audit["override_applied"])
        self.assertEqual(audit["qualified_shot_scale"], "full_length")
        self.assertTrue(audit["conflict"])

    def test_eye_level_without_geometric_evidence_is_not_qualified(self) -> None:
        analysis = {
            "camera": {
                "elevation": "eye_level",
                "elevation_confidence": 0.9,
                "elevation_evidence": [
                    "subject's eyes are at approximately the same height as the camera lens",
                    "no strong view down onto shoulders or up under chin",
                ],
                "elevation_counterevidence": [],
            }
        }
        audit = _camera_audit(analysis)
        self.assertFalse(audit["qualified_semantic_evidence"])
        self.assertEqual(audit["qualified_geometric_evidence"], [])


if __name__ == "__main__":
    unittest.main()
