from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_140 import (
    _coalesce_depth_claims,
    _coalesce_pose_support,
    _orientation_violation_is_anatomy_bridge,
    _preferred_scene_entities,
    _scene_gestalt_claims,
    _support_claims,
)


class ComposeGovernance140Tests(unittest.TestCase):
    def test_signed_torso_subsumes_unsigned_depth_checklist(self) -> None:
        claims = [
            {"id": "shoulder_girdle_depth_rotation"},
            {"id": "pelvis_depth_rotation"},
            {"id": "combined_torso_depth_rotation"},
            {"id": "signed_shoulder_nearer_relation"},
            {"id": "signed_torso_depth_direction"},
        ]
        kept, dropped = _coalesce_depth_claims(claims)
        self.assertEqual(
            [item["id"] for item in kept],
            ["signed_shoulder_nearer_relation", "signed_torso_depth_direction"],
        )
        self.assertEqual(
            set(dropped),
            {
                "shoulder_girdle_depth_rotation",
                "pelvis_depth_rotation",
                "combined_torso_depth_rotation",
            },
        )

    def test_direct_hand_support_subsumes_forearm_via_hand(self) -> None:
        evidence = {
            "pose_orientation": {
                "visible_subject_parts": [
                    {
                        "part": "left hand",
                        "anatomical_side": "left",
                        "support": "supporting head",
                        "contact": "contact with chin",
                    },
                    {
                        "part": "left forearm",
                        "anatomical_side": "left",
                        "geometry": "forearm bent at elbow",
                        "support": "supporting head via hand",
                        "contact": "contact with chin via hand",
                    },
                ]
            }
        }
        audit: dict = {"blocked": []}
        _coalesce_pose_support(evidence, audit)
        parts = evidence["pose_orientation"]["visible_subject_parts"]
        self.assertEqual(parts[0]["support"], "supporting head")
        self.assertIsNone(parts[1]["support"])
        self.assertIsNone(parts[1]["contact"])
        claims = _support_claims(evidence)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["actor_part"], "left hand")
        self.assertEqual(claims[0]["semantic_target"], "head")

    def test_generic_surface_regions_do_not_become_scene_gestalt(self) -> None:
        evidence = {
            "environment_lighting": {
                "important_background_or_nuisance_regions": [
                    {"description": "gray speckled floor"},
                    {"description": "light-colored wall panels with metallic trim"},
                ],
                "scene": {"background_structure": {"notes": "smooth panels with vertical seams"}},
            },
            "non_target_entities": [],
        }
        self.assertEqual(_scene_gestalt_claims(evidence), [])

    def test_semantic_setting_can_replace_detailed_background_inventory(self) -> None:
        evidence = {
            "environment_lighting": {
                "important_background_or_nuisance_regions": [
                    {"description": "blurred background foliage and park elements"}
                ]
            },
            "non_target_entities": [],
        }
        claims = _scene_gestalt_claims(evidence)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["description"], "park setting")
        self.assertEqual(claims[0]["keywords"], ["park"])
        self.assertTrue(claims[0]["semantic_compression_allowed"])

    def test_distinctive_entities_are_preferred_over_generic_surfaces(self) -> None:
        evidence = {
            "non_target_entities": [
                {"description": "black backpack with white logo", "confidence": 0.9},
                {"description": "blue luggage cart with black bag", "confidence": 0.8},
                {"description": "uncertain small object", "confidence": 0.4},
            ]
        }
        preferred = _preferred_scene_entities(evidence)
        self.assertEqual(
            [item["description"] for item in preferred],
            ["black backpack with white logo", "blue luggage cart with black bag"],
        )

    def test_orientation_lint_does_not_attach_next_anatomical_noun(self) -> None:
        caption = "Head tilted down and turned slightly from frontal, left shoulder closer to camera."
        violation = {
            "type": "orientation_side_invented_from_side_neutral_relation",
            "text": "Head tilted down and turned slightly from frontal, left",
        }
        self.assertTrue(_orientation_violation_is_anatomy_bridge(caption, violation))

    def test_real_directional_head_claim_is_not_filtered(self) -> None:
        caption = "Head is turned left."
        violation = {
            "type": "orientation_side_invented_from_side_neutral_relation",
            "text": "Head is turned left",
        }
        self.assertFalse(_orientation_violation_is_anatomy_bridge(caption, violation))


if __name__ == "__main__":
    unittest.main()
