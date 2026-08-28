from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_140 import (
    _coalesce_depth_claims,
    _coalesce_pose_support,
    _extract_summary_apparel,
    _orientation_violation_is_anatomy_bridge,
    _preferred_scene_entities,
    _qualify_side_neutral_standing,
    _scene_claims,
    _scene_gestalt_claims,
    _support_claims,
    lint_caption,
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

    def test_generic_surface_regions_do_not_become_scene_claims(self) -> None:
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
        self.assertEqual(_scene_claims(evidence), [])

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

    def test_concrete_nuisance_objects_remain_protected_from_omission(self) -> None:
        evidence = {
            "environment_lighting": {
                "important_background_or_nuisance_regions": [
                    {"description": "background clutter including yellow bag and boxes"}
                ]
            },
            "non_target_entities": [],
        }
        claims = _scene_claims(evidence)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["description"], "background clutter including yellow bag and boxes")
        self.assertEqual(claims[0]["keywords"], ["bag", "box"])
        self.assertEqual(claims[0]["minimum_keyword_matches"], 2)
        preferred = _preferred_scene_entities(evidence)
        self.assertEqual(preferred[0]["description"], "background clutter including yellow bag and boxes")

    def test_distinctive_entities_are_preferred_over_generic_surfaces(self) -> None:
        evidence = {
            "environment_lighting": {"important_background_or_nuisance_regions": []},
            "non_target_entities": [
                {"description": "black backpack with white logo", "confidence": 0.9},
                {"description": "blue luggage cart with black bag", "confidence": 0.8},
                {"description": "uncertain small object", "confidence": 0.4},
            ],
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

    def test_extended_apparel_quarantine_recovers_halter_and_bottoms(self) -> None:
        analysis = {
            "image_summary": (
                "A woman stands barefoot in a kitchen, wearing a floral halter top and "
                "black high-waisted bottoms, holding a dark patterned garment."
            )
        }
        self.assertEqual(
            _extract_summary_apparel(analysis),
            ["floral halter top", "black high-waisted bottoms"],
        )

    def test_side_neutral_full_body_support_can_qualify_standing(self) -> None:
        evidence = {
            "pose_orientation": {
                "whole_body_posture": {"allowed": [], "authority": "direct_visible_support_only", "evidence": []},
                "visible_subject_parts": [
                    {"part": "torso", "visibility": "full", "support": "on feet"},
                    {"part": "leg", "visibility": "full", "support": "weight-bearing"},
                    {"part": "leg", "visibility": "full", "support": "weight-bearing"},
                    {"part": "feet", "visibility": "partial", "contact": "touching floor", "support": "standing on floor"},
                ],
            }
        }
        audit: dict = {"allowed": []}
        _qualify_side_neutral_standing(evidence, audit)
        self.assertIn("standing", evidence["pose_orientation"]["whole_body_posture"]["allowed"])

    def test_partial_legs_without_feet_do_not_qualify_standing(self) -> None:
        evidence = {
            "pose_orientation": {
                "whole_body_posture": {"allowed": [], "authority": "direct_visible_support_only", "evidence": []},
                "visible_subject_parts": [
                    {"part": "torso", "visibility": "full", "support": "on feet"},
                    {"part": "left leg", "visibility": "partial", "support": "on sand"},
                    {"part": "right leg", "visibility": "partial", "support": "on sand"},
                ],
            }
        }
        audit: dict = {"allowed": []}
        _qualify_side_neutral_standing(evidence, audit)
        self.assertNotIn("standing", evidence["pose_orientation"]["whole_body_posture"]["allowed"])

    def test_supporting_chin_satisfies_hand_supporting_head_claim(self) -> None:
        evidence = {
            "caption_policy": {},
            "pose_orientation": {
                "whole_body_posture": {"allowed": []},
                "qualified_laterality": [{"side": "left", "body_family": "hand"}],
                "qualified_hand_sides": ["left"],
                "visible_subject_parts": [
                    {
                        "part": "left hand",
                        "anatomical_side": "left",
                        "laterality_qualified": True,
                        "geometry": "hand curled, fingers supporting chin",
                        "contact": "contact with chin",
                        "support": "supporting head",
                    }
                ],
            },
            "required_claims": [
                {
                    "id": "support_relation_1",
                    "description": "left hand: supporting head",
                    "support_text": "supporting head",
                    "actor_part": "left hand",
                    "semantic_target": "head",
                }
            ],
            "required_scene_claims": [],
            "hard_constraints": {"visibility": {}},
        }
        result = lint_caption("TOKEN supports the chin with the left hand.", evidence)
        self.assertEqual(result["warning_count"], 0)

    def test_negative_appearance_absence_is_not_authorized(self) -> None:
        evidence = {
            "caption_policy": {},
            "pose_orientation": {"whole_body_posture": {"allowed": []}},
            "required_claims": [],
            "required_scene_claims": [],
            "hard_constraints": {"visibility": {}},
        }
        result = lint_caption("TOKEN is wearing no visible clothing or accessories.", evidence)
        self.assertTrue(any(v["type"] == "unsupported_negative_appearance_claim" for v in result["violations"]))

    def test_torso_angled_depth_is_not_contradicted_by_head_frontal_wording(self) -> None:
        evidence = {
            "caption_policy": {},
            "pose_orientation": {"whole_body_posture": {"allowed": []}},
            "required_claims": [
                {
                    "id": "signed_torso_depth_direction",
                    "nearer_anatomical_side": "left",
                    "description": "the torso is angled in depth rather than square-on to the camera",
                }
            ],
            "required_scene_claims": [],
            "hard_constraints": {"visibility": {}},
        }
        caption = "TOKEN torso is angled in depth rather than square-on, head turned slightly from frontal."
        result = lint_caption(caption, evidence)
        self.assertFalse(any(v["type"] == "contradicts_signed_torso_depth" for v in result["violations"]))


if __name__ == "__main__":
    unittest.main()
