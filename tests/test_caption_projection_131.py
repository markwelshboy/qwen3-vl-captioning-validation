from __future__ import annotations

import unittest

from qwen_caption_validate.caption_lint import lint_caption
from qwen_caption_validate.caption_projection import (
    _dedupe_hand_observations,
    _extract_transient_phrases,
    _pose_parts,
    _qualified_whole_body_posture,
    _required_scene_claims,
)


class CaptionProjection131Tests(unittest.TestCase):
    def test_plain_shirt_is_transient_appearance(self) -> None:
        descriptors = {value.lower() for value in _extract_transient_phrases("They wear a white shirt and blue shorts.")}
        self.assertIn("white shirt", descriptors)
        self.assertIn("blue shorts", descriptors)

    def test_grounded_feet_supporting_body_weight_qualifies_standing(self) -> None:
        posture = _qualified_whole_body_posture(
            [
                {
                    "part": "feet",
                    "geometry": "feet planted on ground",
                    "contact": "contact with ground",
                    "support": "supporting body weight",
                }
            ]
        )
        self.assertIn("standing", posture["allowed"])

    def test_unauthorized_posture_is_removed_but_useful_geometry_survives(self) -> None:
        audit = {"allowed": [], "blocked": [], "notes": []}
        projected = _pose_parts(
            [
                {
                    "part": "upper_legs",
                    "anatomical_side": "unknown",
                    "visibility": "partial",
                    "geometry": "standing, legs slightly apart",
                    "contact": None,
                    "support": "standing on elevator floor",
                    "foreshortening": "none",
                    "laterality_qualified": False,
                }
            ],
            set(),
            audit,
        )
        self.assertEqual(projected[0]["geometry"], "legs slightly apart")
        self.assertIsNone(projected[0]["support"])
        self.assertNotIn("standing", str(projected).lower())
        self.assertTrue(
            any(
                item.get("reason") == "unauthorized_whole_body_posture_removed_from_subordinate_pose_text"
                for item in audit["blocked"]
            )
        )

    def test_side_unspecified_meta_label_is_naturalized_in_freeform_pose(self) -> None:
        audit = {"allowed": [], "blocked": [], "notes": []}
        projected = _pose_parts(
            [
                {
                    "part": "head",
                    "anatomical_side": "midline",
                    "visibility": "full",
                    "geometry": "head turned to side-unspecified",
                    "contact": None,
                    "support": None,
                    "foreshortening": "none",
                    "laterality_qualified": False,
                }
            ],
            set(),
            audit,
        )
        self.assertEqual(projected[0]["geometry"], "head turned to the side")
        self.assertNotIn("unspecified", str(projected).lower())

    def test_conflicted_duplicate_hand_collapses_to_qualified_root(self) -> None:
        fusion = {
            "qualified_body_parts": [
                {
                    "part": "hand",
                    "anatomical_side": "right",
                    "ownership": "target",
                    "visibility": "fragment",
                    "visible_subparts": ["fingers", "palm"],
                    "geometry": "holding mug",
                    "contact": "holding black mug",
                    "support": None,
                    "image_location": "lower right",
                    "fusion_v2": {
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                        "qualified_anatomical_side": "unknown",
                        "laterality_reasons": [
                            "Analyze-v2 side=right conflicts with DWPose hand-root association to ['left']"
                        ],
                    },
                },
                {
                    "part": "hand",
                    "anatomical_side": "left",
                    "ownership": "target",
                    "visibility": "fragment",
                    "visible_subparts": ["fingers", "palm"],
                    "geometry": "holding mug",
                    "contact": "holding black mug",
                    "support": None,
                    "image_location": "lower right",
                    "fusion_v2": {
                        "selection_usable": True,
                        "laterality_selection_usable": True,
                        "qualified_anatomical_side": "left",
                        "laterality_reasons": ["agrees with deterministic root"],
                    },
                },
            ],
            "deterministic_geometry": {
                "hand_candidates": [
                    {
                        "supported_by_nearby_visible_target_wrist": True,
                        "nearest_visible_target_wrist": "left",
                    }
                ]
            },
        }
        audit = {"allowed": [], "blocked": [], "notes": []}
        _dedupe_hand_observations(fusion, audit)
        self.assertEqual(len(fusion["qualified_body_parts"]), 1)
        self.assertEqual(fusion["qualified_body_parts"][0]["anatomical_side"], "left")
        self.assertTrue(
            any(
                item.get("reason") == "duplicate_hand_observation_collapsed_to_qualified_deterministic_root"
                for item in audit["blocked"]
            )
        )

    def test_important_scene_region_becomes_linted_required_claim(self) -> None:
        claims = _required_scene_claims([{"description": "background clutter including yellow bag and boxes"}])
        self.assertEqual(claims[0]["keywords"], ["yellow", "bag", "box"])

        evidence = {
            "caption_policy": {"trigger_token": "sH1Vx"},
            "pose_orientation": {
                "visible_subject_parts": [],
                "qualified_interactions": [],
                "whole_body_posture": {"allowed": []},
                "semantic_orientation": {},
            },
            "hard_constraints": {"visibility": {"not_visible": []}},
            "required_claims": [],
            "required_scene_claims": claims,
        }
        missing = lint_caption("sH1Vx is indoors beside a painting and a lamp.", evidence)
        self.assertTrue(any(item.get("type") == "required_scene_claim_not_detected" for item in missing["warnings"]))

        present = lint_caption("sH1Vx is indoors, with a yellow bag and boxes visible in the background.", evidence)
        self.assertFalse(any(item.get("type") == "required_scene_claim_not_detected" for item in present["warnings"]))


if __name__ == "__main__":
    unittest.main()
