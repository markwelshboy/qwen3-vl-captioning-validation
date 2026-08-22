from __future__ import annotations

import unittest

from qwen_caption_validate.caption_lint import lint_caption
from qwen_caption_validate.caption_projection import (
    _extract_transient_phrases,
    _pose_parts,
    _project_interactions,
    _project_orientation,
    _project_required_claims,
    _project_scene_nuisance_regions,
    _project_3d_geometry,
    _qualified_laterality_refs,
    _qualified_whole_body_posture,
)


class CaptionProjection132Tests(unittest.TestCase):
    def test_richer_old_cache_appearance_is_recovered(self) -> None:
        text = (
            "She wears a brown and black striped cardigan over a light-colored top, "
            "black leggings and brown boots. Another image uses a dark textured turtleneck sweater."
        )
        descriptors = {value.lower() for value in _extract_transient_phrases(text)}
        self.assertIn("brown and black striped cardigan", descriptors)
        self.assertIn("light-colored top", descriptors)
        self.assertIn("black leggings", descriptors)
        self.assertIn("brown boots", descriptors)
        self.assertIn("dark textured turtleneck sweater", descriptors)

    def test_plain_top_is_not_extracted_from_on_top_of(self) -> None:
        descriptors = {value.lower() for value in _extract_transient_phrases("A lamp sits on top of a wooden box.")}
        self.assertNotIn("top", descriptors)

    def test_bilateral_complete_leg_chains_plus_floor_support_qualifies_standing(self) -> None:
        parts = [
            {
                "part": "legs", "anatomical_side": "left", "laterality_qualified": True,
                "geometry": "leg straight, boot visible", "contact": None, "support": "standing on floor",
            },
            {
                "part": "legs", "anatomical_side": "right", "laterality_qualified": True,
                "geometry": "leg straight, boot visible", "contact": None, "support": "standing on floor",
            },
        ]
        deterministic = {
            "connectivity": {
                "left_leg": {"complete": True},
                "right_leg": {"complete": True},
            }
        }
        posture = _qualified_whole_body_posture(parts, deterministic)
        self.assertIn("standing", posture["allowed"])

    def test_bilateral_support_without_complete_leg_chains_does_not_qualify_standing(self) -> None:
        parts = [
            {"part": "legs", "anatomical_side": "left", "laterality_qualified": True, "support": "standing on floor"},
            {"part": "legs", "anatomical_side": "right", "laterality_qualified": True, "support": "standing on floor"},
        ]
        deterministic = {"connectivity": {"left_leg": {"complete": False}, "right_leg": {"complete": False}}}
        posture = _qualified_whole_body_posture(parts, deterministic)
        self.assertNotIn("standing", posture["allowed"])

    def test_cross_body_laterality_is_sanitized_per_referenced_entity(self) -> None:
        parts = [
            {
                "part": "right_arm", "anatomical_side": "right", "laterality_qualified": True,
                "visible_subparts": ["shoulder", "upper arm", "elbow"],
                "visibility": "partial", "geometry": "arm bent", "contact": "right arm in contact with left arm",
                "support": None, "foreshortening": "mild",
            },
            {
                "part": "arm", "anatomical_side": "unknown", "laterality_qualified": False,
                "visible_subparts": ["upper arm", "forearm"],
                "visibility": "partial", "geometry": "arm bent", "contact": None, "support": None,
                "foreshortening": "mild",
            },
            {
                "part": "left_thigh", "anatomical_side": "left", "laterality_qualified": True,
                "visible_subparts": ["upper thigh"],
                "visibility": "partial", "geometry": "thigh bent", "contact": "left thigh in contact with left hand",
                "support": "thigh supporting body weight", "foreshortening": "mild",
            },
            {
                "part": "hand", "anatomical_side": "unknown", "laterality_qualified": False,
                "visible_subparts": ["palm", "fingers"],
                "visibility": "fragment", "geometry": "hand relaxed", "contact": "hand in contact with thigh",
                "support": None, "foreshortening": "mild",
            },
        ]
        refs = _qualified_laterality_refs(parts)
        audit = {"allowed": [], "blocked": [], "notes": []}
        projected = _pose_parts(parts, set(), audit, refs)
        right_arm = next(item for item in projected if item["anatomical_side"] == "right")
        left_thigh = next(item for item in projected if item["part"] == "left thigh")
        self.assertIn("the other arm", right_arm["contact"])
        self.assertNotIn("left arm", right_arm["contact"])
        self.assertIn("the hand", left_thigh["contact"])
        self.assertNotIn("left hand", left_thigh["contact"])

        interactions = _project_interactions(
            [
                {
                    "type": "contact", "actor_part": "right arm", "actor_anatomical_side": "right",
                    "laterality_qualified": True, "target": "left arm", "notes": "right arm crossed over left arm",
                }
            ],
            refs,
            audit,
        )
        self.assertEqual(interactions[0]["target"], "arm")
        self.assertIn("the other arm", interactions[0]["notes"])

    def test_linter_does_not_let_left_thigh_license_left_hand(self) -> None:
        evidence = {
            "caption_policy": {"trigger_token": "BLIND7"},
            "pose_orientation": {
                "visible_subject_parts": [],
                "qualified_interactions": [],
                "qualified_laterality": [
                    {"side": "right", "body_family": "arm"},
                    {"side": "left", "body_family": "leg"},
                ],
                "qualified_hand_sides": [],
                "whole_body_posture": {"allowed": []},
                "semantic_orientation": {},
            },
            "hard_constraints": {"visibility": {"not_visible": []}},
            "required_claims": [], "required_scene_claims": [],
        }
        result = lint_caption("BLIND7 rests the left hand on the left thigh while the right arm crosses the torso.", evidence)
        self.assertFalse(result["passed"])
        violations = [v for v in result["violations"] if v.get("type") == "unqualified_anatomical_laterality"]
        self.assertTrue(any(v.get("body_family") == "hand" for v in violations))
        self.assertFalse(any(v.get("body_family") == "leg" for v in violations))

    def test_target_associated_nuisance_is_not_promoted_to_scene(self) -> None:
        audit = {"allowed": [], "blocked": [], "notes": []}
        base = [
            {"description": "patterned fabric of target's top", "frame_coverage": "small"},
            {"description": "overhead cabin lights and vents", "frame_coverage": "medium"},
        ]
        fusion = [
            {"description": "patterned fabric of target's top", "identity_relevance": "medium"},
            {"description": "overhead cabin lights and vents", "identity_relevance": "none"},
        ]
        projected = _project_scene_nuisance_regions(base, fusion, audit)
        self.assertEqual([x["description"] for x in projected], ["overhead cabin lights and vents"])

    def test_image_plane_direction_becomes_side_neutral_cant_relation(self) -> None:
        audit = {"allowed": [], "blocked": [], "notes": []}
        projected = _project_orientation(
            {"image_plane_body_axis": {"direction": "leans_image_right", "magnitude": "moderate", "confidence": 0.95}},
            audit,
        )
        axis = projected["image_plane_body_axis"]
        self.assertEqual(axis["relation"], "canted_from_vertical_in_image_plane")
        self.assertNotIn("direction", axis)
        self.assertNotIn("right", str(axis).lower())

    def test_identity_only_hair_record_does_not_reach_pose(self) -> None:
        audit = {"allowed": [], "blocked": [], "notes": []}
        parts = [
            {
                "part": "hair", "anatomical_side": "midline", "laterality_qualified": False,
                "visibility": "full", "geometry": "blonde, shoulder-length, slightly tousled",
                "contact": None, "support": None, "foreshortening": "none",
            },
            {
                "part": "head", "anatomical_side": "midline", "laterality_qualified": False,
                "visibility": "full", "geometry": "head tilted slightly down",
                "contact": None, "support": None, "foreshortening": "none",
            },
        ]
        projected = _pose_parts(parts, set(), audit)
        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0]["part"], "head")
        self.assertNotIn("blonde", str(projected).lower())

    def test_redaction_residue_is_removed(self) -> None:
        audit = {"allowed": [], "blocked": [], "notes": []}
        parts = [
            {
                "part": "torso", "anatomical_side": "midline", "laterality_qualified": False,
                "visibility": "partial", "geometry": "torso leaning forward, slightly rotated to",
                "contact": None, "support": None, "foreshortening": "mild",
            },
            {
                "part": "head", "anatomical_side": "midline", "laterality_qualified": False,
                "visibility": "full", "geometry": "head tilted down, face angled toward image",
                "contact": None, "support": None, "foreshortening": "none",
            },
        ]
        projected = _pose_parts(parts, set(), audit)
        self.assertEqual(projected[0]["geometry"], "torso leaning forward")
        self.assertEqual(projected[1]["geometry"], "head tilted down")

    def test_required_3d_claim_projection_removes_machine_constraints(self) -> None:
        projected = _project_required_claims([
            {
                "id": "shoulder_girdle_depth_rotation", "priority": "required", "magnitude_band": "high",
                "instruction": "Mention it", "constraints": ["unsigned", "do_not_name_anatomical_side_from_this_claim"],
            }
        ])
        self.assertNotIn("constraints", projected[0])
        self.assertNotIn("unsigned", str(projected).lower())
        geometry = _project_3d_geometry(
            {"shoulder_girdle_depth_rotation": {"magnitude_band": "high", "direction": "unsigned", "authority": "qualified_component_geometry"}}
        )
        self.assertEqual(geometry, {"shoulder_girdle_depth_rotation": {"magnitude_band": "high"}})

    def test_linter_catches_constraint_narration(self) -> None:
        evidence = {
            "caption_policy": {"trigger_token": "BLIND7"},
            "pose_orientation": {
                "visible_subject_parts": [], "qualified_interactions": [], "qualified_laterality": [],
                "qualified_hand_sides": [], "whole_body_posture": {"allowed": []}, "semantic_orientation": {},
            },
            "hard_constraints": {"visibility": {"not_visible": []}},
            "required_claims": [], "required_scene_claims": [],
        }
        result = lint_caption("BLIND7 has strong shoulder depth, with no signed anatomical side specified.", evidence)
        self.assertFalse(result["passed"])
        self.assertTrue(any(v.get("type") == "constraint_narration_meta_language" for v in result["violations"]))


if __name__ == "__main__":
    unittest.main()
