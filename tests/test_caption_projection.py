from __future__ import annotations

import json
import unittest

from qwen_caption_validate.caption_lint import lint_caption
from qwen_caption_validate.caption_projection import build_caption_projection


def _analysis() -> dict:
    def vis(state: str, confidence: float = 0.95) -> dict:
        return {"visibility": state, "confidence": confidence, "evidence": f"synthetic {state}"}

    orientation = {
        "torso_yaw": {"direction": "anatomical_left", "magnitude": "moderate", "confidence": 0.9},
        "torso_pitch": {"direction": "backward", "magnitude": "moderate", "confidence": 0.9},
        "torso_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
        "image_plane_body_axis": {"direction": "near-horizontal", "magnitude": "strong", "confidence": 0.95},
        "head_yaw": {"direction": "anatomical_right", "magnitude": "moderate", "confidence": 0.9},
        "head_pitch": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
        "head_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
    }
    visibility = {
        "head": vis("visible"),
        "left_shoulder": vis("visible"),
        "right_shoulder": vis("visible"),
        "left_hip": vis("not_visible"),
        "right_hip": vis("not_visible"),
        "left_knee": vis("not_visible"),
        "right_knee": vis("not_visible"),
        "left_ankle": vis("not_visible"),
        "right_ankle": vis("not_visible"),
    }
    return {
        "schema_version": "2.1",
        "image_summary": (
            "A person with brown hair and a beard wears a purple hoodie, black pants and a headband, "
            "with hair pulled back. Tattoos are visible on both arms."
        ),
        "framing": {
            "shot_scale": "medium_close_up",
            "subject_extent": "head through waist",
            "subject_frame_coverage": "large",
            "photographic_archetype": "candid",
        },
        "target_subject": {
            "orientation": orientation,
            "gaze": {"target": "off_camera", "image_direction": "image_right", "notes": None},
            "expression_state": ["relaxed"],
            "geometry_landmark_visibility": visibility,
            "visible_body_parts": [],
            "interactions": [],
        },
        "scene": {
            "environment_type": "indoor",
            "environment_confidence": 0.95,
            "illumination": {"type": "natural", "directionality": "soft", "contrast": "low", "notes": None},
        },
        "non_target_entities": [],
        "embedded_depictions": [],
        "nuisance_regions": [],
        "uncertainties": ["synthetic uncertainty must never reach Compose"],
    }


def _fusion() -> dict:
    analysis = _analysis()
    return {
        "fusion": {
            "schema_version": "analysis-fusion-2.3",
            "framing_audit": {
                "semantic_framing": analysis["framing"],
                "qualified_shot_scale": "medium_close_up",
                "override_applied": False,
            },
            "orientation_semantics": analysis["target_subject"]["orientation"],
            "projected_body_axis_audit": {"conflict": False},
            "deterministic_geometry": {
                "connectivity": {
                    "left_arm": {
                        "visible": ["left_shoulder", "left_elbow", "left_wrist"],
                        "visible_count": 3,
                        "complete": True,
                    },
                    "right_arm": {
                        "visible": ["right_shoulder", "right_elbow"],
                        "visible_count": 2,
                        "complete": False,
                    },
                },
                "hand_candidates": [
                    {
                        "supported_by_nearby_visible_target_wrist": True,
                        "nearest_visible_target_wrist": "left",
                    }
                ],
            },
            "qualified_body_parts": [
                {
                    "part": "head",
                    "anatomical_side": "midline",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": ["face", "eyes", "hair", "beard", "sunglasses"],
                    "connectivity_to_target_chain": "connected_visible",
                    "geometry": "head turned moderately to anatomical_right",
                    "contact": "resting on bedspread",
                    "support": "supported by bedspread",
                    "foreshortening": "none",
                    "image_location": "upper right",
                    "confidence": 0.95,
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "midline",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                    },
                },
                {
                    "part": "torso",
                    "anatomical_side": "midline",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": ["upper chest", "black tank top"],
                    "connectivity_to_target_chain": "connected_visible",
                    "geometry": "reclined, lying on back",
                    "contact": "resting on bedspread",
                    "support": "supported by bedspread",
                    "foreshortening": "mild",
                    "image_location": "center",
                    "confidence": 0.95,
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "midline",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                    },
                },
                {
                    "part": "left arm",
                    "anatomical_side": "left",
                    "ownership": "target",
                    "visibility": "partial",
                    "visible_subparts": ["shoulder", "upper arm", "forearm", "hand"],
                    "connectivity_to_target_chain": "connected_visible",
                    "geometry": "arm bent at elbow, hand resting on abdomen",
                    "contact": "hand resting on abdomen",
                    "support": "supported by torso",
                    "foreshortening": "mild",
                    "image_location": "left side",
                    "confidence": 0.9,
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "left",
                        "selection_usable": True,
                        "laterality_selection_usable": True,
                    },
                },
                {
                    "part": "right arm",
                    "anatomical_side": "right",
                    "ownership": "target",
                    "visibility": "partial",
                    "visible_subparts": ["shoulder", "upper arm", "forearm", "hand"],
                    "connectivity_to_target_chain": "connected_visible",
                    "geometry": "arm bent at elbow, hand resting on abdomen",
                    "contact": "hand resting on abdomen",
                    "support": "supported by torso",
                    "foreshortening": "mild",
                    "image_location": "right side",
                    "confidence": 0.9,
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "unknown",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                    },
                },
            ],
            "qualified_interactions": [
                {
                    "type": "contact",
                    "actor_part": "hands",
                    "actor_ownership": "target",
                    "target": "abdomen",
                    "evidence_status": "observed",
                    "confidence": 0.95,
                    "notes": "both hands are resting on the abdomen",
                    "fusion_v2": {
                        "qualified_actor_ownership": "target",
                        "qualified_actor_anatomical_side": "unknown",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                    },
                }
            ],
            "non_target_entities": [],
            "embedded_depictions": [],
            "nuisance_regions": [],
            "uncertainties": ["synthetic uncertainty"],
            "sam3d_geometry_audit": {
                "landmark_visibility": analysis["target_subject"]["geometry_landmark_visibility"],
                "target_provenance": {"context_risk": "no_semantic_multi_subject_risk_detected"},
                "shoulder_depth_rotation": {
                    "magnitude_deg": 61.0,
                    "authority": "qualified_component_geometry",
                },
                "hip_depth_rotation": {
                    "magnitude_deg": 55.0,
                    "authority": "reconstructed_prior_only",
                },
                "torso_depth_rotation": {
                    "magnitude_deg": 58.0,
                    "authority": "report_only_partial_image_support",
                },
            },
        }
    }


class CaptionProjectionTests(unittest.TestCase):
    def test_projection_13_is_task_shaped_and_salvages_only_transient_appearance(self) -> None:
        evidence, audit = build_caption_projection(
            _fusion(),
            _analysis(),
            caption_policy={
                "trigger_token": "sH1Vx",
                "subject_pronoun": "they",
                "object_pronoun": "them",
                "possessive_pronoun": "their",
                "protected_traits": ["facial hair"],
            },
        )
        self.assertEqual(evidence["schema_version"], "caption-evidence-1.3")
        descriptors = {value.lower() for value in evidence["transient_appearance"]["descriptors"]}
        self.assertIn("sunglasses", descriptors)
        self.assertIn("black tank top", descriptors)
        self.assertIn("purple hoodie", descriptors)
        self.assertIn("black pants", descriptors)
        self.assertIn("headband", descriptors)
        self.assertIn("hair pulled back", descriptors)
        self.assertNotIn("hair", descriptors)
        self.assertNotIn("beard", descriptors)
        self.assertNotIn("tattoo", " ".join(descriptors))
        self.assertNotIn("coverage_limitations", evidence)
        self.assertTrue(
            any(
                item.get("path") == "analysis.image_summary[appearance-only quarantine]"
                for item in audit["projection"]["allowed"]
            )
        )

    def test_mixed_scene_summary_keeps_appearance_without_reclassifying_background_objects(self) -> None:
        analysis = _analysis()
        analysis["image_summary"] = (
            "A shirtless person stands in a wooded outdoor environment, wearing blue shorts, white socks, "
            "and dark shoes. A yellow bag and boxes are visible in the background."
        )
        analysis["scene"]["background_structure"] = {
            "texture_complexity": "high",
            "structural_complexity": "high",
            "specular_reflective": "none",
            "repeated_geometry": None,
            "strong_lines_or_angles": "medium",
            "reflections_present": False,
            "notes": "dense trees, moss-covered logs, and green undergrowth",
        }
        fusion = _fusion()
        fusion["fusion"]["nuisance_regions"] = [
            {
                "description": "yellow bag and boxes in background",
                "image_location": "background",
                "frame_coverage": "medium",
                "texture_complexity": "medium",
                "structural_complexity": "medium",
                "specular_reflective": "none",
                "identity_relevance": "low",
                "pose_relevance": "low",
                "entropy_focus_candidate": True,
            }
        ]

        evidence, audit = build_caption_projection(fusion, analysis, caption_policy={"trigger_token": "sH1Vx"})
        descriptors = {value.lower() for value in evidence["transient_appearance"]["descriptors"]}
        self.assertIn("shirtless", descriptors)
        self.assertIn("blue shorts", descriptors)
        self.assertIn("white socks", descriptors)
        self.assertIn("dark shoes", descriptors)
        self.assertFalse(any("bag" in value or "box" in value for value in descriptors))

        background = evidence["environment_lighting"]["scene"]["background_structure"]
        self.assertIn("dense trees", background["notes"])
        regions = evidence["environment_lighting"]["important_background_or_nuisance_regions"]
        self.assertTrue(any("yellow bag" in str(item.get("description") or "") for item in regions))
        self.assertTrue(
            any(
                item.get("path") == "analysis.scene.background_structure"
                and item.get("reason") == "structured_scene_context_is_caption_safe"
                for item in audit["projection"]["allowed"]
            )
        )

    def test_side_unspecified_internal_label_never_reaches_compose(self) -> None:
        evidence, _ = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        orientation = evidence["pose_orientation"]["semantic_orientation"]
        self.assertEqual(orientation["torso_yaw"]["relation"], "turned_from_frontal")
        self.assertEqual(orientation["head_yaw"]["relation"], "turned_from_frontal")
        self.assertNotIn("side_unspecified", json.dumps(evidence))

    def test_distal_pruning_removes_phantom_hand_and_plural_interaction(self) -> None:
        evidence, audit = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        parts = evidence["pose_orientation"]["visible_subject_parts"]
        unknown_arm = next(item for item in parts if item["part"] == "arm")
        self.assertNotIn("hand", str(unknown_arm.get("geometry") or "").lower())
        self.assertNotIn("hand", str(unknown_arm.get("contact") or "").lower())
        self.assertEqual(evidence["pose_orientation"]["qualified_interactions"], [])
        self.assertTrue(
            any(
                item.get("reason") == "distal_hand_claim_withheld_without_deterministic_wrist_or_hand_root_support"
                for item in audit["projection"]["blocked"]
            )
        )

    def test_horizontal_gaze_direction_is_withheld(self) -> None:
        evidence, audit = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        self.assertEqual(evidence["pose_orientation"]["gaze"], {"target": "off_camera"})
        self.assertTrue(
            any(
                item.get("reason") == "horizontal_frame_direction_withheld_from_caption_projection"
                for item in audit["projection"]["blocked"]
            )
        )

    def test_direct_support_qualifies_lying_and_reclined_but_not_standing(self) -> None:
        evidence, _ = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        allowed = set(evidence["pose_orientation"]["whole_body_posture"]["allowed"])
        self.assertEqual(allowed, {"lying", "reclined"})

        bad = lint_caption(
            "sH1Vx stands in a medium close-up with the shoulders strongly staggered in depth.",
            evidence,
        )
        self.assertFalse(bad["passed"])
        self.assertTrue(any(item.get("type") == "unsupported_whole_body_posture" for item in bad["violations"]))

        good = lint_caption(
            "sH1Vx lies reclined with the shoulders strongly staggered in depth.",
            evidence,
        )
        self.assertTrue(good["passed"])

    def test_high_shoulder_depth_remains_required(self) -> None:
        evidence, _ = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        self.assertEqual(evidence["required_claims"][0]["id"], "shoulder_girdle_depth_rotation")
        self.assertEqual(evidence["required_claims"][0]["magnitude_band"], "very_high")

    def test_linter_catches_side_invented_from_side_neutral_relation(self) -> None:
        evidence, _ = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        bad = lint_caption(
            "sH1Vx lies reclined with the head turned moderately to the right and the shoulders strongly staggered in depth.",
            evidence,
        )
        self.assertFalse(bad["passed"])
        self.assertTrue(
            any(item.get("type") == "orientation_side_invented_from_side_neutral_relation" for item in bad["violations"])
        )

        good = lint_caption(
            "sH1Vx lies reclined with the head turned moderately away from frontal and the shoulders strongly staggered in depth.",
            evidence,
        )
        self.assertTrue(good["passed"])
        self.assertEqual(good["warning_count"], 0)


if __name__ == "__main__":
    unittest.main()