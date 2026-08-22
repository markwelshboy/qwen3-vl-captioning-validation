from __future__ import annotations

import unittest

from qwen_caption_validate.caption_lint import lint_caption
from qwen_caption_validate.caption_projection import build_caption_projection


def _analysis() -> dict:
    def vis(state: str, confidence: float = 0.95) -> dict:
        return {"visibility": state, "confidence": confidence, "evidence": f"synthetic {state}"}

    orientation = {
        "torso_yaw": {"direction": "anatomical_left", "magnitude": "moderate", "confidence": 0.9},
        "torso_pitch": {"direction": "reclined", "magnitude": "strong", "confidence": 0.95},
        "torso_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
        "image_plane_body_axis": {"direction": "near_horizontal", "magnitude": "strong", "confidence": 0.95},
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
        "image_summary": "Synthetic summary with identity traits and clothing.",
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
                    "contact": None,
                    "support": None,
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
                    "geometry": "reclined on back",
                    "contact": "resting on bed",
                    "support": "supported by bed",
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
    def test_projection_is_task_shaped_and_filters_identity_descriptors(self) -> None:
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
        self.assertEqual(evidence["schema_version"], "caption-evidence-1.2")
        descriptors = {value.lower() for value in evidence["transient_appearance"]["descriptors"]}
        self.assertIn("sunglasses", descriptors)
        self.assertIn("black tank top", descriptors)
        self.assertNotIn("hair", descriptors)
        self.assertNotIn("beard", descriptors)
        self.assertIn("facial hair", evidence["caption_policy"]["protected_traits"])
        self.assertIn("natural hair color", evidence["caption_policy"]["protected_traits"])
        self.assertIn("projection", audit)

    def test_distal_pruning_removes_phantom_hand_and_plural_interaction(self) -> None:
        evidence, audit = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        parts = evidence["pose_orientation"]["visible_subject_parts"]
        unknown_arm = next(item for item in parts if item["part"] == "arm")
        self.assertNotIn("hand", str(unknown_arm.get("geometry") or "").lower())
        self.assertNotIn("hand", str(unknown_arm.get("contact") or "").lower())
        interactions = evidence["pose_orientation"]["qualified_interactions"]
        self.assertEqual(interactions, [])
        blocked = audit["projection"]["blocked"]
        self.assertTrue(
            any(
                item.get("reason") == "distal_hand_claim_withheld_without_deterministic_wrist_or_hand_root_support"
                for item in blocked
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

    def test_high_shoulder_depth_remains_required(self) -> None:
        evidence, _ = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        self.assertEqual(evidence["required_claims"][0]["id"], "shoulder_girdle_depth_rotation")
        self.assertEqual(evidence["required_claims"][0]["magnitude_band"], "very_high")

    def test_linter_catches_side_invented_from_side_unspecified(self) -> None:
        evidence, _ = build_caption_projection(_fusion(), _analysis(), caption_policy={"trigger_token": "sH1Vx"})
        bad = lint_caption(
            "sH1Vx reclines with the head turned moderately to the right and the shoulders strongly staggered in depth.",
            evidence,
        )
        self.assertFalse(bad["passed"])
        self.assertTrue(
            any(item.get("type") == "orientation_side_invented_from_side_unspecified" for item in bad["violations"])
        )

        good = lint_caption(
            "sH1Vx reclines with the head turned moderately to one side and the shoulders strongly staggered in depth.",
            evidence,
        )
        self.assertTrue(good["passed"])
        self.assertEqual(good["warning_count"], 0)


if __name__ == "__main__":
    unittest.main()
