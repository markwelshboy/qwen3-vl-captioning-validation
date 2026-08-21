from __future__ import annotations

import copy
import unittest

from qwen_caption_validate.caption_evidence import build_caption_evidence


def _analysis() -> dict:
    def vis(state: str, confidence: float = 0.95) -> dict:
        return {"visibility": state, "confidence": confidence, "evidence": f"synthetic {state}"}

    return {
        "schema_version": "2.1",
        "image_summary": "A subject whose raw summary contains pose and clothing prose.",
        "framing": {
            "shot_scale": "medium",
            "subject_extent": "head through upper thighs",
            "subject_frame_coverage": "medium",
            "photographic_archetype": "portrait",
        },
        "target_subject": {
            "orientation": {
                "torso_yaw": {"direction": "anatomical_left", "magnitude": "moderate", "confidence": 0.9},
                "torso_pitch": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                "torso_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                "image_plane_body_axis": {"direction": "upright", "magnitude": "none", "confidence": 0.95},
                "head_yaw": {"direction": "frontal", "magnitude": "none", "confidence": 0.95},
                "head_pitch": {"direction": "neutral", "magnitude": "none", "confidence": 0.95},
                "head_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.95},
            },
            "gaze": {"target": "camera_lens", "image_direction": "image_right", "notes": "looks to the right"},
            "expression_state": ["neutral"],
            "geometry_landmark_visibility": {
                "head": vis("visible"),
                "left_shoulder": vis("visible"),
                "right_shoulder": vis("visible"),
                "left_hip": vis("not_visible", 0.99),
                "right_hip": vis("not_visible", 0.99),
                "left_knee": vis("not_visible", 0.99),
                "right_knee": vis("not_visible", 0.99),
                "left_ankle": vis("not_visible", 0.99),
                "right_ankle": vis("not_visible", 0.99),
            },
            "visible_body_parts": [],
            "interactions": [],
        },
        "scene": {
            "environment_type": "indoor",
            "environment_confidence": 0.95,
            "illumination": {"type": "artificial", "directionality": "flat", "contrast": "medium", "notes": None},
        },
        "non_target_entities": [],
        "embedded_depictions": [],
        "nuisance_regions": [],
        "uncertainties": ["left hand may be outside the crop"],
    }


def _fusion(*, provenance_risk: bool = False, projected_conflict: bool = False) -> dict:
    return {
        "fusion": {
            "schema_version": "analysis-fusion-2.3",
            "framing_audit": {
                "semantic_framing": _analysis()["framing"],
                "qualified_shot_scale": "medium",
                "override_applied": False,
            },
            "orientation_semantics": _analysis()["target_subject"]["orientation"],
            "projected_body_axis_audit": {"conflict": projected_conflict},
            "qualified_body_parts": [
                {
                    "part": "right_hand",
                    "anatomical_side": "right",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": ["hand", "fingers"],
                    "connectivity_to_target_chain": "connected_visible",
                    "geometry": "right hand raised with a slight lean to anatomical_left",
                    "contact": None,
                    "support": None,
                    "foreshortening": "mild",
                    "image_location": "right side of frame",
                    "confidence": 0.95,
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "right",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                    },
                },
                {
                    "part": "fingers",
                    "anatomical_side": "left",
                    "ownership": "target",
                    "visibility": "fragment",
                    "visible_subparts": ["two fingertips"],
                    "connectivity_to_target_chain": "disconnected_in_crop",
                    "geometry": None,
                    "contact": None,
                    "support": None,
                    "foreshortening": "unknown",
                    "image_location": "edge",
                    "confidence": 0.7,
                    "fusion_v2": {
                        "qualified_ownership": "unknown",
                        "qualified_anatomical_side": "unknown",
                        "selection_usable": False,
                        "laterality_selection_usable": False,
                    },
                },
            ],
            "qualified_interactions": [
                {
                    "type": "holding",
                    "actor_part": "right hand",
                    "actor_ownership": "target",
                    "target": "cup",
                    "evidence_status": "observed",
                    "confidence": 0.95,
                    "notes": "right hand grips cup",
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
            "uncertainties": ["both hands may be near the abdomen"],
            "sam3d_geometry_audit": {
                "landmark_visibility": _analysis()["target_subject"]["geometry_landmark_visibility"],
                "target_provenance": {
                    "context_risk": "requires_review" if provenance_risk else "no_semantic_multi_subject_risk_detected"
                },
                "shoulder_depth_rotation": {
                    "magnitude_deg": 68.0,
                    "authority": "qualified_component_geometry",
                },
                "hip_depth_rotation": {
                    "magnitude_deg": 70.0,
                    "authority": "reconstructed_prior_only",
                },
                "torso_depth_rotation": {
                    "magnitude_deg": 69.0,
                    "authority": "report_only_partial_image_support",
                },
                "signed_depth_diagnostics": {"authority": "diagnostic_only_sign_not_validated"},
            },
        }
    }


class CaptionEvidenceTests(unittest.TestCase):
    def test_only_qualified_visible_sam3d_component_is_exposed(self) -> None:
        evidence, audit = build_caption_evidence(_fusion(), _analysis())
        self.assertEqual(evidence["schema_version"], "caption-evidence-1.1")
        self.assertEqual(
            evidence["qualified_3d_geometry"]["shoulder_girdle_depth_rotation"]["magnitude_band"],
            "very_high",
        )
        self.assertNotIn("pelvis_depth_rotation", evidence["qualified_3d_geometry"])
        self.assertNotIn("combined_torso_depth_rotation", evidence["qualified_3d_geometry"])
        self.assertIn("left_hip", evidence["visibility_constraints"]["not_visible"])
        self.assertTrue(any(item.get("reason") == "reconstructed_prior_only" for item in audit["blocked"]))

    def test_high_qualified_3d_geometry_becomes_required_claim(self) -> None:
        evidence, _ = build_caption_evidence(_fusion(), _analysis())
        self.assertEqual(len(evidence["required_claims"]), 1)
        self.assertEqual(evidence["required_claims"][0]["id"], "shoulder_girdle_depth_rotation")
        self.assertEqual(evidence["required_claims"][0]["magnitude_band"], "very_high")

    def test_provenance_risk_blocks_even_qualified_sam3d_components(self) -> None:
        evidence, audit = build_caption_evidence(_fusion(provenance_risk=True), _analysis())
        self.assertEqual(evidence["qualified_3d_geometry"], {})
        self.assertEqual(evidence["required_claims"], [])
        self.assertTrue(any(item.get("reason") == "target_provenance_requires_review" for item in audit["blocked"]))

    def test_unqualified_body_part_is_removed_and_laterality_is_redacted(self) -> None:
        evidence, _ = build_caption_evidence(_fusion(), _analysis())
        self.assertEqual(len(evidence["visible_subject_parts"]), 1)
        part = evidence["visible_subject_parts"][0]
        self.assertEqual(part["anatomical_side"], "unknown")
        self.assertNotIn("right", part["part"].lower())
        self.assertNotIn("right", str(part["geometry"]).lower())
        self.assertNotIn("anatomical_left", str(part["geometry"]).lower())
        self.assertIn("side-unspecified", str(part["geometry"]).lower())
        self.assertNotIn("image_location", part)

        interaction = evidence["qualified_interactions"][0]
        self.assertEqual(interaction["actor_anatomical_side"], "unknown")
        self.assertNotIn("right", interaction["actor_part"].lower())
        self.assertNotIn("right", str(interaction["notes"]).lower())

    def test_plural_hand_interaction_requires_two_distinct_qualified_hands(self) -> None:
        fused = copy.deepcopy(_fusion())
        fused["fusion"]["qualified_interactions"][0]["actor_part"] = "both hands"
        fused["fusion"]["qualified_interactions"][0]["notes"] = "both hands hold the cup"
        evidence, audit = build_caption_evidence(fused, _analysis())
        self.assertEqual(evidence["qualified_interactions"], [])
        self.assertTrue(
            any(
                item.get("reason") == "plural_hand_interaction_lacks_two_distinct_qualified_hands"
                for item in audit["blocked"]
            )
        )

    def test_gaze_direction_is_explicitly_frame_relative_and_notes_are_withheld(self) -> None:
        evidence, _ = build_caption_evidence(_fusion(), _analysis())
        self.assertEqual(evidence["gaze"]["frame_direction"], "right_side_of_image_frame")
        self.assertNotIn("notes", evidence["gaze"])

    def test_unqualified_semantic_anatomical_direction_becomes_side_unspecified(self) -> None:
        evidence, audit = build_caption_evidence(_fusion(), _analysis())
        self.assertEqual(evidence["semantic_orientation"]["torso_yaw"]["direction"], "side_unspecified")
        self.assertTrue(
            any(item.get("reason") == "anatomical_direction_not_independently_qualified" for item in audit["blocked"])
        )

    def test_projected_geometry_conflict_withholds_semantic_image_plane_axis(self) -> None:
        evidence, audit = build_caption_evidence(_fusion(projected_conflict=True), _analysis())
        self.assertNotIn("image_plane_body_axis", evidence["semantic_orientation"])
        self.assertTrue(
            any(item.get("reason") == "conflicts_with_deterministic_projected_geometry" for item in audit["blocked"])
        )

    def test_raw_summary_and_uncertainties_are_not_exposed_to_compose(self) -> None:
        evidence, audit = build_caption_evidence(_fusion(), _analysis())
        self.assertNotIn("image_summary", evidence)
        self.assertNotIn("uncertainties", evidence)
        self.assertTrue(any(item.get("path") == "analysis.image_summary" for item in audit["blocked"]))
        self.assertTrue(any(item.get("path") == "fusion.uncertainties" for item in audit["blocked"]))


if __name__ == "__main__":
    unittest.main()
