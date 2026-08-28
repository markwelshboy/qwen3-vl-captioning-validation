from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_142 import build_caption_projection, lint_caption


class ComposeGovernance142Tests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "fusion": {
                "schema_version": "analysis-fusion-2.3.5",
                "framing_audit": {},
                "orientation_semantics": {
                    "torso_yaw": {"direction": "unknown", "magnitude": "unknown", "confidence": 0.0},
                    "head_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                },
                "projected_body_axis_audit": {},
                "qualified_body_parts": [
                    {
                        "part": "head",
                        "anatomical_side": "midline",
                        "ownership": "target",
                        "visibility": "full",
                        "visible_subparts": ["face"],
                        "connectivity_to_target_chain": "connected_visible",
                        "geometry": "head facing forward with slight turn",
                        "contact": None,
                        "support": None,
                        "foreshortening": "none",
                        "confidence": 0.99,
                        "fusion_v2": {
                            "qualified_ownership": "target",
                            "qualified_anatomical_side": "midline",
                            "selection_usable": True,
                            "laterality_selection_usable": False,
                            "reasons": [],
                            "laterality_reasons": [],
                        },
                    }
                ],
                "qualified_interactions": [],
                "sam3d_geometry_audit": {
                    "target_provenance": {"context_risk": "normal"},
                    "landmark_visibility": {},
                },
                "qualified_upper_torso_depth_relation": {
                    "magnitude": "strong",
                    "relation": "upper torso strongly turned in depth, near side-on rather than square-on to the camera",
                    "authority": "qualified_visible_shoulder_depth_rotation",
                },
                "qualified_head_torso_relation": {
                    "magnitude": "strong",
                    "relation": "head turned substantially toward the camera relative to the strongly depth-turned upper torso",
                    "camera_relation": "toward_camera",
                    "authority": "qualified_shoulder_depth_plus_visible_camera_frontal_head_and_camera_gaze",
                },
            }
        }

    def _analysis(self) -> dict:
        return {
            "schema_version": "2.1",
            "framing": {},
            "target_subject": {
                "orientation": {},
                "gaze": {"target": "camera_lens", "image_direction": "image_center"},
                "expression_state": [],
                "geometry_landmark_visibility": {},
            },
            "scene": {},
            "non_target_entities": [],
            "embedded_depictions": [],
            "nuisance_regions": [],
            "image_summary": "",
        }

    def test_relative_head_and_upper_torso_relations_replace_ambiguous_yaw(self) -> None:
        evidence, _ = build_caption_projection(self._payload(), self._analysis())
        pose = evidence["pose_orientation"]
        self.assertNotIn("head_yaw", pose["semantic_orientation"])
        self.assertNotIn("torso_yaw", pose["semantic_orientation"])
        self.assertEqual(pose["head_torso_relation"]["camera_relation"], "toward_camera")
        self.assertIn("near side-on", pose["upper_torso_depth_relation"]["relation"])
        head = next(item for item in pose["visible_subject_parts"] if item["part"] == "head")
        self.assertIn("toward the camera", head["geometry"])
        ids = {item["id"] for item in evidence["required_claims"]}
        self.assertIn("head_turn_toward_camera_relative_torso", ids)
        self.assertIn("upper_torso_side_on_relation", ids)

    def test_provenance_review_blocks_synthetic_projection_relations(self) -> None:
        payload = self._payload()
        payload["fusion"]["sam3d_geometry_audit"]["target_provenance"]["context_risk"] = "requires_review"
        evidence, audit = build_caption_projection(payload, self._analysis())
        pose = evidence["pose_orientation"]
        self.assertNotIn("head_torso_relation", pose)
        self.assertNotIn("upper_torso_depth_relation", pose)
        ids = {item["id"] for item in evidence["required_claims"]}
        self.assertNotIn("head_turn_toward_camera_relative_torso", ids)
        self.assertNotIn("upper_torso_side_on_relation", ids)
        projection = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
        blocked = projection.get("blocked") or []
        self.assertTrue(any(item.get("reason") == "sam3d_target_provenance_requires_review" for item in blocked if isinstance(item, dict)))

    def test_forward_head_wording_is_rejected_when_relative_turn_is_required(self) -> None:
        evidence, _ = build_caption_projection(self._payload(), self._analysis())
        lint = lint_caption(
            "sH1Vx has a strongly angled torso. The head faces forward with a slight turn.",
            evidence,
        )
        self.assertTrue(any(item.get("type") == "contradicts_head_torso_camera_turn" for item in lint["violations"]))

    def test_natural_relative_pose_covers_both_new_claims(self) -> None:
        evidence, _ = build_caption_projection(self._payload(), self._analysis())
        lint = lint_caption(
            "sH1Vx has the upper torso turned nearly side-on to the camera while turning the head toward the camera.",
            evidence,
        )
        new_warnings = {
            item.get("claim_id")
            for item in lint["warnings"]
            if item.get("claim_id") in {"head_turn_toward_camera_relative_torso", "upper_torso_side_on_relation"}
        }
        self.assertEqual(new_warnings, set())
        self.assertFalse(any(item.get("type") == "contradicts_head_torso_camera_turn" for item in lint["violations"]))


if __name__ == "__main__":
    unittest.main()
