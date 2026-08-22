from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_133 import (
    _guard_complementary_side_inference,
    _signed_required_claims,
    _sync_refined_laterality,
    lint_caption,
)


class ComposeGovernance133Tests(unittest.TestCase):
    def test_refined_side_syncs_legacy_anatomical_side_for_projection(self) -> None:
        payload = {
            "fusion": {
                "qualified_body_parts": [
                    {
                        "part": "left arm",
                        "anatomical_side": "right",
                        "fusion_v2": {
                            "qualified_anatomical_side": "left",
                            "laterality_selection_usable": True,
                        },
                    }
                ]
            }
        }
        out = _sync_refined_laterality(payload)
        self.assertEqual(out["fusion"]["qualified_body_parts"][0]["anatomical_side"], "left")
        self.assertEqual(payload["fusion"]["qualified_body_parts"][0]["anatomical_side"], "right")

    def test_unsigned_actor_strips_sided_same_family_target(self) -> None:
        evidence = {
            "pose_orientation": {
                "qualified_interactions": [
                    {
                        "actor_part": "arm",
                        "actor_anatomical_side": "unknown",
                        "target": "left arm",
                        "notes": "arm crossed over left arm",
                    }
                ]
            }
        }
        audit = {"blocked": []}
        _guard_complementary_side_inference(evidence, audit)
        item = evidence["pose_orientation"]["qualified_interactions"][0]
        self.assertEqual(item["target"], "arm")
        self.assertEqual(item["notes"], "arm crossed over arm")
        self.assertTrue(audit["blocked"])

    def test_signed_depth_creates_must_cover_claims(self) -> None:
        fusion = {
            "signed_depth_authority_audit": {
                "components": {
                    "shoulder": {
                        "action": "qualified",
                        "nearer_anatomical_side": "left",
                        "magnitude_deg": 17.982,
                    }
                },
                "torso_direction": {
                    "action": "qualified",
                    "nearer_anatomical_side": "left",
                },
            }
        }
        claims = _signed_required_claims(fusion)
        self.assertEqual([item["id"] for item in claims], [
            "signed_shoulder_nearer_relation",
            "signed_torso_depth_direction",
        ])
        self.assertEqual(claims[0]["magnitude_band"], "moderate")

    def test_linter_drops_only_cross_sentence_orientation_false_positive(self) -> None:
        evidence = {
            "caption_policy": {"trigger_token": "BLIND7"},
            "pose_orientation": {
                "visible_subject_parts": [],
                "qualified_interactions": [],
                "qualified_laterality": [{"side": "left", "body_family": "shoulder"}],
                "qualified_hand_sides": [],
                "whole_body_posture": {"allowed": []},
                "semantic_orientation": {
                    "head_pitch": {"direction": "down", "magnitude": "slight", "confidence": 0.9},
                    "head_yaw": {"direction": "side_unspecified", "magnitude": "slight", "confidence": 0.9},
                },
            },
            "hard_constraints": {"visibility": {"not_visible": []}},
            "required_claims": [],
            "required_scene_claims": [],
        }
        caption = "BLIND7 has her head tilted down. Left shoulder slightly forward and closer to camera."
        result = lint_caption(caption, evidence)
        self.assertFalse(any(v.get("type") == "orientation_side_invented_from_side_neutral_relation" for v in result["violations"]))

    def test_signed_torso_claim_warns_when_omitted_and_flags_frontal_contradiction(self) -> None:
        evidence = {
            "caption_policy": {"trigger_token": "BLIND7"},
            "pose_orientation": {
                "visible_subject_parts": [],
                "qualified_interactions": [],
                "qualified_laterality": [],
                "qualified_hand_sides": [],
                "whole_body_posture": {"allowed": []},
                "semantic_orientation": {},
            },
            "hard_constraints": {"visibility": {"not_visible": []}},
            "required_claims": [
                {
                    "id": "signed_torso_depth_direction",
                    "priority": "required",
                    "nearer_anatomical_side": "left",
                }
            ],
            "required_scene_claims": [],
        }
        result = lint_caption("BLIND7 stands upright. The torso is frontal.", evidence)
        self.assertTrue(any(w.get("claim_id") == "signed_torso_depth_direction" for w in result["warnings"]))
        self.assertTrue(any(v.get("type") == "contradicts_signed_torso_depth" for v in result["violations"]))


if __name__ == "__main__":
    unittest.main()
