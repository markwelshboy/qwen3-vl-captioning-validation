from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_fusion import fuse_semantic_v3
from qwen_caption_validate.semantic_v3_pose_adapter import adapt_pose_v016


class SemanticV3FusionTests(unittest.TestCase):
    def _extract(self) -> dict:
        return {"extract": {"schema_version": "visual-extract-3.0"}}

    def _gestalt(self) -> dict:
        return {
            "schema_valid": True,
            "gestalt": {
                "schema_version": "composition-gestalt-1.4",
                "framing": {"shot_scale": "close_up"},
                "environment": {"space": "indoor"},
                "background_regions": [],
                "foreground_relations": [],
            },
        }

    def _pose(self, *, public: str, candidate: str, recovery: bool, hand_reject: bool = False) -> dict:
        hand = {
            "geometry_match": not hand_reject,
            "support_class": "not_matched" if hand_reject else "matched",
            "crop_support": 0.0 if hand_reject else 0.8,
            "rejection_reason": "open_hand_distal_proximity_without_proximal_palm_wrist_support" if hand_reject else None,
            "side": "left",
            "v14_proximal_chain_guard": {"geometry_match": False if hand_reject else True},
        }
        return {
            "profile": {
                "schema_version": "sam3d-relational-pose-profile-0.16",
                "sam3d_projected_pose": {
                    "pose": public,
                    "best_candidate_pose": candidate,
                    "crop_support": 0.04 if public == "uncertain" else 0.24,
                    "support_class": "reconstruction_dominant" if public == "uncertain" else "moderately_crop_supported",
                    "winner_margin": 0.78 if public == "uncertain" else 0.22,
                    "assertion_authority": {
                        "selected_path": "already_withheld_before_v15" if public == "uncertain" else "posture_region_weights",
                        "selected_path_authority": 0.04 if public == "uncertain" else 0.24,
                        "withheld_reason": "insufficient_observed_support" if public == "uncertain" else None,
                        "authority_semantics": "pose_joint_corroboration_not_literal_visual_crop_extent",
                    },
                    "semantic_recovery": {
                        "needed": recovery,
                        "candidate_pose": candidate if recovery else None,
                        "reason": "strong_reconstruction_candidate_but_pose_joint_authority_is_insufficient" if recovery else None,
                        "candidate_score": 0.93 if recovery else None,
                        "winner_margin": 0.78 if recovery else None,
                        "recommended_fusion_action": "seek_independent_analyze_or_scene_semantics_for_broad_pose" if recovery else None,
                    },
                    "physical_governance": {
                        "governed_best_score": 0.93 if candidate == "standing" else 0.55,
                    },
                    "posture_modifier_diagnostic": {"pose_family_for_modifier": candidate, "suggested_modifiers": []},
                },
                "relations": {"head_supported_by_hand": hand},
                "policy": {},
            }
        }

    def test_pose_adapter_uses_public_authority_not_raw_reconstruction(self) -> None:
        pose = adapt_pose_v016(self._pose(public="uncertain", candidate="standing", recovery=True))
        self.assertEqual(pose["public"]["value"], "unknown")
        self.assertEqual(pose["public"]["authority_state"], "withheld")
        self.assertEqual(pose["reconstruction"]["best_candidate"], "standing")
        self.assertTrue(pose["semantic_recovery"]["needed"])

    def test_00018_style_control_stays_posture_unknown_and_withholds_hand_ownership(self) -> None:
        analyze = {
            "schema_valid": True,
            "source_extract_sha256": "abc",
            "analyze": {
                "schema_version": "semantic-analyze-3.0",
                "posture": {"value": "unknown", "confidence": 0.0, "assessment": "unknown"},
                "actions": [{"value": "posing"}],
                "interactions": [{
                    "type": "gesture",
                    "actor_part": "right_hand",
                    "actor_ownership": "target",
                    "target_ref": "target_subject",
                    "target_text": None,
                    "interpretation": "hand under chin",
                    "confidence": 0.9,
                    "limitations": [],
                }],
                "ownership_assessments": [{
                    "part": "right_hand",
                    "ownership": "target",
                    "confidence": 0.9,
                    "evidence": ["claimed connected chain"],
                    "limitations": [],
                }],
                "support_context": [],
            },
        }
        fused = fuse_semantic_v3(
            image_key="00018",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(public="uncertain", candidate="standing", recovery=True, hand_reject=True),
        )
        canonical = fused["canonical"]
        self.assertEqual(canonical["posture"]["value"], "unknown")
        self.assertEqual(canonical["posture"]["status"], "withheld")
        self.assertEqual(canonical["posture"]["pose_reconstruction"], "standing")
        self.assertEqual(canonical["interactions"][0]["actor_ownership"], "unknown")
        self.assertEqual(canonical["ownership_assessments"][0]["ownership"], "unknown")
        self.assertFalse(canonical["physical_relations"]["head_supported_by_hand"]["value"])
        self.assertGreaterEqual(len(fused["conflicts"]), 2)

    def test_seated_control_uses_public_pose_and_preserves_unresolved_exact_support(self) -> None:
        analyze = {
            "schema_valid": True,
            "source_extract_sha256": "def",
            "analyze": {
                "schema_version": "semantic-analyze-3.0",
                "posture": {"value": "seated", "confidence": 0.9, "assessment": "supported"},
                "actions": [{"value": "drinking"}, {"value": "typing"}],
                "interactions": [],
                "ownership_assessments": [],
                "support_context": [{
                    "subject_relation": "seated_on",
                    "target_ref": None,
                    "target_description": "surface under lap",
                    "target_status": "unresolved",
                    "confidence": 0.65,
                    "evidence": ["knees bent"],
                    "limitations": ["exact seat not visible"],
                }],
            },
        }
        fused = fuse_semantic_v3(
            image_key="seated",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(public="sitting", candidate="sitting", recovery=False),
        )
        canonical = fused["canonical"]
        self.assertEqual(canonical["posture"]["value"], "seated")
        self.assertEqual(canonical["posture"]["status"], "asserted")
        self.assertEqual(canonical["posture"]["authority"], "pose_v0.16_public")
        self.assertEqual(canonical["support_context"][0]["target_status"], "unresolved")
        self.assertIsNone(canonical["support_context"][0]["target_ref"])

    def test_semantic_recovery_can_restore_supported_broad_posture_without_using_reconstruction_as_vote(self) -> None:
        analyze = {
            "schema_valid": True,
            "source_extract_sha256": "ghi",
            "analyze": {
                "schema_version": "semantic-analyze-3.0",
                "posture": {"value": "seated", "confidence": 0.9, "assessment": "supported"},
                "actions": [],
                "interactions": [],
                "ownership_assessments": [],
                "support_context": [],
            },
        }
        fused = fuse_semantic_v3(
            image_key="recovery",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(public="uncertain", candidate="standing", recovery=True),
        )
        posture = fused["canonical"]["posture"]
        self.assertEqual(posture["value"], "seated")
        self.assertEqual(posture["status"], "recovered")
        self.assertEqual(posture["authority"], "semantic_recovery_via_analyze")
        self.assertEqual(posture["pose_reconstruction"], "standing")
        self.assertTrue(any(c["type"] == "withheld_reconstruction_disagrees_with_semantic_recovery" for c in fused["conflicts"]))

    def test_gestalt_is_composition_input_not_a_posture_vote(self) -> None:
        analyze = {
            "schema_valid": True,
            "source_extract_sha256": "jkl",
            "analyze": {
                "schema_version": "semantic-analyze-3.0",
                "posture": {"value": "unknown", "confidence": 0.0, "assessment": "unknown"},
                "actions": [], "interactions": [], "ownership_assessments": [], "support_context": [],
            },
        }
        gestalt = self._gestalt()
        gestalt["gestalt"]["composition_summary"] = "looks seated"
        fused = fuse_semantic_v3(
            image_key="no_vote",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=gestalt,
            pose_record=self._pose(public="uncertain", candidate="standing", recovery=True),
        )
        self.assertEqual(fused["canonical"]["posture"]["value"], "unknown")


if __name__ == "__main__":
    unittest.main()
