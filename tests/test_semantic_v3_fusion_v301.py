from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_fusion_v301 import FUSION_VERSION, fuse_semantic_v3


class SemanticV3FusionV301Tests(unittest.TestCase):
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
                    "physical_governance": {"governed_best_score": 0.93 if candidate == "standing" else 0.55},
                    "posture_modifier_diagnostic": {
                        "pose_family_for_modifier": candidate,
                        "lean_severity": "upright",
                        "lean_direction": "direction_indeterminate",
                        "shoulder_line_tilt_severity": "level",
                        "suggested_modifiers": [],
                        "suggested_compound_pose_modifier": None,
                    },
                },
                "relations": {"head_supported_by_hand": hand},
                "policy": {},
            }
        }

    def _analyze(self, posture_value: str, confidence: float, assessment: str) -> dict:
        return {
            "schema_valid": True,
            "source_extract_sha256": "abc",
            "analyze": {
                "schema_version": "semantic-analyze-3.0",
                "posture": {"value": posture_value, "confidence": confidence, "assessment": assessment},
                "actions": [],
                "interactions": [],
                "ownership_assessments": [],
                "support_context": [],
            },
        }

    def test_public_pose_authority_is_not_mislabeled_as_confidence(self) -> None:
        analyze = self._analyze("seated", 0.9, "supported")
        fused = fuse_semantic_v3(
            image_key="seated",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(public="sitting", candidate="sitting", recovery=False),
        )
        posture = fused["canonical"]["posture"]
        self.assertEqual(fused["schema_version"], FUSION_VERSION)
        self.assertEqual(posture["confidence"], 0.9)
        self.assertEqual(posture["pose_joint_authority"], 0.24)
        self.assertIn("not probabilistic confidence", posture["pose_joint_authority_semantics"])

    def test_withheld_public_pose_cannot_leak_reconstruction_through_modifiers(self) -> None:
        fused = fuse_semantic_v3(
            image_key="withheld",
            extract_wrapper=self._extract(),
            analyze_artifact=self._analyze("unknown", 0.0, "unknown"),
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(public="uncertain", candidate="standing", recovery=True),
        )
        modifiers = fused["canonical"]["pose_modifiers"]
        self.assertEqual(fused["canonical"]["posture"]["value"], "unknown")
        self.assertEqual(modifiers["authority_state"], "withheld")
        self.assertIsNone(modifiers["pose_family"])
        self.assertEqual(modifiers["reconstruction_pose_family"], "standing")
        self.assertTrue(modifiers["provenance_only"])

    def test_rejected_proximal_chain_downgrades_whole_hand_specificity(self) -> None:
        analyze = self._analyze("unknown", 0.0, "unknown")
        analyze["analyze"]["interactions"] = [{
            "type": "gesture",
            "actor_part": "right_hand",
            "actor_ownership": "target",
            "target_ref": "target_subject",
            "target_text": None,
            "interpretation": "hand under chin",
            "confidence": 0.9,
            "limitations": [],
        }]
        analyze["analyze"]["ownership_assessments"] = [{
            "part": "right_hand",
            "ownership": "target",
            "confidence": 0.9,
            "evidence": ["claimed chain"],
            "limitations": [],
        }]
        fused = fuse_semantic_v3(
            image_key="00018",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(public="uncertain", candidate="standing", recovery=True, hand_reject=True),
        )
        interaction = fused["canonical"]["interactions"][0]
        ownership = fused["canonical"]["ownership_assessments"][0]
        self.assertEqual(interaction["actor_ownership"], "unknown")
        self.assertEqual(interaction["actor_part"], "distal_hand_or_finger_fragment")
        self.assertEqual(ownership["ownership"], "unknown")
        self.assertEqual(ownership["part"], "distal_hand_or_finger_fragment")

    def test_unresolved_support_keeps_broad_relation_but_drops_ungrounded_target_description(self) -> None:
        analyze = self._analyze("seated", 0.9, "supported")
        analyze["analyze"]["support_context"] = [{
            "subject_relation": "seated_on",
            "target_ref": None,
            "target_description": "surface under lap",
            "target_status": "unresolved",
            "confidence": 0.65,
            "evidence": ["knees bent"],
            "limitations": ["exact seat not visible"],
        }]
        fused = fuse_semantic_v3(
            image_key="seated",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(public="sitting", candidate="sitting", recovery=False),
        )
        support = fused["canonical"]["support_context"][0]
        self.assertEqual(support["subject_relation"], "seated_on")
        self.assertEqual(support["target_status"], "unresolved")
        self.assertIsNone(support["target_ref"])
        self.assertIsNone(support["target_description"])


if __name__ == "__main__":
    unittest.main()
