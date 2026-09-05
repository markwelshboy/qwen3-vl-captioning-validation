from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_fusion_v302 import FUSION_VERSION, fuse_semantic_v3


class SemanticV3FusionV302Tests(unittest.TestCase):
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

    def _pose(self, *, hand_reject: bool = False) -> dict:
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
                    "pose": "uncertain" if hand_reject else "sitting",
                    "best_candidate_pose": "standing" if hand_reject else "sitting",
                    "crop_support": 0.04 if hand_reject else 0.24,
                    "support_class": "reconstruction_dominant" if hand_reject else "moderately_crop_supported",
                    "winner_margin": 0.78 if hand_reject else 0.22,
                    "assertion_authority": {
                        "selected_path": "already_withheld_before_v15" if hand_reject else "posture_region_weights",
                        "selected_path_authority": 0.04 if hand_reject else 0.24,
                        "withheld_reason": "insufficient_observed_support" if hand_reject else None,
                        "authority_semantics": "pose_joint_corroboration_not_literal_visual_crop_extent",
                    },
                    "semantic_recovery": {
                        "needed": hand_reject,
                        "candidate_pose": "standing" if hand_reject else None,
                        "reason": "strong_reconstruction_candidate_but_pose_joint_authority_is_insufficient" if hand_reject else None,
                        "candidate_score": 0.93 if hand_reject else None,
                        "winner_margin": 0.78 if hand_reject else None,
                        "recommended_fusion_action": "seek_independent_analyze_or_scene_semantics_for_broad_pose" if hand_reject else None,
                    },
                    "physical_governance": {"governed_best_score": 0.93 if hand_reject else 0.55},
                    "posture_modifier_diagnostic": {
                        "pose_family_for_modifier": "standing" if hand_reject else "sitting",
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

    def _analyze(self) -> dict:
        return {
            "schema_valid": True,
            "source_extract_sha256": "abc",
            "analyze": {
                "schema_version": "semantic-analyze-3.0",
                "posture": {"value": "unknown", "confidence": 0.0, "assessment": "unknown"},
                "actions": [],
                "interactions": [],
                "ownership_assessments": [],
                "support_context": [],
            },
        }

    def test_rejected_whole_chain_evidence_is_removed_after_fragment_downgrade(self) -> None:
        analyze = self._analyze()
        analyze["analyze"]["actions"] = [{
            "value": "posing",
            "confidence": 0.9,
            "evidence_status": "contextual",
            "evidence": ["direct gaze", "hand gesture with fingers extended under chin", "smiling"],
            "limitations": [],
        }]
        analyze["analyze"]["interactions"] = [{
            "type": "gesture",
            "actor_part": "right_hand",
            "actor_ownership": "target",
            "target_ref": "target_subject",
            "target_text": None,
            "interpretation": "hand under chin",
            "evidence_status": "observed",
            "confidence": 0.9,
            "evidence": ["hand_under_chin", "index_and_middle_fingers_extended", "thumb_tucked"],
            "limitations": ["no visible wrist or forearm connection; ownership of hand is candidate but not contradicted"],
        }]
        analyze["analyze"]["ownership_assessments"] = [{
            "part": "right_hand",
            "ownership": "target",
            "confidence": 0.9,
            "evidence": ["connected_visible to right_arm", "visible_subparts include forearm and hand", "extended arm with hand near face"],
            "limitations": ["no visible wrist or watch; ownership of potential wrist accessory is uncertain"],
        }]
        fused = fuse_semantic_v3(
            image_key="00018",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(hand_reject=True),
        )
        self.assertEqual(fused["schema_version"], FUSION_VERSION)
        action = fused["canonical"]["actions"][0]
        interaction = fused["canonical"]["interactions"][0]
        ownership = fused["canonical"]["ownership_assessments"][0]
        self.assertNotIn("hand gesture with fingers extended under chin", action["evidence"])
        self.assertEqual(interaction["actor_part"], "distal_hand_or_finger_fragment")
        self.assertEqual(interaction["evidence"], ["index_and_middle_fingers_extended"])
        self.assertEqual(ownership["evidence"], [])
        self.assertTrue(any("No positive target-ownership evidence" in value for value in ownership["limitations"]))

    def test_action_evidence_denied_by_own_limitation_is_removed(self) -> None:
        analyze = self._analyze()
        analyze["analyze"]["posture"] = {"value": "seated", "confidence": 0.9, "assessment": "supported"}
        analyze["analyze"]["actions"] = [{
            "value": "drinking",
            "confidence": 0.9,
            "evidence_status": "contextual",
            "evidence": ["left hand holding cup", "cup raised to mouth", "lips touching cup"],
            "limitations": ["no direct visual of lips touching cup, inferred from hand position and cup orientation"],
        }]
        fused = fuse_semantic_v3(
            image_key="seated",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(hand_reject=False),
        )
        action = fused["canonical"]["actions"][0]
        self.assertEqual(action["value"], "drinking")
        self.assertEqual(action["confidence"], 0.9)
        self.assertEqual(action["evidence"], ["left hand holding cup", "cup raised to mouth"])
        self.assertTrue(any(item.get("type") == "remove_evidence_denied_by_item_limitation" for item in fused["authority_adjustments"]))

    def test_unrelated_evidence_is_not_removed(self) -> None:
        analyze = self._analyze()
        analyze["analyze"]["posture"] = {"value": "seated", "confidence": 0.9, "assessment": "supported"}
        analyze["analyze"]["actions"] = [{
            "value": "typing",
            "confidence": 0.9,
            "evidence_status": "contextual",
            "evidence": ["right hand on laptop keyboard", "fingers positioned for typing"],
            "limitations": ["no direct visual of finger motion or key depression, inferred from hand placement"],
        }]
        fused = fuse_semantic_v3(
            image_key="seated",
            extract_wrapper=self._extract(),
            analyze_artifact=analyze,
            gestalt_artifact=self._gestalt(),
            pose_record=self._pose(hand_reject=False),
        )
        self.assertEqual(
            fused["canonical"]["actions"][0]["evidence"],
            ["right hand on laptop keyboard", "fingers positioned for typing"],
        )


if __name__ == "__main__":
    unittest.main()
