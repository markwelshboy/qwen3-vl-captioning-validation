from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_pose_language import summarize_pose_language


class SemanticV3PoseLanguageTests(unittest.TestCase):
    def _record(
        self,
        *,
        public_pose="uncertain",
        best_candidate="standing",
        recovery_needed=False,
        candidate_score=0.9,
        winner_margin=0.5,
        shoulder_yaw=0.0,
        body_yaw=0.0,
        shoulder_authority=0,
        hip_authority=0,
        lean_severity="upright",
        lean_direction="direction_indeterminate",
        lean_authority=0.0,
        accepted=None,
        in_frame=None,
        extrapolated=None,
        relations=None,
    ):
        return {
            "profile": {
                "schema_version": "sam3d-relational-pose-profile-0.16",
                "sam3d_projected_pose": {
                    "pose": public_pose,
                    "best_candidate_pose": best_candidate,
                    "crop_support": 0.25,
                    "support_class": "test",
                    "winner_margin": winner_margin,
                    "assertion_authority": {
                        "withheld_reason": "insufficient_observed_support" if public_pose == "uncertain" and recovery_needed else None,
                    },
                    "semantic_recovery": {
                        "needed": recovery_needed,
                        "candidate_pose": best_candidate if recovery_needed else None,
                        "candidate_score": candidate_score if recovery_needed else None,
                        "winner_margin": winner_margin if recovery_needed else None,
                    },
                    "physical_governance": {"governed_best_score": candidate_score},
                    "posture_scores": {best_candidate: candidate_score},
                    "body_orientation_diagnostic": {
                        "available": True,
                        "camera_relative_only": True,
                        "front_vs_back_resolved": False,
                        "shoulder_yaw_from_frontal_deg": shoulder_yaw,
                        "body_yaw_from_frontal_deg": body_yaw,
                        "shoulder_authority_percent": shoulder_authority,
                        "hip_authority_percent": hip_authority,
                    },
                    "posture_modifier_diagnostic": {
                        "pose_family_for_modifier": best_candidate,
                        "lean_severity": lean_severity,
                        "lean_direction": lean_direction,
                        "torso_inclination_authority": lean_authority,
                        "shoulder_line_tilt_severity": "level",
                    },
                },
                "evidence_support": {
                    "dwpose_accepted_joint_names": accepted or [],
                    "dwpose_in_frame_accepted_joint_names": in_frame or [],
                    "projected_fit_residual": {
                        "extrapolated_dwpose_joint_names": extrapolated or [],
                    },
                },
                "relations": relations or {},
                "policy": {},
            }
        }

    def test_withheld_posture_is_conditional_not_caption_ready(self):
        result = summarize_pose_language(self._record(
            public_pose="uncertain",
            best_candidate="standing",
            recovery_needed=True,
            candidate_score=0.93,
            winner_margin=0.78,
        ))
        self.assertEqual(result["caption_ready_phrases"], [])
        self.assertEqual(result["conditional_hints"][0]["phrase"], "standing")
        self.assertEqual(result["conditional_hints"][0]["requires"], "semantic_corroboration")

    def test_cropped_shoulders_can_describe_upper_body_side_on(self):
        result = summarize_pose_language(self._record(
            recovery_needed=True,
            best_candidate="sitting",
            shoulder_yaw=82.0,
            body_yaw=82.0,
            shoulder_authority=57,
            hip_authority=0,
        ))
        self.assertIn("Upper body nearly side-on to the camera.", result["caption_ready_phrases"])
        self.assertEqual(result["components"]["orientation"]["scope"], "upper body")
        self.assertTrue(result["components"]["orientation"]["camera_relative_only"])

    def test_small_camera_relative_turn_is_omitted(self):
        result = summarize_pose_language(self._record(
            public_pose="standing",
            shoulder_yaw=18.0,
            body_yaw=18.0,
            shoulder_authority=99,
            hip_authority=58,
        ))
        self.assertEqual(result["caption_ready_phrases"], ["Standing."])
        self.assertNotIn("orientation", result["components"])

    def test_tight_crop_requires_explicit_out_of_frame_hips(self):
        result = summarize_pose_language(self._record(
            accepted=["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            in_frame=["left_shoulder", "right_shoulder"],
            extrapolated=["left_hip", "right_hip"],
        ))
        self.assertIn("Tight crop around the head and upper torso.", result["caption_ready_phrases"])

        no_crop_evidence = summarize_pose_language(self._record(
            accepted=["left_shoulder", "right_shoulder"],
            in_frame=["left_shoulder", "right_shoulder"],
            extrapolated=[],
        ))
        self.assertNotIn("framing", no_crop_evidence["components"])

    def test_positive_fist_support_becomes_natural_relation(self):
        result = summarize_pose_language(self._record(
            relations={
                "head_supported_by_hand": {
                    "geometry_match": True,
                    "side": "left",
                    "crop_support": 0.66,
                },
                "head_supported_by_fist": {
                    "geometry_match": True,
                    "side": "left",
                    "crop_support": 0.66,
                },
            }
        ))
        self.assertIn("Head resting on the left fist.", result["caption_ready_phrases"])
        self.assertNotIn("Head supported by the left hand.", result["caption_ready_phrases"])

    def test_rejected_hand_support_never_becomes_caption_language(self):
        result = summarize_pose_language(self._record(
            relations={
                "head_supported_by_hand": {
                    "geometry_match": False,
                    "side": "left",
                    "crop_support": 0.0,
                    "rejection_reason": "open_hand_distal_proximity_without_proximal_palm_wrist_support",
                },
            }
        ))
        self.assertFalse(any("hand" in phrase.lower() for phrase in result["caption_ready_phrases"]))

    def test_depth_direction_lean_stays_conditional(self):
        result = summarize_pose_language(self._record(
            public_pose="crouching",
            best_candidate="crouching",
            shoulder_yaw=55.0,
            body_yaw=55.0,
            shoulder_authority=80,
            hip_authority=80,
            lean_severity="moderate",
            lean_direction="forward_possible",
            lean_authority=0.8,
        ))
        self.assertIn(
            "Crouching with the torso strongly turned sideways to the camera while leaning noticeably.",
            result["caption_ready_phrases"],
        )
        forward = [item for item in result["conditional_hints"] if item.get("kind") == "lean_direction"]
        self.assertEqual(forward[0]["phrase"], "leaning forward")
        self.assertEqual(forward[0]["requires"], "semantic_corroboration")

    def test_caption_language_never_leaks_raw_angles(self):
        result = summarize_pose_language(self._record(
            public_pose="seated",
            shoulder_yaw=71.0,
            body_yaw=71.0,
            shoulder_authority=75,
            hip_authority=16,
        ))
        joined = " ".join(result["caption_ready_phrases"])
        self.assertNotIn("deg", joined.lower())
        self.assertNotRegex(joined, r"\b\d+(?:\.\d+)?°")


if __name__ == "__main__":
    unittest.main()
