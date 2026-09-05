from __future__ import annotations

import unittest

from pydantic import ValidationError

from qwen_caption_validate.semantic_v3_kiss_fusion import fuse_kiss
from qwen_caption_validate.semantic_v3_kiss_packet import SemanticPacketKISS1


class SemanticV3KISSTests(unittest.TestCase):
    def _packet(self, **overrides):
        packet = {
            "version": "kiss1",
            "subject": [],
            "body": [],
            "interactions": [],
            "scene": [],
            "composition": [],
            "uncertainties": [],
        }
        packet.update(overrides)
        return {
            "schema_version": "semantic-v3-kiss-packet-0.1",
            "packet_schema_version": "kiss1",
            "schema_valid": True,
            "packet": packet,
        }

    def _pose(
        self,
        *,
        public_pose="uncertain",
        best_candidate="standing",
        withheld_reason=None,
        crop_support=0.2,
        modifier_family="standing",
        relations=None,
    ):
        return {
            "profile": {
                "schema_version": "sam3d-relational-pose-profile-0.16",
                "sam3d_projected_pose": {
                    "pose": public_pose,
                    "best_candidate_pose": best_candidate,
                    "crop_support": crop_support,
                    "support_class": "test",
                    "winner_margin": 0.7,
                    "assertion_authority": {
                        "selected_path": "test",
                        "selected_path_authority": crop_support,
                        "withheld_reason": withheld_reason,
                        "authority_semantics": "test",
                    },
                    "semantic_recovery": {
                        "needed": bool(withheld_reason),
                        "candidate_pose": best_candidate,
                        "reason": "test" if withheld_reason else None,
                        "candidate_score": 0.9 if withheld_reason else None,
                        "winner_margin": 0.7 if withheld_reason else None,
                        "recommended_fusion_action": "seek_semantics" if withheld_reason else None,
                    },
                    "physical_governance": {
                        "governed_best_score": 0.9,
                    },
                    "posture_modifier_diagnostic": {
                        "pose_family_for_modifier": modifier_family,
                        "lean_severity": "mild",
                        "lean_direction": "left",
                        "shoulder_line_tilt_severity": "mild",
                        "suggested_modifiers": ["leaning_left"],
                        "suggested_compound_pose_modifier": None,
                    },
                },
                "relations": relations or {},
                "policy": {},
            }
        }

    def test_packet_bins_may_all_be_empty(self):
        packet = SemanticPacketKISS1.model_validate({"v": "kiss1"})
        self.assertEqual(packet.subject, [])
        self.assertEqual(packet.body, [])
        self.assertEqual(packet.interactions, [])
        self.assertEqual(packet.scene, [])
        self.assertEqual(packet.composition, [])
        self.assertEqual(packet.uncertainties, [])

    def test_packet_output_is_hard_bounded(self):
        with self.assertRaises(ValidationError):
            SemanticPacketKISS1.model_validate({
                "v": "kiss1",
                "s": [f"fact {index}" for index in range(9)],
            })
        with self.assertRaises(ValidationError):
            SemanticPacketKISS1.model_validate({
                "v": "kiss1",
                "b": ["x" * 141],
            })

    def test_public_pose_is_structured_authority(self):
        fused = fuse_kiss(
            image_key="seated",
            packet_artifact=self._packet(body=["person beside a laptop"]),
            pose_record=self._pose(
                public_pose="sitting",
                best_candidate="sitting",
                modifier_family="sitting",
                crop_support=0.24,
            ),
        )
        posture = fused["canonical"]["posture"]
        self.assertEqual(posture["value"], "seated")
        self.assertEqual(posture["status"], "asserted")
        self.assertEqual(posture["authority"], "pose_v0.16_public")
        self.assertEqual(posture["pose_joint_authority"], 0.24)

    def test_withheld_reconstruction_stays_provenance_only(self):
        fused = fuse_kiss(
            image_key="crop",
            packet_artifact=self._packet(body=["close portrait"]),
            pose_record=self._pose(
                public_pose="uncertain",
                best_candidate="standing",
                withheld_reason="insufficient_observed_support",
                modifier_family="standing",
            ),
        )
        self.assertEqual(fused["canonical"]["posture"]["value"], "unknown")
        self.assertEqual(fused["canonical"]["posture"]["status"], "withheld")
        self.assertIsNone(fused["canonical"]["pose_modifiers"])
        self.assertEqual(
            fused["provenance"]["pose_reconstruction_provenance"]["best_candidate"],
            "standing",
        )

    def test_only_positive_governed_relations_become_canonical(self):
        fused = fuse_kiss(
            image_key="relations",
            packet_artifact=self._packet(),
            pose_record=self._pose(
                relations={
                    "head_supported_by_fist": {
                        "geometry_match": True,
                        "side": "left",
                        "support_class": "strong",
                        "crop_support": 0.8,
                    },
                    "head_supported_by_hand": {
                        "geometry_match": False,
                        "side": "right",
                        "support_class": "rejected",
                        "crop_support": 0.3,
                        "rejection_reason": "test",
                    },
                }
            ),
        )
        relations = fused["canonical"]["physical_relations"]
        self.assertIn("head_supported_by_fist", relations)
        self.assertNotIn("head_supported_by_hand", relations)

    def test_fusion_has_no_analyze_or_gestalt_dependency(self):
        packet = self._packet(
            subject=["dark shirt"],
            scene=["window behind subject"],
            composition=["tight crop"],
        )
        fused = fuse_kiss(
            image_key="simple",
            packet_artifact=packet,
            pose_record=self._pose(),
        )
        self.assertEqual(fused["canonical"]["semantic"]["subject"], ["dark shirt"])
        self.assertEqual(fused["canonical"]["semantic"]["scene"], ["window behind subject"])
        self.assertNotIn("analyze", fused)
        self.assertNotIn("gestalt", fused)
        self.assertNotIn("analyze", fused["provenance"])
        self.assertNotIn("gestalt", fused["provenance"])


if __name__ == "__main__":
    unittest.main()
