from __future__ import annotations

import unittest
from unittest.mock import patch

from qwen_caption_validate.caption_projection_151 import (
    _apply_structural_semantic_economy,
    build_caption_projection,
    lint_caption,
)
from qwen_caption_validate.compose_governance_151 import _GOVERNANCE_ADDENDUM


class ComposeGovernance151Tests(unittest.TestCase):
    def _semantic_evidence(self) -> dict:
        return {
            "pose_orientation": {
                "whole_body_posture": {
                    "allowed": ["reclining"],
                    "authority": "pose_semantics_v0.10_fact",
                    "evidence": ["semantic fact"],
                },
                "semantic_pose": {
                    "schema_version": "caption-semantic-pose-1.0",
                    "posture": "reclining",
                    "gestures": ["left forearm resting on the table"],
                    "authority": "pose-semantics-0.10",
                },
                "visible_subject_parts": [
                    {
                        "part": "left leg",
                        "geometry": "knee bent",
                        "support": "foot flat on bed",
                    }
                ],
                "qualified_interactions": [
                    {"type": "contact", "actor_part": "left hand", "target": "thigh"}
                ],
                "gesture_semantics": [{"type": "legacy_chin_gesture"}],
                "semantic_orientation": {"torso_yaw": {"magnitude": "moderate"}},
                "upper_torso_depth_relation": {"relation": "upper torso turned in depth"},
                "head_torso_relation": {"relation": "head turned toward camera"},
            },
            "required_claims": [
                {
                    "id": "semantic_pose_posture",
                    "priority": "required",
                    "description": "subject is reclining",
                },
                {
                    "id": "semantic_pose_gesture_1",
                    "priority": "required",
                    "description": "left forearm resting on the table",
                },
                {
                    "id": "salient_interaction_1",
                    "priority": "required",
                    "description": "left hand rests on thigh",
                },
                {
                    "id": "chin_rest_on_hand_gesture",
                    "priority": "required",
                    "description": "chin resting on hand",
                },
                {
                    "id": "framing_subject_extent",
                    "priority": "required",
                    "description": "close-up framing",
                },
                {
                    "id": "upper_torso_side_on_relation",
                    "priority": "required",
                    "description": "upper torso turned in depth",
                },
            ],
        }

    def test_structural_economy_removes_component_pose_arrays(self) -> None:
        evidence = self._semantic_evidence()
        audit = {"projection": {"allowed": [], "blocked": [], "pose_semantics_integration": {}}}
        _apply_structural_semantic_economy(evidence, audit)
        pose = evidence["pose_orientation"]
        self.assertEqual(pose["visible_subject_parts"], [])
        self.assertEqual(pose["qualified_interactions"], [])
        self.assertEqual(pose["gesture_semantics"], [])
        economy = audit["projection"]["pose_semantics_integration"]["structural_semantic_economy"]
        self.assertEqual(economy["component_pose_fields_removed"]["visible_subject_parts"], 1)
        self.assertEqual(economy["component_pose_fields_removed"]["qualified_interactions"], 1)

    def test_structural_economy_keeps_semantic_pose_and_independent_orientation(self) -> None:
        evidence = self._semantic_evidence()
        audit = {"projection": {"allowed": [], "blocked": [], "pose_semantics_integration": {}}}
        _apply_structural_semantic_economy(evidence, audit)
        pose = evidence["pose_orientation"]
        self.assertEqual(pose["semantic_pose"]["posture"], "reclining")
        self.assertEqual(pose["semantic_pose"]["gestures"], ["left forearm resting on the table"])
        self.assertIn("semantic_orientation", pose)
        self.assertIn("upper_torso_depth_relation", pose)
        self.assertIn("head_torso_relation", pose)

    def test_legacy_component_claims_are_removed_but_semantic_and_context_claims_survive(self) -> None:
        evidence = self._semantic_evidence()
        audit = {"projection": {"allowed": [], "blocked": [], "pose_semantics_integration": {}}}
        _apply_structural_semantic_economy(evidence, audit)
        ids = {item["id"] for item in evidence["required_claims"]}
        self.assertNotIn("salient_interaction_1", ids)
        self.assertNotIn("chin_rest_on_hand_gesture", ids)
        self.assertIn("semantic_pose_posture", ids)
        self.assertIn("semantic_pose_gesture_1", ids)
        self.assertIn("framing_subject_extent", ids)
        self.assertIn("upper_torso_side_on_relation", ids)

    def test_build_revision_151_applies_structural_economy(self) -> None:
        evidence = self._semantic_evidence()
        audit = {"projection": {"allowed": [], "blocked": [], "pose_semantics_integration": {}}}
        with patch(
            "qwen_caption_validate.caption_projection_151._build_150",
            return_value=(evidence, audit),
        ):
            out, out_audit = build_caption_projection({}, {}, pose_semantics={"schema_version": "pose-semantics-0.10"})
        self.assertEqual(out["projection_revision"], "1.5.1")
        self.assertEqual(out["pose_orientation"]["visible_subject_parts"], [])
        self.assertEqual(out_audit["projection"]["schema_version"], "caption-projection-audit-1.5.1")

    def test_without_pose_semantics_structural_pruning_is_not_applied(self) -> None:
        evidence = self._semantic_evidence()
        audit = {"projection": {"allowed": [], "blocked": [], "pose_semantics_integration": {}}}
        with patch(
            "qwen_caption_validate.caption_projection_151._build_150",
            return_value=(evidence, audit),
        ):
            out, _ = build_caption_projection({}, {}, pose_semantics=None)
        self.assertEqual(len(out["pose_orientation"]["visible_subject_parts"]), 1)

    def test_lint_revision_and_prompt_describe_structural_boundary(self) -> None:
        with patch(
            "qwen_caption_validate.caption_projection_151._lint_150",
            return_value={"schema_version": "old", "violations": [], "warnings": [], "passed": True},
        ):
            result = lint_caption("sH1Vx reclines.", {})
        self.assertEqual(result["schema_version"], "caption-authority-lint-1.5.1")
        self.assertIn("deliberately removed", _GOVERNANCE_ADDENDUM)
        self.assertIn("Do not expand `reclining`", _GOVERNANCE_ADDENDUM)
        self.assertIn("Do not expand `seated`", _GOVERNANCE_ADDENDUM)


if __name__ == "__main__":
    unittest.main()
