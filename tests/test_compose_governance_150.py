from __future__ import annotations

import unittest
from unittest.mock import patch

from qwen_caption_validate.caption_projection_150 import (
    _install_semantic_pose,
    build_caption_projection,
    lint_caption,
)
from qwen_caption_validate.compose_governance_150 import (
    _GOVERNANCE_ADDENDUM,
    _pose_semantics_ok,
)


class ComposeGovernance150Tests(unittest.TestCase):
    def _evidence(self) -> dict:
        return {
            "pose_orientation": {
                "whole_body_posture": {
                    "allowed": ["standing"],
                    "authority": "older_projection",
                    "evidence": ["old"],
                }
            },
            "required_claims": [
                {
                    "id": "whole_body_posture_standing",
                    "priority": "required",
                    "description": "subject is standing",
                },
                {
                    "id": "framing_subject_extent",
                    "priority": "required",
                    "description": "medium framing",
                },
            ],
        }

    def test_fact_posture_replaces_older_projection_authority(self) -> None:
        evidence = self._evidence()
        audit = {"projection": {"allowed": [], "blocked": []}}
        semantics = {
            "schema_version": "pose-semantics-0.10",
            "preferred_pose": {"posture": "seated", "gestures": []},
            "posture_candidate": None,
        }
        _install_semantic_pose(evidence, audit, semantics)

        posture = evidence["pose_orientation"]["whole_body_posture"]
        self.assertEqual(posture["allowed"], ["seated"])
        self.assertEqual(posture["authority"], "pose_semantics_v0.10_fact")
        ids = [item["id"] for item in evidence["required_claims"]]
        self.assertNotIn("whole_body_posture_standing", ids)
        self.assertIn("semantic_pose_posture", ids)
        self.assertIn("framing_subject_extent", ids)
        claim = next(item for item in evidence["required_claims"] if item["id"] == "semantic_pose_posture")
        self.assertEqual(claim["posture"], "seated")

    def test_candidate_posture_is_audit_only_and_clears_old_fact(self) -> None:
        evidence = self._evidence()
        audit = {"projection": {"allowed": [], "blocked": []}}
        semantics = {
            "schema_version": "pose-semantics-0.10",
            "preferred_pose": {"posture": None, "gestures": []},
            "posture_candidate": {
                "label": "seated",
                "status": "candidate",
                "model_confidence": 0.90,
                "confidence_band": "strong",
                "review_recommended": True,
                "authority": "top_down_gestalt_hypothesis_not_independently_verified",
            },
        }
        _install_semantic_pose(evidence, audit, semantics)

        self.assertEqual(evidence["pose_orientation"]["whole_body_posture"]["allowed"], [])
        self.assertIsNone(evidence["pose_orientation"]["semantic_pose"]["posture"])
        self.assertFalse(any(item["id"] == "semantic_pose_posture" for item in evidence["required_claims"]))
        self.assertNotIn("posture_candidate", evidence["pose_orientation"])
        integration = audit["projection"]["pose_semantics_integration"]
        self.assertEqual(integration["posture_candidate_audit_only"]["label"], "seated")
        self.assertFalse(integration["candidate_exposed_to_caption_evidence"])

    def test_caption_preferred_gesture_becomes_required_semantic_claim(self) -> None:
        evidence = self._evidence()
        audit = {"projection": {"allowed": [], "blocked": []}}
        semantics = {
            "schema_version": "pose-semantics-0.10",
            "preferred_pose": {
                "posture": "seated",
                "gestures": ["left forearm resting on the table"],
            },
        }
        _install_semantic_pose(evidence, audit, semantics)
        gestures = evidence["pose_orientation"]["semantic_pose"]["gestures"]
        self.assertEqual(gestures, ["left forearm resting on the table"])
        claim = next(
            item
            for item in evidence["required_claims"]
            if str(item.get("id") or "").startswith("semantic_pose_gesture_")
        )
        self.assertEqual(claim["description"], "left forearm resting on the table")
        self.assertIn("Do not also serialize lower-level", claim["instruction"])

    def test_build_revision_150_calls_semantic_overlay(self) -> None:
        base_evidence = self._evidence()
        base_audit = {"projection": {"allowed": [], "blocked": []}}
        semantics = {
            "schema_version": "pose-semantics-0.10",
            "preferred_pose": {"posture": "reclining", "gestures": []},
        }
        with patch(
            "qwen_caption_validate.caption_projection_150._build_143",
            return_value=(base_evidence, base_audit),
        ):
            evidence, audit = build_caption_projection({}, {}, pose_semantics=semantics)
        self.assertEqual(evidence["projection_revision"], "1.5.0")
        self.assertEqual(evidence["pose_orientation"]["semantic_pose"]["posture"], "reclining")
        self.assertEqual(audit["projection"]["schema_version"], "caption-projection-audit-1.5.0")

    def test_lint_requires_fact_posture_but_not_candidate(self) -> None:
        evidence = {
            "pose_orientation": {
                "semantic_pose": {"posture": "seated", "gestures": []}
            },
            "required_claims": [],
        }
        with patch(
            "qwen_caption_validate.caption_projection_150._lint_143",
            return_value={"violations": [], "warnings": [], "violation_count": 0, "warning_count": 0, "passed": True},
        ):
            result = lint_caption("The subject looks toward the camera.", evidence)
        self.assertEqual(result["warning_count"], 1)
        self.assertEqual(result["warnings"][0]["claim_id"], "semantic_pose_posture")

        evidence["pose_orientation"]["semantic_pose"]["posture"] = None
        with patch(
            "qwen_caption_validate.caption_projection_150._lint_143",
            return_value={"violations": [], "warnings": [], "violation_count": 0, "warning_count": 0, "passed": True},
        ):
            result = lint_caption("The subject looks toward the camera.", evidence)
        self.assertEqual(result["warning_count"], 0)

    def test_runner_requires_exact_frozen_pose_semantics_schema(self) -> None:
        self.assertTrue(_pose_semantics_ok({"schema_version": "pose-semantics-0.10"}))
        self.assertFalse(_pose_semantics_ok({"schema_version": "pose-semantics-0.9"}))
        self.assertFalse(_pose_semantics_ok(None))

    def test_prompt_makes_fact_candidate_boundary_and_semantic_economy_explicit(self) -> None:
        self.assertIn("qualified FACT", _GOVERNANCE_ADDENDUM)
        self.assertIn("candidate posture is deliberately absent", _GOVERNANCE_ADDENDUM)
        self.assertIn("Semantic economy is mandatory", _GOVERNANCE_ADDENDUM)
        self.assertIn("do not repeat knee angles", _GOVERNANCE_ADDENDUM)


if __name__ == "__main__":
    unittest.main()
