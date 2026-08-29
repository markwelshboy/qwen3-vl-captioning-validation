from __future__ import annotations

import unittest
from unittest.mock import patch

from qwen_caption_validate.caption_projection_152 import build_caption_projection, lint_caption


class ComposeGovernance152Tests(unittest.TestCase):
    def _reclining_evidence(self) -> dict:
        return {
            "pose_orientation": {
                "whole_body_posture": {
                    "allowed": ["reclining"],
                    "authority": "pose_semantics_v0.10_fact",
                },
                "semantic_pose": {
                    "posture": "reclining",
                    "gestures": [],
                    "authority": "pose-semantics-0.10",
                },
            }
        }

    def test_reclined_legacy_violation_is_normalized_for_semantic_reclining(self) -> None:
        base = {
            "schema_version": "caption-authority-lint-1.5.1",
            "passed": False,
            "violation_count": 1,
            "warning_count": 0,
            "violations": [
                {
                    "type": "unsupported_whole_body_posture",
                    "posture": "reclined",
                    "allowed_postures": ["reclining"],
                }
            ],
            "warnings": [],
        }
        with patch("qwen_caption_validate.caption_projection_152._lint_151", return_value=base):
            result = lint_caption("sH1Vx is reclining.", self._reclining_evidence())
        self.assertTrue(result["passed"])
        self.assertEqual(result["violation_count"], 0)
        self.assertEqual(result["normalized_findings"][0]["legacy_posture"], "reclined")
        self.assertEqual(result["normalized_findings"][0]["semantic_posture"], "reclining")

    def test_other_violations_are_not_suppressed(self) -> None:
        base = {
            "schema_version": "caption-authority-lint-1.5.1",
            "passed": False,
            "violation_count": 2,
            "warning_count": 0,
            "violations": [
                {
                    "type": "unsupported_whole_body_posture",
                    "posture": "reclined",
                    "allowed_postures": ["reclining"],
                },
                {
                    "type": "unqualified_anatomical_laterality",
                    "text": "right hand",
                    "side": "right",
                    "body_family": "hand",
                },
            ],
            "warnings": [],
        }
        with patch("qwen_caption_validate.caption_projection_152._lint_151", return_value=base):
            result = lint_caption("sH1Vx reclines with the right hand visible.", self._reclining_evidence())
        self.assertFalse(result["passed"])
        self.assertEqual(result["violation_count"], 1)
        self.assertEqual(result["violations"][0]["type"], "unqualified_anatomical_laterality")

    def test_reclined_violation_is_not_normalized_without_semantic_reclining_fact(self) -> None:
        evidence = {
            "pose_orientation": {
                "semantic_pose": {"posture": None, "gestures": [], "authority": "pose-semantics-0.10"}
            }
        }
        base = {
            "schema_version": "caption-authority-lint-1.5.1",
            "passed": False,
            "violation_count": 1,
            "warning_count": 0,
            "violations": [
                {"type": "unsupported_whole_body_posture", "posture": "reclined", "allowed_postures": []}
            ],
            "warnings": [],
        }
        with patch("qwen_caption_validate.caption_projection_152._lint_151", return_value=base):
            result = lint_caption("sH1Vx is reclining.", evidence)
        self.assertFalse(result["passed"])
        self.assertEqual(result["violation_count"], 1)

    def test_projection_revision_updates_without_changing_151_evidence(self) -> None:
        evidence = {"projection_revision": "1.5.1", "pose_orientation": {}}
        audit = {"projection": {"schema_version": "caption-projection-audit-1.5.1", "notes": []}}
        with patch(
            "qwen_caption_validate.caption_projection_152._build_151",
            return_value=(evidence, audit),
        ):
            out, out_audit = build_caption_projection({}, {}, pose_semantics=None)
        self.assertEqual(out["projection_revision"], "1.5.2")
        self.assertEqual(out_audit["projection"]["schema_version"], "caption-projection-audit-1.5.2")
        self.assertIn("reclined/reclining", out_audit["projection"]["notes"][-1])


if __name__ == "__main__":
    unittest.main()
