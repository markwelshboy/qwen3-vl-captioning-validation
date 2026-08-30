from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_157 import (
    _apply_structural_economy_157,
    lint_caption,
)


class CaptionProjection157Tests(unittest.TestCase):
    def _evidence(self) -> dict:
        return {
            "caption_policy": {},
            "pose_orientation": {
                "semantic_pose": {"posture": None, "gestures": []},
                "semantic_orientation": {},
                "subject_geometry_orientation": {
                    "schema_version": "caption-subject-geometry-orientation-1.1"
                },
                "qualified_3d_geometry": {
                    "shoulder_girdle_depth_rotation": {"magnitude_band": "low"},
                    "pelvis_depth_rotation": {"magnitude_band": "moderate"},
                    "combined_torso_depth_rotation": {"magnitude_band": "moderate"},
                },
            },
            "visibility_constraints": {
                "visible": ["left_shoulder", "right_shoulder"],
                "partial": ["left_elbow"],
                "not_visible": ["left_hip", "right_hip"],
                "unknown": ["left_ankle"],
            },
            "required_claims": [
                {"id": "shoulder_girdle_depth_rotation"},
                {"id": "pelvis_depth_rotation"},
                {"id": "combined_torso_depth_rotation"},
                {"id": "framing_subject_extent"},
            ],
        }

    def _audit(self) -> dict:
        return {
            "projection": {
                "subject_geometry_semantics_integration": {
                    "fact_source": {
                        "body_orientation": {
                            "yaw_deg": 0.0,
                            "orientation": "frontal",
                            "faces_frame": None,
                        }
                    }
                },
                "blocked": [],
            }
        }

    def test_structural_economy_quarantines_depth_components_and_positive_visibility(self) -> None:
        evidence = self._evidence()
        audit = self._audit()
        _apply_structural_economy_157(evidence, audit)

        self.assertEqual(evidence["pose_orientation"]["qualified_3d_geometry"], {})
        self.assertEqual(evidence["visibility_constraints"], {"not_visible": ["left_hip", "right_hip"]})
        ids = [item.get("id") for item in evidence["required_claims"]]
        self.assertEqual(ids, ["framing_subject_extent"])

        economy = audit["projection"]["projection_157_structural_economy"]
        self.assertIn("shoulder_girdle_depth_rotation", economy["component_depth_geometry_audit_only"])
        self.assertIn("visible", economy["positive_visibility_audit_only"])
        self.assertEqual(economy["caption_visibility_policy"], "not_visible_only")

    def test_lint_rejects_suppressed_frontal_yaw_and_component_depth_prose(self) -> None:
        evidence = self._evidence()
        audit = self._audit()
        _apply_structural_economy_157(evidence, audit)
        lint = lint_caption(
            "sH1Vx wears a cardigan. The body remains near-frontal with low shoulder girdle depth rotation.",
            evidence,
        )
        types = [item.get("type") for item in lint.get("violations") or []]
        self.assertIn("suppressed_frontal_body_orientation_resurfaced", types)
        self.assertIn("component_depth_geometry_resurfaced", types)


if __name__ == "__main__":
    unittest.main()
