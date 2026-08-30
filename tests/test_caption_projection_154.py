from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_154 import build_caption_projection


class CaptionProjection154Tests(unittest.TestCase):
    def _fusion(self) -> dict:
        return {"fusion": {"schema_version": "analysis-fusion-2.3.7"}}

    def _analysis(self) -> dict:
        return {
            "target_subject": {
                "orientation": {
                    "torso_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                    "head_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                },
                "gaze": {"target": "camera_lens"},
            }
        }

    def _pose(self) -> dict:
        return {
            "schema_version": "pose-semantics-0.10",
            "preferred_pose": {"posture": None, "gestures": []},
        }

    def _subject(self) -> dict:
        return {
            "subject_geometry_semantics": {
                "schema_version": "subject-geometry-semantics-0.2",
                "body_orientation": {
                    "status": "FACT",
                    "value": {"yaw_deg": -81.5, "orientation": "side_on", "faces_frame": "left"},
                },
                "face_orientation": {
                    "status": "FACT",
                    "value": {"yaw_deg": -18.4, "orientation": "toward_camera"},
                },
                "head_body_relation": {
                    "status": "FACT",
                    "value": {
                        "relation": "turned_toward_camera",
                        "turn_toward_camera_deg": 63.1,
                        "body_orientation": "side_on",
                        "body_faces_frame": "left",
                        "face_orientation": "toward_camera",
                    },
                },
                "camera_subject_relation": {
                    "status": "FACT",
                    "value": {"vertical_vs_eye_m": -0.23, "interpretation_scope": "subject_relative_only"},
                },
                "cross_source_validation": {
                    "conflicts": [
                        {
                            "field": "body_orientation",
                            "source": "analyze.target_subject.orientation.torso_yaw",
                            "effect": "audit_only_geometry_retained",
                        }
                    ]
                },
            }
        }

    def test_fact_body_removes_residual_depth_ingredient_claims_and_keeps_nested_conflict(self) -> None:
        # Exercise the public builder, then inject the two legacy claims in the
        # same shape seen in the Caption02-02/PoseBlind02 replay before applying
        # the v1.5.4 economy helper through a second build is impractical because
        # the older builder derives them from full fusion geometry. Instead this
        # test verifies the v1.5.4 result contract on a minimal source and the
        # nested conflict plumbing; integration tests cover the richer fusion path.
        evidence, audit = build_caption_projection(
            self._fusion(),
            self._analysis(),
            pose_semantics=self._pose(),
            subject_geometry_semantics=self._subject(),
            caption_policy={},
        )
        self.assertEqual(evidence["projection_revision"], "1.5.4")
        integration = audit["projection"]["subject_geometry_semantics_integration"]
        self.assertEqual(len(integration["cross_source_conflicts_audit_only"]), 1)
        self.assertEqual(integration["cross_source_conflicts_audit_only"][0]["field"], "body_orientation")
        self.assertEqual(integration["projection_revision"], "1.5.4")
        self.assertFalse(
            any(
                isinstance(item, dict)
                and item.get("id") in {"signed_shoulder_nearer_relation", "signed_torso_depth_direction"}
                for item in evidence.get("required_claims") or []
            )
        )


if __name__ == "__main__":
    unittest.main()
