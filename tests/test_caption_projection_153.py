from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_153 import _install_subject_geometry_orientation


class CaptionProjection153Tests(unittest.TestCase):
    def _evidence(self) -> dict:
        return {
            "pose_orientation": {
                "semantic_orientation": {
                    "torso_yaw": {"direction": "frontal", "magnitude": "slight"},
                    "head_yaw": {"direction": "frontal", "magnitude": "none"},
                    "torso_pitch": {"direction": "neutral"},
                },
                "upper_torso_depth_relation": {
                    "relation": "upper torso strongly turned in depth",
                    "source_magnitude_deg": 80.0,
                },
                "head_torso_relation": {
                    "relation": "head turned substantially toward the camera relative to the torso"
                },
            },
            "required_claims": [
                {"id": "upper_torso_side_on_relation", "description": "legacy torso relation"},
                {"id": "head_turn_toward_camera_relative_torso", "description": "legacy head relation"},
                {"id": "framing_subject_extent", "description": "framing remains"},
            ],
        }

    def _audit(self) -> dict:
        return {"projection": {"allowed": [], "blocked": [], "notes": []}}

    def _semantics(self, *, compound=True, body_status="FACT", face_orientation="toward_camera") -> dict:
        return {
            "subject_geometry_semantics": {
                "schema_version": "subject-geometry-semantics-0.2",
                "body_orientation": {
                    "status": body_status,
                    "value": {
                        "yaw_deg": -81.5,
                        "orientation": "side_on",
                        "faces_frame": "left",
                    },
                },
                "face_orientation": {
                    "status": "FACT",
                    "value": {"yaw_deg": -18.4, "orientation": face_orientation},
                },
                "head_body_relation": {
                    "status": "FACT" if compound else "WITHHELD",
                    "value": {
                        "relation": "turned_toward_camera",
                        "turn_toward_camera_deg": 63.1,
                        "body_orientation": "side_on",
                        "body_faces_frame": "left",
                        "face_orientation": face_orientation,
                    } if compound else None,
                },
                "camera_subject_relation": {
                    "status": "FACT",
                    "value": {
                        "vertical_vs_eye_m": -0.23,
                        "side": "subject_left",
                        "optical_axis_pitch_deg": 9.9,
                    },
                },
                "cross_source_conflicts": [
                    {"field": "body_orientation", "source": "Analyze", "effect": "audit_only_geometry_retained"}
                ],
            }
        }

    def test_compound_fact_replaces_legacy_component_orientation(self) -> None:
        evidence = self._evidence()
        audit = self._audit()
        _install_subject_geometry_orientation(evidence, audit, self._semantics(compound=True))

        pose = evidence["pose_orientation"]
        self.assertNotIn("torso_yaw", pose["semantic_orientation"])
        self.assertNotIn("head_yaw", pose["semantic_orientation"])
        self.assertIn("torso_pitch", pose["semantic_orientation"])
        self.assertNotIn("upper_torso_depth_relation", pose)
        self.assertNotIn("head_torso_relation", pose)

        orientation = pose["subject_geometry_orientation"]
        self.assertEqual(orientation["body_orientation"], {"orientation": "side_on", "faces_frame": "left"})
        self.assertEqual(orientation["face_orientation"], {"orientation": "toward_camera"})
        self.assertEqual(orientation["head_body_relation"]["relation"], "turned_toward_camera")
        self.assertNotIn("yaw_deg", orientation["body_orientation"])

        claim_ids = [item["id"] for item in evidence["required_claims"]]
        self.assertIn("subject_geometry_compound_orientation", claim_ids)
        self.assertNotIn("upper_torso_side_on_relation", claim_ids)
        self.assertNotIn("head_turn_toward_camera_relative_torso", claim_ids)
        self.assertIn("framing_subject_extent", claim_ids)

        integration = audit["projection"]["subject_geometry_semantics_integration"]
        self.assertEqual(integration["fact_source"]["body_orientation"]["yaw_deg"], -81.5)
        self.assertEqual(len(integration["cross_source_conflicts_audit_only"]), 1)
        self.assertEqual(integration["camera_subject_relation_audit_only"]["status"], "FACT")

    def test_noncompound_body_fact_gets_one_body_claim(self) -> None:
        evidence = self._evidence()
        audit = self._audit()
        _install_subject_geometry_orientation(evidence, audit, self._semantics(compound=False))
        claim_ids = [item["id"] for item in evidence["required_claims"]]
        self.assertIn("subject_geometry_body_orientation", claim_ids)
        self.assertNotIn("subject_geometry_compound_orientation", claim_ids)
        # A camera-facing face is useful evidence but need not duplicate gaze as a required clause.
        self.assertNotIn("subject_geometry_face_orientation", claim_ids)

    def test_profile_face_fact_is_required_when_not_compound(self) -> None:
        evidence = self._evidence()
        audit = self._audit()
        _install_subject_geometry_orientation(
            evidence,
            audit,
            self._semantics(compound=False, face_orientation="profile"),
        )
        claim_ids = [item["id"] for item in evidence["required_claims"]]
        self.assertIn("subject_geometry_body_orientation", claim_ids)
        self.assertIn("subject_geometry_face_orientation", claim_ids)

    def test_candidate_body_does_not_replace_legacy_body_authority(self) -> None:
        evidence = self._evidence()
        audit = self._audit()
        semantics = self._semantics(compound=False, body_status="CANDIDATE")
        _install_subject_geometry_orientation(evidence, audit, semantics)

        pose = evidence["pose_orientation"]
        self.assertIn("torso_yaw", pose["semantic_orientation"])
        self.assertIn("upper_torso_depth_relation", pose)
        claim_ids = [item["id"] for item in evidence["required_claims"]]
        self.assertIn("upper_torso_side_on_relation", claim_ids)
        self.assertNotIn("subject_geometry_body_orientation", claim_ids)


if __name__ == "__main__":
    unittest.main()
