from __future__ import annotations

import unittest

from qwen_caption_validate.subject_geometry_semantics_02 import (
    CANDIDATE,
    FACT,
    WITHHELD,
    build_subject_geometry_semantics,
)


class SubjectGeometrySemantics02Tests(unittest.TestCase):
    def _diagnostic(
        self,
        *,
        body_orientation="side_on",
        body_yaw=80.0,
        faces_frame="right",
        face_orientation="toward_camera",
        face_yaw=8.0,
        turn=72.0,
        compound=True,
        body_gate=True,
        face_gate=True,
    ) -> dict:
        return {
            "schema_version": "sam3d-subject-geometry-diagnostic-0.2",
            "body_camera_relation": {
                "yaw_deg": body_yaw,
                "orientation_band": body_orientation,
                "faces_frame": faces_frame,
            },
            "face_camera_relation": {
                "yaw_deg": face_yaw,
                "orientation_band": face_orientation,
                "head_turn_toward_camera_deg": turn,
            },
            "camera_relative_subject": {
                "vertical_vs_eye": 0.22,
                "vertical_vs_shoulders": 0.47,
                "side": "subject_right",
                "optical_axis_pitch_deg": -23.0,
                "optical_axis_yaw_deg": 27.0,
                "camera_pose_pattern": "camera_above_subject_aimed_down",
            },
            "body_frame_landmarks": {
                "canonical_lateral_axis_check": {"plus_x_is_subject_left": True}
            },
            "dwpose_visibility_gate": {
                "body_yaw_observation_gate": body_gate,
                "face_yaw_observation_gate": face_gate,
            },
            "compound_pose_hint": {
                "body_orientation": body_orientation,
                "body_faces_frame": faces_frame,
                "head_relation": "turned_toward_camera",
                "face_orientation": face_orientation,
                "head_turn_toward_camera_deg": turn,
            } if compound else None,
        }

    def _analysis(
        self,
        *,
        torso_direction="frontal",
        torso_magnitude="none",
        head_direction="frontal",
        head_magnitude="none",
        gaze="camera_lens",
    ) -> dict:
        return {
            "target_subject": {
                "orientation": {
                    "torso_yaw": {
                        "direction": torso_direction,
                        "magnitude": torso_magnitude,
                        "confidence": 0.9,
                    },
                    "head_yaw": {
                        "direction": head_direction,
                        "magnitude": head_magnitude,
                        "confidence": 0.9,
                    },
                },
                "gaze": {"target": gaze},
            }
        }

    def _fusion(self, *, upper=None, provenance_risk=None) -> dict:
        fusion = {
            "sam3d_geometry_audit": {
                "target_provenance": {"context_risk": provenance_risk}
            }
        }
        if upper == "agree":
            fusion["qualified_upper_torso_depth_relation"] = {
                "authority": "qualified_visible_shoulder_depth_rotation",
                "source_magnitude_deg": 80.0,
                "relation": "upper torso strongly turned in depth, near side-on rather than square-on to the camera",
            }
        elif upper == "disagree":
            fusion["qualified_upper_torso_depth_relation"] = {
                "authority": "qualified_visible_shoulder_depth_rotation",
                "source_magnitude_deg": 10.0,
                "relation": "upper torso nearly square-on to the camera",
            }
        return fusion

    def test_analyze_frontal_does_not_veto_visibility_gated_side_on_geometry(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(),
            self._analysis(torso_direction="frontal", torso_magnitude="none"),
            self._fusion(),
        )
        self.assertEqual(out["body_orientation"]["status"], FACT)
        self.assertEqual(out["body_orientation"]["value"]["orientation"], "side_on")
        conflicts = out["cross_source_validation"]["conflicts"]
        self.assertTrue(any(item["field"] == "body_orientation" for item in conflicts))

    def test_missing_fusion_does_not_block_calibrated_body_fact(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(body_orientation="three_quarter", body_yaw=53.0, faces_frame="right", compound=False),
            self._analysis(torso_direction="anatomical_right", torso_magnitude="slight", gaze="off_camera"),
            None,
        )
        self.assertEqual(out["body_orientation"]["status"], FACT)
        self.assertEqual(out["body_orientation"]["value"]["orientation"], "three_quarter")

    def test_independent_fusion_geometry_conflict_demotes_body(self) -> None:
        diagnostic = self._diagnostic(body_orientation="frontal", body_yaw=4.0, faces_frame=None, compound=False)
        out = build_subject_geometry_semantics(
            diagnostic,
            self._analysis(torso_direction="frontal", torso_magnitude="none"),
            self._fusion(upper="agree"),
        )
        # For a frontal SAM3D band, a strong near-side-on Fusion relation is an
        # independent geometric contradiction according to the v0.1 validator.
        self.assertEqual(out["body_orientation"]["status"], CANDIDATE)

    def test_side_on_body_and_camera_facing_face_compound_to_fact_even_if_analyze_body_is_wrong(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(),
            self._analysis(torso_direction="frontal", torso_magnitude="slight"),
            self._fusion(upper="agree"),
        )
        self.assertEqual(out["body_orientation"]["status"], FACT)
        self.assertEqual(out["face_orientation"]["status"], FACT)
        self.assertEqual(out["head_body_relation"]["status"], FACT)
        self.assertEqual(out["compound_pose_fact"]["body_orientation"], "side_on")
        self.assertEqual(out["compound_pose_fact"]["head_relation"], "turned_toward_camera")

    def test_subject_relative_camera_is_fact_but_not_caption_photographic_elevation(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(compound=False),
            self._analysis(),
            self._fusion(),
        )
        camera = out["camera_subject_relation"]
        self.assertEqual(camera["status"], FACT)
        self.assertEqual(camera["value"]["interpretation_scope"], "subject_relative_only")
        self.assertEqual(camera["value"]["camera_pose_pattern"], "camera_above_subject_aimed_down")
        self.assertIsNone(out["preferred_orientation"]["camera_subject_relation"])
        self.assertIsNotNone(out["preferred_subject_geometry"]["camera_subject_relation"])

    def test_missing_observation_gate_withholds_reconstruction(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(body_gate=False),
            self._analysis(),
            self._fusion(),
        )
        self.assertEqual(out["body_orientation"]["status"], WITHHELD)
        self.assertEqual(out["camera_subject_relation"]["status"], WITHHELD)

    def test_provenance_review_withholds_reconstruction_semantics(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(),
            self._analysis(),
            self._fusion(provenance_risk="requires_review"),
        )
        self.assertEqual(out["body_orientation"]["status"], WITHHELD)
        self.assertEqual(out["face_orientation"]["status"], WITHHELD)
        self.assertEqual(out["camera_subject_relation"]["status"], WITHHELD)


if __name__ == "__main__":
    unittest.main()
