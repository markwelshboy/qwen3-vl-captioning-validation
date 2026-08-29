from __future__ import annotations

import unittest

from qwen_caption_validate.subject_geometry_semantics import (
    CANDIDATE,
    FACT,
    WITHHELD,
    build_subject_geometry_semantics,
)


class SubjectGeometrySemanticsTests(unittest.TestCase):
    def _diagnostic(
        self,
        *,
        body_orientation="three_quarter",
        body_yaw=52.0,
        faces_frame="right",
        face_orientation="toward_camera",
        face_yaw=8.0,
        turn=44.0,
        compound=True,
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
                "body_yaw_observation_gate": True,
                "face_yaw_observation_gate": True,
            },
            "compound_pose_hint": {
                "body_orientation": body_orientation,
                "body_faces_frame": faces_frame,
                "head_relation": "turned_toward_camera",
                "face_orientation": face_orientation,
                "head_turn_toward_camera_deg": turn,
            } if compound else None,
        }

    def _analysis(self, *, torso_direction="anatomical_right", torso_magnitude="strong", head_direction="frontal", head_magnitude="slight", gaze="camera_lens") -> dict:
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

    def _fusion(self, *, upper=True, head=True, provenance_risk=None) -> dict:
        fusion = {
            "sam3d_geometry_audit": {
                "target_provenance": {"context_risk": provenance_risk}
            }
        }
        if upper:
            fusion["qualified_upper_torso_depth_relation"] = {
                "authority": "qualified_visible_shoulder_depth_rotation",
                "source_magnitude_deg": 52.0,
                "relation": "upper torso strongly turned in depth rather than square-on to the camera",
            }
        if head:
            fusion["qualified_head_torso_relation"] = {
                "camera_relation": "toward_camera",
                "magnitude": "strong",
                "relation": "head turned substantially toward the camera relative to the torso",
                "authority": "qualified_visible_head_torso_rotation",
            }
        return fusion

    def test_three_quarter_body_and_camera_facing_face_become_fact(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(),
            self._analysis(),
            self._fusion(),
        )
        self.assertEqual(out["body_orientation"]["status"], FACT)
        self.assertEqual(out["face_orientation"]["status"], FACT)
        self.assertEqual(out["head_body_relation"]["status"], FACT)
        self.assertEqual(out["preferred_orientation"]["body_orientation"]["orientation"], "three_quarter")
        self.assertEqual(out["preferred_orientation"]["body_orientation"]["faces_frame"], "right")
        self.assertEqual(out["compound_pose_fact"]["head_relation"], "turned_toward_camera")

    def test_side_on_body_with_frontal_face_compounds_cleanly(self) -> None:
        diagnostic = self._diagnostic(
            body_orientation="side_on",
            body_yaw=-82.0,
            faces_frame="left",
            face_orientation="toward_camera",
            face_yaw=-18.0,
            turn=64.0,
        )
        analysis = self._analysis(torso_direction="anatomical_left", torso_magnitude="strong")
        fusion = self._fusion()
        fusion["qualified_upper_torso_depth_relation"]["source_magnitude_deg"] = 82.0
        fusion["qualified_upper_torso_depth_relation"]["relation"] = "upper torso near side-on rather than square-on to the camera"
        out = build_subject_geometry_semantics(diagnostic, analysis, fusion)
        self.assertEqual(out["body_orientation"]["status"], FACT)
        self.assertEqual(out["head_body_relation"]["status"], FACT)
        self.assertEqual(out["compound_pose_fact"]["body_orientation"], "side_on")
        self.assertEqual(out["compound_pose_fact"]["body_faces_frame"], "left")

    def test_strong_analyze_conflict_keeps_body_candidate(self) -> None:
        diagnostic = self._diagnostic(
            body_orientation="side_on",
            body_yaw=80.0,
            faces_frame="right",
        )
        analysis = self._analysis(torso_direction="frontal", torso_magnitude="none")
        out = build_subject_geometry_semantics(diagnostic, analysis, self._fusion(upper=False))
        self.assertEqual(out["body_orientation"]["status"], CANDIDATE)
        self.assertIsNone(out["preferred_orientation"]["body_orientation"])

    def test_subject_relative_camera_stays_candidate(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(),
            self._analysis(),
            self._fusion(),
        )
        camera = out["camera_subject_relation"]
        self.assertEqual(camera["status"], CANDIDATE)
        self.assertEqual(camera["value"]["camera_pose_pattern"], "camera_above_subject_aimed_down")
        self.assertIsNone(out["preferred_orientation"]["camera_subject_relation"])

    def test_provenance_review_withholds_reconstruction_semantics(self) -> None:
        out = build_subject_geometry_semantics(
            self._diagnostic(),
            self._analysis(),
            self._fusion(provenance_risk="requires_review"),
        )
        self.assertEqual(out["body_orientation"]["status"], WITHHELD)
        self.assertEqual(out["face_orientation"]["status"], WITHHELD)
        self.assertNotEqual(out["head_body_relation"]["status"], FACT)


if __name__ == "__main__":
    unittest.main()
