from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_155 import build_caption_projection, lint_caption


class CaptionProjection155Tests(unittest.TestCase):
    def _fusion(self) -> dict:
        return {"fusion": {"schema_version": "analysis-fusion-2.3.7"}}

    def _analysis(self, *, head_pitch: str = "neutral") -> dict:
        head_mag = "moderate" if head_pitch == "down" else "none"
        return {
            "target_subject": {
                "orientation": {
                    "torso_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                    "head_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                    "torso_pitch": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                    "torso_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                    "head_pitch": {"direction": head_pitch, "magnitude": head_mag, "confidence": 0.95},
                    "head_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                    "image_plane_body_axis": {"relation": "upright_in_image_plane", "magnitude": "none", "confidence": 0.95},
                },
                "gaze": {"target": "object"},
            }
        }

    def _pose(self, posture=None) -> dict:
        return {
            "schema_version": "pose-semantics-0.10",
            "preferred_pose": {"posture": posture, "gestures": []},
        }

    def _subject(self, *, body: str, face: str, compound: bool = False) -> dict:
        body_value = {"yaw_deg": 0.0, "orientation": body, "faces_frame": None}
        if body == "side_on":
            body_value.update(yaw_deg=-81.5, faces_frame="left")
        elif body == "three_quarter":
            body_value.update(yaw_deg=52.6, faces_frame="right")
        elif body == "slightly_angled":
            body_value.update(yaw_deg=20.6, faces_frame="right")
        face_value = {"yaw_deg": -15.4, "orientation": face}
        if face == "three_quarter":
            face_value["yaw_deg"] = 35.2
        elif face == "profile":
            face_value["yaw_deg"] = -96.0
        head = {"status": "WITHHELD", "value": None}
        if compound:
            head = {
                "status": "FACT",
                "value": {
                    "relation": "turned_toward_camera",
                    "turn_toward_camera_deg": 63.1,
                    "body_orientation": body,
                    "body_faces_frame": body_value.get("faces_frame"),
                    "face_orientation": face,
                },
            }
        return {
            "subject_geometry_semantics": {
                "schema_version": "subject-geometry-semantics-0.2",
                "body_orientation": {"status": "FACT", "value": body_value},
                "face_orientation": {"status": "FACT", "value": face_value},
                "head_body_relation": head,
                "camera_subject_relation": {"status": "FACT", "value": {"interpretation_scope": "subject_relative_only"}},
                "cross_source_validation": {"conflicts": []},
            }
        }

    def test_frontal_and_near_frontal_are_audit_only_but_head_down_survives(self) -> None:
        evidence, audit = build_caption_projection(
            self._fusion(),
            self._analysis(head_pitch="down"),
            pose_semantics=self._pose("standing"),
            subject_geometry_semantics=self._subject(body="frontal", face="toward_camera"),
            caption_policy={},
        )
        self.assertEqual(evidence["projection_revision"], "1.5.5")
        pose = evidence["pose_orientation"]
        orientation = pose["subject_geometry_orientation"]
        self.assertNotIn("body_orientation", orientation)
        self.assertNotIn("face_orientation", orientation)
        self.assertNotIn("face_yaw_orientation", orientation)
        self.assertEqual(pose["semantic_orientation"]["head_pitch"]["direction"], "down")
        self.assertNotIn("torso_pitch", pose["semantic_orientation"])
        self.assertNotIn("image_plane_body_axis", pose["semantic_orientation"])
        v155 = audit["projection"]["subject_geometry_semantics_155"]
        self.assertTrue(v155["near_frontal_face_yaw_caption_suppressed"])
        self.assertIn("frontal_body_orientation_audit_only", v155)
        self.assertIn("image_plane_body_axis", v155["neutral_semantic_orientation_audit_only"])

    def test_three_quarter_face_is_exposed_explicitly_as_yaw_only(self) -> None:
        evidence, _ = build_caption_projection(
            self._fusion(),
            self._analysis(),
            pose_semantics=self._pose("standing"),
            subject_geometry_semantics=self._subject(body="three_quarter", face="three_quarter"),
            caption_policy={},
        )
        orientation = evidence["pose_orientation"]["subject_geometry_orientation"]
        self.assertEqual(orientation["face_yaw_orientation"]["yaw_band"], "three_quarter")
        self.assertEqual(orientation["face_yaw_orientation"]["axis"], "horizontal_yaw_only")
        self.assertNotIn("face_orientation", orientation)
        ids = [item.get("id") for item in evidence.get("required_claims") or [] if isinstance(item, dict)]
        self.assertIn("subject_geometry_face_yaw_orientation", ids)
        self.assertNotIn("subject_geometry_face_orientation", ids)

    def test_compound_relation_retains_relative_yaw_without_exposing_near_frontal_face(self) -> None:
        evidence, _ = build_caption_projection(
            self._fusion(),
            self._analysis(),
            pose_semantics=self._pose(None),
            subject_geometry_semantics=self._subject(body="side_on", face="toward_camera", compound=True),
            caption_policy={},
        )
        orientation = evidence["pose_orientation"]["subject_geometry_orientation"]
        self.assertEqual(orientation["body_orientation"]["orientation"], "side_on")
        self.assertNotIn("face_yaw_orientation", orientation)
        self.assertEqual(orientation["head_body_relation"]["relation"], "turned_toward_camera")
        self.assertEqual(orientation["head_body_relation"]["face_yaw_band"], "near_frontal")
        self.assertEqual(orientation["head_body_relation"]["relation_scope"], "compensating_horizontal_yaw_relative_to_body")

    def test_lint_rejects_unlicensed_head_turn_and_upright_posture(self) -> None:
        evidence, _ = build_caption_projection(
            self._fusion(),
            self._analysis(),
            pose_semantics=self._pose(None),
            subject_geometry_semantics=self._subject(body="slightly_angled", face="toward_camera"),
            caption_policy={},
        )
        lint = lint_caption(
            "sH1Vx has an upright posture with the head turned toward the camera.",
            evidence,
        )
        types = [item.get("type") for item in lint.get("violations") or []]
        self.assertIn("unsupported_head_turn_toward_camera", types)
        self.assertIn("unsupported_whole_body_posture", types)


if __name__ == "__main__":
    unittest.main()
