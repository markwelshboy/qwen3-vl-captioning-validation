from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_156 import build_caption_projection, lint_caption


class CaptionProjection156Tests(unittest.TestCase):
    def _fusion(self) -> dict:
        return {"fusion": {"schema_version": "analysis-fusion-2.3.7"}}

    def _analysis(self) -> dict:
        return {
            "target_subject": {
                "orientation": {
                    "torso_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                    "head_yaw": {"direction": "frontal", "magnitude": "slight", "confidence": 0.9},
                    "torso_pitch": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                    "torso_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                    "head_pitch": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                    "head_roll": {"direction": "neutral", "magnitude": "none", "confidence": 0.9},
                    "image_plane_body_axis": {"relation": "upright_in_image_plane", "magnitude": "none", "confidence": 0.95},
                },
                "gaze": {"target": "camera_lens"},
            }
        }

    def _pose(self) -> dict:
        return {
            "schema_version": "pose-semantics-0.10",
            "preferred_pose": {"posture": None, "gestures": []},
        }

    def _subject(self, body: str) -> dict:
        value = {"yaw_deg": 0.0, "orientation": body, "faces_frame": None}
        if body == "side_on":
            value.update(yaw_deg=-81.5, faces_frame="left")
        return {
            "subject_geometry_semantics": {
                "schema_version": "subject-geometry-semantics-0.2",
                "body_orientation": {"status": "FACT", "value": value},
                "face_orientation": {
                    "status": "FACT",
                    "value": {"yaw_deg": -15.0, "orientation": "toward_camera"},
                },
                "head_body_relation": {"status": "WITHHELD", "value": None},
                "camera_subject_relation": {
                    "status": "FACT",
                    "value": {"interpretation_scope": "subject_relative_only"},
                },
                "cross_source_validation": {"conflicts": []},
            }
        }

    def test_absent_body_orientation_rejects_invented_nonfrontal_yaw(self) -> None:
        evidence, _ = build_caption_projection(
            self._fusion(),
            self._analysis(),
            pose_semantics=self._pose(),
            subject_geometry_semantics=self._subject("frontal"),
            caption_policy={},
        )
        self.assertEqual(evidence["projection_revision"], "1.5.6")
        orientation = evidence["pose_orientation"]["subject_geometry_orientation"]
        self.assertNotIn("body_orientation", orientation)

        lint = lint_caption(
            "sH1Vx wears a cardigan. The body is slightly angled toward the camera.",
            evidence,
        )
        types = [item.get("type") for item in lint.get("violations") or []]
        self.assertIn("unsupported_body_camera_orientation", types)

    def test_framing_three_quarter_is_not_mistaken_for_body_yaw(self) -> None:
        evidence, _ = build_caption_projection(
            self._fusion(),
            self._analysis(),
            pose_semantics=self._pose(),
            subject_geometry_semantics=self._subject("frontal"),
            caption_policy={},
        )
        lint = lint_caption(
            "sH1Vx is framed in a three-quarter view from mid-thigh to the head.",
            evidence,
        )
        types = [item.get("type") for item in lint.get("violations") or []]
        self.assertNotIn("unsupported_body_camera_orientation", types)

    def test_governed_side_on_body_allows_side_on_prose(self) -> None:
        evidence, _ = build_caption_projection(
            self._fusion(),
            self._analysis(),
            pose_semantics=self._pose(),
            subject_geometry_semantics=self._subject("side_on"),
            caption_policy={},
        )
        orientation = evidence["pose_orientation"]["subject_geometry_orientation"]
        self.assertEqual(orientation["body_orientation"]["orientation"], "side_on")
        lint = lint_caption(
            "sH1Vx has the body nearly side-on to the camera, facing frame left.",
            evidence,
        )
        types = [item.get("type") for item in lint.get("violations") or []]
        self.assertNotIn("unsupported_body_camera_orientation", types)


if __name__ == "__main__":
    unittest.main()
