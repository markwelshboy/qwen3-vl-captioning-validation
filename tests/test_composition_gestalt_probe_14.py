from __future__ import annotations

import json
import unittest

from qwen_caption_validate.composition_gestalt_probe_14 import DEFAULT_PROMPT, DEFAULT_SCHEMA
from qwen_caption_validate.runner import validate_analysis


class CompositionGestaltProbe14Tests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "schema_version": "composition-gestalt-1.4",
            "camera": {"elevation": "eye_level", "pitch": "level", "confidence": 0.8, "evidence": [], "counterevidence": []},
            "capture": {"mode": "external_camera", "confidence": 0.8, "device_visibility": "not_visible", "evidence": []},
            "subject_orientation": {
                "body_orientation": "side_on",
                "body_faces_frame": "left",
                "body_confidence": 0.95,
                "torso_evidence_quality": "strong",
                "body_evidence": ["shoulder plane strongly depth-staggered"],
                "body_counterevidence": [],
                "head_relative_body": "turned_toward_camera",
                "head_confidence": 0.95,
                "head_evidence": ["face is much more camera-facing than torso"],
            },
            "framing": {"shot_scale": "close_up", "visible_extent": "upper torso to top of head", "subject_frame_fill": "tight", "confidence": 0.95},
            "environment": {"space": "outdoor", "lighting_context": "natural_daylight", "confidence": 0.8, "evidence": [], "counterevidence": []},
            "background_regions": [],
            "support_context": [],
            "foreground_relations": [],
            "salient_body_configuration": [],
            "composition_summary": "body nearly side-on facing frame-left, with the head turned back toward the camera",
            "uncertainties": [],
        }

    def test_schema_accepts_posture_independent_orientation(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(self._payload(), schema), [])

    def test_prompt_explicitly_decouples_body_orientation_from_posture_and_face(self) -> None:
        prompt = DEFAULT_PROMPT.read_text(encoding="utf-8")
        self.assertIn("BODY ORIENTATION IS INDEPENDENT OF POSTURE", prompt)
        self.assertIn("JUDGE THE TORSO FIRST, FACE SECOND", prompt)
        self.assertIn("camera-facing face must NOT drag a side-on torso toward frontal", prompt)
        self.assertIn("body_faces_frame", prompt)
        self.assertIn("head_relative_body", prompt)


if __name__ == "__main__":
    unittest.main()
