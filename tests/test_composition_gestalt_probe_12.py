from __future__ import annotations

import json
import unittest

from qwen_caption_validate.composition_gestalt_probe_12 import DEFAULT_PROMPT, DEFAULT_SCHEMA
from qwen_caption_validate.runner import validate_analysis


class CompositionGestaltProbe12Tests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "schema_version": "composition-gestalt-1.2",
            "camera": {
                "elevation": "eye_level",
                "pitch": "level",
                "confidence": 0.8,
                "evidence": [],
                "counterevidence": [],
            },
            "capture": {
                "mode": "external_camera",
                "confidence": 0.8,
                "selfie_holding_hand": "unknown",
                "holding_hand_confidence": 0.0,
                "device_visibility": "not_visible",
                "evidence": [],
            },
            "framing": {
                "shot_scale": "close_up",
                "visible_extent": "upper torso to top of head",
                "subject_frame_fill": "tight",
                "confidence": 0.95,
            },
            "environment": {
                "space": "indoor",
                "lighting_context": "mixed",
                "confidence": 0.9,
                "evidence": ["window and interior wall"],
                "counterevidence": [],
            },
            "background_regions": [
                {
                    "description": "large gridded window",
                    "relation_to_subject": "behind_subject",
                    "frame_location": "spanning",
                    "evidence_status": "observed",
                    "confidence": 0.95,
                }
            ],
            "support_context": [],
            "foreground_relations": [],
            "salient_body_configuration": [],
            "composition_summary": "tightly framed indoor portrait with a large window behind the person",
            "uncertainties": [],
        }

    def test_schema_accepts_environment_background_and_frame_fill(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(self._payload(), schema), [])

    def test_prompt_defines_background_depth_and_image_frame_semantics(self) -> None:
        prompt = DEFAULT_PROMPT.read_text(encoding="utf-8")
        self.assertIn("relation_to_subject is DEPTH/COMPOSITION", prompt)
        self.assertIn("background frame_location is IMAGE-FRAME ONLY", prompt)
        self.assertIn("subject_frame_fill", prompt)
        self.assertIn("clearly outdoor", prompt)


if __name__ == "__main__":
    unittest.main()
