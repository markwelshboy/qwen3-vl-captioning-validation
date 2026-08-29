from __future__ import annotations

import json
import unittest
from pathlib import Path

from qwen_caption_validate.runner import validate_analysis


ROOT = Path(__file__).resolve().parents[1]


class CompositionGestaltProbe11Tests(unittest.TestCase):
    def _schema(self) -> dict:
        return json.loads((ROOT / "schemas" / "composition_gestalt_v1_1.schema.json").read_text(encoding="utf-8"))

    def _payload(self) -> dict:
        return {
            "schema_version": "composition-gestalt-1.1",
            "camera": {
                "elevation": "low",
                "pitch": "upward",
                "confidence": 0.9,
                "evidence": ["lens is below the seated subject's torso"],
                "counterevidence": [],
            },
            "capture": {
                "mode": "external_camera",
                "confidence": 0.9,
                "selfie_holding_hand": "unknown",
                "holding_hand_confidence": 0.0,
                "device_visibility": "implied",
                "evidence": [],
            },
            "framing": {
                "shot_scale": "near_full_body",
                "visible_extent": "from the feet to the top of the head",
                "confidence": 0.9,
            },
            "support_context": [
                {
                    "subject_relation": "standing_on",
                    "target": "floor",
                    "target_description": "light wood floor",
                    "evidence_status": "observed",
                    "confidence": 0.95,
                }
            ],
            "foreground_relations": [],
            "salient_body_configuration": [
                {
                    "description": "bending forward with one leg lifted",
                    "evidence_status": "observed",
                    "confidence": 0.9,
                }
            ],
            "composition_summary": "low-angle view with the lens below the subject",
            "uncertainties": [],
        }

    def test_schema_accepts_standing_on_floor(self) -> None:
        self.assertEqual(validate_analysis(self._payload(), self._schema()), [])

    def test_schema_still_rejects_frame_side_as_anatomical_hand(self) -> None:
        payload = self._payload()
        payload["capture"]["mode"] = "handheld_selfie"
        payload["capture"]["selfie_holding_hand"] = "image_left"
        payload["capture"]["holding_hand_confidence"] = 0.8
        self.assertTrue(validate_analysis(payload, self._schema()))

    def test_prompt_distinguishes_lens_height_from_eye_alignment(self) -> None:
        prompt = (ROOT / "prompts" / "composition_gestalt_v1_1.txt").read_text(encoding="utf-8")
        self.assertIn("PHYSICAL LENS HEIGHT", prompt)
        self.assertIn("Do NOT call a shot eye_level merely because the face is centered", prompt)
        self.assertIn("visible_extent must name the highest and lowest visible parts coherently", prompt)
        self.assertIn("standing_on", prompt)


if __name__ == "__main__":
    unittest.main()
