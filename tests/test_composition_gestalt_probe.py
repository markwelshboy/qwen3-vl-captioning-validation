from __future__ import annotations

import json
import unittest
from pathlib import Path

from qwen_caption_validate.composition_gestalt_probe import _matches
from qwen_caption_validate.runner import validate_analysis


ROOT = Path(__file__).resolve().parents[1]


class CompositionGestaltProbeTests(unittest.TestCase):
    def test_schema_accepts_minimal_valid_payload(self) -> None:
        schema = json.loads((ROOT / "schemas" / "composition_gestalt_v1.schema.json").read_text(encoding="utf-8"))
        payload = {
            "schema_version": "composition-gestalt-1.0",
            "camera": {
                "elevation": "low",
                "pitch": "upward",
                "confidence": 0.9,
                "evidence": ["camera is visibly below the subject"],
                "counterevidence": [],
            },
            "capture": {
                "mode": "handheld_selfie",
                "confidence": 0.9,
                "selfie_holding_hand": "right",
                "holding_hand_confidence": 0.8,
                "device_visibility": "implied",
                "evidence": ["one arm leads toward the lens"],
            },
            "framing": {
                "shot_scale": "medium_close_up",
                "visible_extent": "head and upper torso",
                "confidence": 0.95,
            },
            "support_context": [
                {
                    "subject_relation": "reclining_on",
                    "target": "bed",
                    "target_description": "white bed",
                    "evidence_status": "observed",
                    "confidence": 0.95,
                }
            ],
            "foreground_relations": [
                {
                    "type": "camera_between_legs",
                    "description": "camera looks between raised legs",
                    "evidence_status": "observed",
                    "confidence": 0.9,
                }
            ],
            "salient_body_configuration": [
                {
                    "description": "one leg lifted",
                    "evidence_status": "observed",
                    "confidence": 0.95,
                }
            ],
            "composition_summary": "reclining on a bed, viewed from low between raised legs",
            "uncertainties": [],
        }
        self.assertEqual(validate_analysis(payload, schema), [])

    def test_schema_rejects_frame_side_as_anatomical_hand(self) -> None:
        schema = json.loads((ROOT / "schemas" / "composition_gestalt_v1.schema.json").read_text(encoding="utf-8"))
        payload = {
            "schema_version": "composition-gestalt-1.0",
            "camera": {"elevation": "unknown", "pitch": "unknown", "confidence": 0.0, "evidence": [], "counterevidence": []},
            "capture": {
                "mode": "handheld_selfie",
                "confidence": 0.8,
                "selfie_holding_hand": "image_left",
                "holding_hand_confidence": 0.8,
                "device_visibility": "implied",
                "evidence": [],
            },
            "framing": {"shot_scale": "unknown", "visible_extent": None, "confidence": 0.0},
            "support_context": [],
            "foreground_relations": [],
            "salient_body_configuration": [],
            "composition_summary": None,
            "uncertainties": [],
        }
        self.assertTrue(validate_analysis(payload, schema))

    def test_only_matching_supports_substrings_and_globs(self) -> None:
        path = Path("poseblind-02_00017.png")
        self.assertTrue(_matches(path, ["00017"]))
        self.assertTrue(_matches(path, ["poseblind-02_*.png"]))
        self.assertFalse(_matches(path, ["00003"]))

    def test_prompt_forbids_bare_frame_to_anatomical_conversion(self) -> None:
        prompt = (ROOT / "prompts" / "composition_gestalt_v1.txt").read_text(encoding="utf-8")
        self.assertIn("PERSON'S anatomical left/right", prompt)
        self.assertIn("Never use bare", prompt)
        self.assertIn("supporting bed", prompt)


if __name__ == "__main__":
    unittest.main()
