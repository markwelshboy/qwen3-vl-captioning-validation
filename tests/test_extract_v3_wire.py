from __future__ import annotations

import json
import unittest

from qwen_caption_validate.extract_v3 import DEFAULT_SCHEMA
from qwen_caption_validate.extract_v3_contract import audit_extract_contract
from qwen_caption_validate.extract_v3_wire import DEFAULT_WIRE_SCHEMA
from qwen_caption_validate.extract_v3_wire_contract import expand_extract_wire
from qwen_caption_validate.runner import validate_analysis


class ExtractV3WireTests(unittest.TestCase):
    def _wire(self) -> dict:
        landmark = ["visible", "h", "region represented"]
        return {
            "v": "x3w1",
            "o": "Person indoors with a red car visible through a window.",
            "f": ["medium_close_up", "head through upper torso", "large", ["lower torso exits crop"]],
            "s": {
                "cl": [["shirt", ["dark"], "center", "partial", "h"]],
                "ac": [["watch_strap", ["white"], "lower center", "partial", "h"]],
                "hs": [],
                "ex": ["slight smile"],
                "bp": [],
                "lm": {
                    "hd": list(landmark), "ls": list(landmark), "rs": list(landmark),
                    "lh": ["not_visible", "h", "below crop"],
                    "rh": ["not_visible", "h", "below crop"],
                    "lk": ["not_visible", "h", "below crop"],
                    "rk": ["not_visible", "h", "below crop"],
                    "la": ["not_visible", "h", "below crop"],
                    "ra": ["not_visible", "h", "below crop"],
                },
                "or": [["shoulders depth-staggered"], ["face more frontal than torso"], ["torso axis upright"]],
                "g": ["camera_lens", "image_center", "h", ["eyes near lens"]],
                "ix": [],
            },
            "e": [
                ["e1", "vehicle", "car", ["red", "blurred"], "blurred", "background right", "through_opening", "h"],
                ["e2", "architecture", "window", [], "partial", "right", "background", "h"],
            ],
            "r": [["e1", "visible_through", "e2", None, "observed", "h", ["window surrounds car"]]],
            "sc": {
                "env": ["indoor", "h", ["interior wall and window"], []],
                "ill": ["mixed", "mixed", "medium", []],
                "bg": ["low", "medium", "low", False, "medium", False, ["window frame"]],
                "br": [["window and exterior", "behind_subject", "right", "observed", "h"]],
                "nr": [],
            },
            "co": ["dominant", [], ["subject fills center"]],
            "h": {
                "p": ["unknown", "l", [], ["lower body cropped"]],
                "to": ["three_quarter", "left", "h", ["shoulder depth asymmetry"], []],
                "ho": ["frontal", "neutral", "neutral", "h", ["face near camera-facing"], []],
                "hb": ["turned_toward_camera", "h", ["face more frontal than torso"], []],
                "cam": ["unknown", "unknown", "l", [], ["height ambiguous"]],
                "cap": ["external_camera", "m", []],
                "sup": [],
                "act": [],
            },
            "u": ["lower-body posture unresolved"],
        }

    def test_wire_schema_accepts_representative_record(self) -> None:
        schema = json.loads(DEFAULT_WIRE_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(self._wire(), schema), [])

    def test_wire_expands_to_valid_canonical_extract(self) -> None:
        canonical, metadata = expand_extract_wire(self._wire())
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(canonical, schema), [])
        self.assertEqual(metadata["warnings"], [])
        self.assertEqual(canonical["entities"][0]["id"], "entity_01")
        self.assertEqual(canonical["target_subject"]["transient_appearance"]["accessories"][0]["descriptors"], ["white"])

    def test_expanded_extract_satisfies_both_reconstruction_contracts(self) -> None:
        canonical, _ = expand_extract_wire(self._wire())
        audit = audit_extract_contract(canonical)
        self.assertTrue(audit["analyze_reconstructable"])
        self.assertTrue(audit["gestalt_reconstructable"])
        self.assertEqual(audit["analyze_missing_paths"], [])
        self.assertEqual(audit["gestalt_missing_paths"], [])

    def test_confidence_bands_are_deterministic_not_model_precision(self) -> None:
        canonical, metadata = expand_extract_wire(self._wire())
        self.assertEqual(metadata["confidence_band_mapping"], {"h": 0.9, "m": 0.65, "l": 0.35, "u": 0.0})
        self.assertEqual(canonical["entities"][0]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
