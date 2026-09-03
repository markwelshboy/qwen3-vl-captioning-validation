from __future__ import annotations

import json
import unittest

from qwen_caption_validate.extract_v3 import DEFAULT_SCHEMA
from qwen_caption_validate.extract_v3_contract import audit_extract_contract
from qwen_caption_validate.extract_v3_models import ExtractWireV1, VisualExtractV3
from qwen_caption_validate.extract_v3_wire_contract import CONFIDENCE_BANDS, expand_extract_wire
from qwen_caption_validate.runner import validate_analysis


class ExtractV3PydanticTests(unittest.TestCase):
    def _wire_dict(self) -> dict:
        landmark_visible = {"v": "visible", "q": "h", "e": "region represented"}
        landmark_missing = {"v": "not_visible", "q": "h", "e": "outside crop"}
        return {
            "v": "x3p1",
            "o": "Person indoors with a red car visible through a window.",
            "f": {"z": "medium_close_up", "x": "head through upper torso", "c": "large", "o": ["lower torso exits crop"]},
            "s": {
                "cl": [{"c": "shirt", "d": ["dark"], "l": "center", "v": "partial", "q": "h"}],
                "ac": [{"c": "watch_strap", "d": ["white"], "l": "lower center", "v": "partial", "q": "h"}],
                "hs": ["hair falls over shoulders"],
                "ex": ["slight smile"],
                "bp": [],
                "lm": {
                    "hd": landmark_visible,
                    "ls": landmark_visible,
                    "rs": landmark_visible,
                    "lh": landmark_missing,
                    "rh": landmark_missing,
                    "lk": landmark_missing,
                    "rk": landmark_missing,
                    "la": landmark_missing,
                    "ra": landmark_missing,
                },
                "or": {"t": ["shoulders depth-staggered"], "h": ["face more frontal than torso"], "a": ["torso axis near upright"]},
                "g": {"t": "camera_lens", "d": "image_center", "q": "h", "c": ["eyes near lens"]},
                "ix": [],
            },
            "e": [
                {"i": "e1", "t": "vehicle", "c": "car", "d": ["red", "blurred"], "v": "blurred", "l": "background right", "z": "through_opening", "q": "h"},
                {"i": "e2", "t": "architecture", "c": "window", "d": [], "v": "partial", "l": "right", "z": "background", "q": "h"},
            ],
            "r": [
                {"s": "e1", "p": "visible_through", "o": "e2", "x": None, "e": "observed", "q": "h", "c": ["window surrounds car"]}
            ],
            "sc": {
                "env": {"v": "indoor", "q": "h", "c": ["interior wall and window"], "x": []},
                "ill": {"t": "mixed", "d": "mixed", "k": "medium", "o": []},
                "bg": {"t": "low", "s": "medium", "p": "low", "r": False, "l": "medium", "f": False, "o": ["window frame"]},
                "br": [{"d": "window and exterior view", "r": "behind_subject", "l": "right", "e": "observed", "q": "h"}],
                "nr": [],
            },
            "co": {"d": "dominant", "f": [], "v": ["subject fills center"]},
            "h": {
                "p": {"v": "unknown", "q": "l", "c": [], "l": ["lower body cropped"]},
                "to": {"b": "three_quarter", "f": "left", "q": "h", "c": ["shoulder depth asymmetry"], "l": []},
                "ho": {"y": "frontal", "p": "neutral", "r": "neutral", "q": "h", "c": ["face near camera-facing"], "l": []},
                "hb": {"v": "turned_toward_camera", "q": "h", "c": ["face more frontal than torso"], "l": []},
                "cam": {"e": "unknown", "p": "unknown", "q": "l", "c": [], "x": ["height relationship ambiguous"]},
                "cap": {"m": "external_camera", "q": "m", "c": []},
                "sup": [],
                "act": [],
            },
            "u": ["lower-body posture unresolved"],
        }

    def test_generated_schema_uses_aliases_and_forbids_extra_fields(self) -> None:
        schema = ExtractWireV1.model_json_schema(by_alias=True)
        self.assertIn("v", schema["properties"])
        self.assertIn("f", schema["properties"])
        self.assertNotIn("schema_version", schema["properties"])
        self.assertFalse(schema["additionalProperties"])

    def test_wire_json_round_trip_is_pydantic_valid(self) -> None:
        wire = ExtractWireV1.model_validate(self._wire_dict())
        compact = wire.model_dump_json(by_alias=True)
        restored = ExtractWireV1.model_validate_json(compact)
        self.assertEqual(restored.schema_version, "x3p1")
        self.assertEqual(restored.entities[0].class_name, "car")
        self.assertEqual(restored.subject.accessories[0].category, "watch_strap")

    def test_expansion_is_canonical_and_reconstructable(self) -> None:
        wire = ExtractWireV1.model_validate(self._wire_dict())
        canonical_model, metadata = expand_extract_wire(wire)
        self.assertIsInstance(canonical_model, VisualExtractV3)
        canonical = canonical_model.model_dump(mode="json", by_alias=True)

        legacy_schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(canonical, legacy_schema), [])
        audit = audit_extract_contract(canonical)
        self.assertTrue(audit["analyze_reconstructable"])
        self.assertTrue(audit["gestalt_reconstructable"])
        self.assertEqual(metadata["wire_schema_version"], "x3p1")
        self.assertEqual(canonical["entities"][0]["class"], "car")
        self.assertEqual(canonical["target_subject"]["transient_appearance"]["accessories"][0]["descriptors"], ["white"])

    def test_confidence_bands_are_fixed_not_fake_precision(self) -> None:
        self.assertEqual(CONFIDENCE_BANDS, {"h": 0.90, "m": 0.65, "l": 0.35, "u": 0.00})


if __name__ == "__main__":
    unittest.main()
