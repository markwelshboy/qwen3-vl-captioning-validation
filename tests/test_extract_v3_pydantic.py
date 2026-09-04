from __future__ import annotations

import copy
import json
import unittest

from pydantic import ValidationError

from qwen_caption_validate.extract_v3 import DEFAULT_SCHEMA
from qwen_caption_validate.extract_v3_contract import audit_extract_contract
from qwen_caption_validate.extract_v3_models import VisualExtractV3
from qwen_caption_validate.extract_v3_models_runtime import ExtractWireV1Runtime as ExtractWireV1
from qwen_caption_validate.extract_v3_wire_contract import CONFIDENCE_BANDS, expand_extract_wire
from qwen_caption_validate.runner import validate_analysis


class ExtractV3PydanticTests(unittest.TestCase):
    def _wire_dict(self) -> dict:
        landmark_visible = {"v": "visible", "q": "h", "e": "region represented"}
        landmark_missing = {"v": "not_visible", "q": "h", "e": "outside crop"}
        return {
            "v": "x3p2",
            "o": "Person indoors with a red car visible through a window.",
            "f": {
                "z": "medium_close_up",
                "x": "head through upper torso",
                "c": "large",
                "o": ["lower torso exits crop"],
            },
            "s": {
                "cl": [{"c": "shirt", "d": ["dark"], "l": "center", "v": "partial", "q": "h"}],
                "ac": [{"c": "watch_strap", "d": ["white"], "l": "lower center", "v": "partial", "q": "h"}],
                "mk": [{"c": "tattoo", "d": ["dark linework"], "l": "forearm", "v": "partial", "q": "h"}],
                "hs": ["hair falls over shoulders"],
                "ex": ["slight smile"],
                "bp": [],
                "lm": {
                    "head": landmark_visible,
                    "lshoulder": landmark_visible,
                    "rshoulder": landmark_visible,
                    "lhip": landmark_missing,
                    "rhip": landmark_missing,
                    "lknee": landmark_missing,
                    "rknee": landmark_missing,
                    "lankle": landmark_missing,
                    "rankle": landmark_missing,
                },
                "or": {
                    "t": ["shoulders depth-staggered"],
                    "h": ["face more frontal than torso"],
                    "a": ["torso axis near upright"],
                },
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

        landmark_props = schema["$defs"]["WireLandmarks"]["properties"]
        self.assertIn("lhip", landmark_props)
        self.assertIn("rhip", landmark_props)
        self.assertNotIn("lh", landmark_props)
        self.assertNotIn("rh", landmark_props)

    def test_wire_json_round_trip_is_pydantic_valid(self) -> None:
        wire = ExtractWireV1.model_validate(self._wire_dict())
        compact = wire.model_dump_json(by_alias=True)
        restored = ExtractWireV1.model_validate_json(compact)
        self.assertEqual(restored.schema_version, "x3p2")
        self.assertEqual(restored.entities[0].class_name, "car")
        self.assertEqual(restored.subject.accessories[0].category, "watch_strap")
        self.assertEqual(restored.subject.markings[0].category, "tattoo")

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
        self.assertEqual(metadata["wire_schema_version"], "x3p2")
        self.assertEqual(canonical["entities"][0]["class"], "car")
        appearance = canonical["target_subject"]["transient_appearance"]
        self.assertEqual(appearance["accessories"][0]["descriptors"], ["white"])
        self.assertEqual(appearance["markings"][0]["category"], "tattoo")

    def test_relation_vocabulary_excludes_duplicate_semantic_channels(self) -> None:
        schema = ExtractWireV1.model_json_schema(by_alias=True)
        predicates = schema["$defs"]["WireRelation"]["properties"]["p"]["enum"]
        self.assertNotIn("wearing", predicates)
        self.assertNotIn("holding", predicates)
        self.assertNotIn("touching", predicates)
        self.assertNotIn("supports_candidate", predicates)
        self.assertIn("visible_through", predicates)
        self.assertIn("occludes", predicates)

    def test_dangling_entity_reference_is_rejected(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["r"][0]["o"] = "e3"
        with self.assertRaises(ValidationError):
            ExtractWireV1.model_validate(data)

    def test_duplicate_entity_id_is_rejected(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["e"][1]["i"] = "e1"
        with self.assertRaises(ValidationError):
            ExtractWireV1.model_validate(data)

    def test_non_contiguous_entity_ids_are_warning_not_failure(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["e"][1]["i"] = "e3"
        wire = ExtractWireV1.model_validate(data)
        self.assertTrue(any("non-contiguous" in warning for warning in wire.semantic_warnings()))

    def test_self_relation_is_rejected(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["r"][0]["o"] = "e1"
        with self.assertRaises(ValidationError):
            ExtractWireV1.model_validate(data)

    def test_full_body_face_dominant_is_warning_not_failure(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["f"]["z"] = "full_body"
        data["f"]["c"] = "face_dominant"
        wire = ExtractWireV1.model_validate(data)
        self.assertTrue(any("full_body" in warning for warning in wire.semantic_warnings()))

    def test_frontal_torso_frame_direction_is_warning_not_failure(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["h"]["to"]["b"] = "frontal"
        data["h"]["to"]["f"] = "left"
        wire = ExtractWireV1.model_validate(data)
        self.assertTrue(any("torso hypothesis inconsistency" in warning for warning in wire.semantic_warnings()))

    def test_confidence_bands_are_fixed_not_fake_precision(self) -> None:
        self.assertEqual(CONFIDENCE_BANDS, {"h": 0.90, "m": 0.65, "l": 0.35, "u": 0.00})


if __name__ == "__main__":
    unittest.main()
