from __future__ import annotations

import copy
import json
import unittest

from pydantic import ValidationError

from qwen_caption_validate.extract_v3 import DEFAULT_SCHEMA
from qwen_caption_validate.extract_v3_contract import audit_extract_contract
from qwen_caption_validate.extract_v3_models import VisualExtractV3
from qwen_caption_validate.extract_v3_models_x3p3 import ExtractWireX3P3Runtime
from qwen_caption_validate.extract_v3_wire_contract_x3p3 import expand_extract_wire
from qwen_caption_validate.runner import validate_analysis


class ExtractV3PydanticX3P3Tests(unittest.TestCase):
    def _wire_dict(self) -> dict:
        visible = {"v": "visible", "q": "h", "e": "region represented"}
        missing = {"v": "not_visible", "q": "h", "e": "outside crop"}
        return {
            "v": "x3p3",
            "o": "Person indoors with a window and isolated fingers crossing foreground.",
            "f": {
                "z": "medium_close_up",
                "x": "head through upper torso",
                "c": "large",
                "o": ["lower body exits crop"],
            },
            "s": {
                "cl": [
                    {"c": "shirt", "d": ["dark"], "l": "center", "v": "partial", "q": "h"}
                ],
                "ac": [
                    {"c": "watch_strap", "d": ["white"], "l": "lower center", "v": "partial", "q": "h"}
                ],
                "mk": [
                    {"c": "tattoo", "d": ["dark linework"], "l": "forearm", "v": "partial", "q": "h"}
                ],
                "hs": ["hair falls over shoulders"],
                "ex": ["neutral expression"],
                "bp": [],
                "hf": [
                    {
                        "p": "fingers",
                        "n": 2,
                        "a": "unknown",
                        "o": "unknown",
                        "k": "disconnected_in_crop",
                        "g": ["two elongated finger-like regions"],
                        "c": ["crosses foreground near neck"],
                        "l": "lower foreground",
                        "q": "h",
                    }
                ],
                "lm": {
                    "head": visible,
                    "lshoulder": visible,
                    "rshoulder": visible,
                    "lhip": missing,
                    "rhip": missing,
                    "lknee": missing,
                    "rknee": missing,
                    "lankle": missing,
                    "rankle": missing,
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
                {
                    "i": "e1",
                    "t": "architecture",
                    "c": "window",
                    "d": [],
                    "v": "partial",
                    "l": "right",
                    "z": "background",
                    "q": "h",
                }
            ],
            "r": [
                {
                    "s": "t",
                    "p": "in_front_of",
                    "o": "e1",
                    "x": None,
                    "e": "observed",
                    "q": "h",
                    "c": ["window behind subject"],
                }
            ],
            "sc": {
                "env": {"v": "indoor", "q": "h", "c": ["interior wall and window"], "x": []},
                "ill": {"t": "mixed", "d": "mixed", "k": "medium", "o": []},
                "bg": {
                    "t": "low",
                    "s": "medium",
                    "p": "low",
                    "r": False,
                    "l": "medium",
                    "f": False,
                    "o": ["window frame"],
                },
                "br": [
                    {
                        "d": "window",
                        "r": "behind_subject",
                        "l": "right",
                        "e": "observed",
                        "q": "h",
                    }
                ],
                "nr": [],
            },
            "co": {"d": "dominant", "f": ["isolated fingers cross foreground"], "v": ["subject fills center"]},
            "h": {
                "p": {"v": "unknown", "q": "l", "c": [], "l": ["lower body cropped"]},
                "to": {"b": "three_quarter", "f": "left", "q": "h", "c": ["shoulder depth asymmetry"], "l": []},
                "ho": {"y": "frontal", "p": "neutral", "r": "neutral", "q": "h", "c": ["face near camera-facing"], "l": []},
                "hb": {"v": "turned_toward_camera", "q": "h", "c": ["face more frontal than torso"], "l": []},
                "cam": {"e": "unknown", "p": "unknown", "q": "l", "c": [], "x": ["height ambiguous"]},
                "cap": {"m": "external_camera", "q": "m", "c": []},
                "sup": [],
                "act": [],
            },
            "u": ["isolated finger ownership unresolved"],
        }

    def test_schema_exposes_explicit_fragment_channel(self) -> None:
        schema = ExtractWireX3P3Runtime.model_json_schema(by_alias=True)
        self.assertEqual(schema["properties"]["v"]["const"], "x3p3")
        subject_props = schema["$defs"]["WireSubjectX3P3"]["properties"]
        self.assertIn("hf", subject_props)
        self.assertIn("bp", subject_props)

        fragment_props = schema["$defs"]["WireHumanFragmentX3P3"]["properties"]
        ownership_values = fragment_props["o"]["enum"]
        connectivity_values = fragment_props["k"]["enum"]
        self.assertNotIn("target", ownership_values)
        self.assertEqual(set(ownership_values), {"other", "unknown"})
        self.assertNotIn("connected_visible", connectivity_values)
        self.assertNotIn("connected_but_occluded", connectivity_values)

    def test_fragment_cannot_claim_target_ownership(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["s"]["hf"][0]["o"] = "target"
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate(data)

    def test_fragment_cannot_claim_connected_visible(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["s"]["hf"][0]["k"] = "connected_visible"
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate(data)

    def test_markings_are_precision_first(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["s"]["mk"][0]["q"] = "m"
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate(data)

        data = copy.deepcopy(self._wire_dict())
        data["s"]["mk"] = []
        ExtractWireX3P3Runtime.model_validate(data)

    def test_support_cannot_target_subject(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["h"]["sup"] = [
            {
                "r": "seated_on",
                "t": "t",
                "d": None,
                "e": "contextual",
                "q": "m",
                "c": ["context"],
            }
        ]
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate(data)

    def test_empty_support_is_valid_and_placeholder_support_is_not(self) -> None:
        ExtractWireX3P3Runtime.model_validate(self._wire_dict())

        data = copy.deepcopy(self._wire_dict())
        data["h"]["sup"] = [
            {
                "r": "seated_on",
                "t": None,
                "d": None,
                "e": "contextual",
                "q": "m",
                "c": [],
            }
        ]
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate(data)

    def test_posture_support_conflict_is_warning_not_failure(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["h"]["p"]["v"] = "seated"
        data["h"]["sup"] = [
            {
                "r": "lying_on",
                "t": "e1",
                "d": None,
                "e": "hypothesis",
                "q": "l",
                "c": ["ambiguous support geometry"],
            }
        ]
        wire = ExtractWireX3P3Runtime.model_validate(data)
        self.assertTrue(
            any("posture=seated" in warning and "lying_on" in warning for warning in wire.semantic_warnings())
        )

    def test_dangling_refs_remain_hard_failures(self) -> None:
        data = copy.deepcopy(self._wire_dict())
        data["r"][0]["o"] = "e9"
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate(data)

    def test_fragment_expands_to_safe_canonical_body_part(self) -> None:
        wire = ExtractWireX3P3Runtime.model_validate(self._wire_dict())
        canonical_model, metadata = expand_extract_wire(wire)
        canonical = canonical_model.model_dump(mode="json", by_alias=True)

        fragment = canonical["target_subject"]["visible_body_parts"][-1]
        self.assertEqual(fragment["part"], "fingers")
        self.assertEqual(fragment["visibility"], "fragment")
        self.assertEqual(fragment["ownership_candidate"], "unknown")
        self.assertEqual(fragment["connectivity_to_target_chain"], "disconnected_in_crop")
        self.assertEqual(fragment["visible_subparts"], ["visible_count=2"])
        self.assertEqual(metadata["wire_schema_version"], "x3p3")
        self.assertEqual(metadata["ambiguous_human_fragment_count"], 1)

    def test_expansion_stays_canonical_and_reconstructable(self) -> None:
        wire = ExtractWireX3P3Runtime.model_validate(self._wire_dict())
        canonical_model, _ = expand_extract_wire(wire)
        self.assertIsInstance(canonical_model, VisualExtractV3)
        canonical = canonical_model.model_dump(mode="json", by_alias=True)

        legacy_schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(canonical, legacy_schema), [])
        audit = audit_extract_contract(canonical)
        self.assertTrue(audit["analyze_reconstructable"])
        self.assertTrue(audit["gestalt_reconstructable"])


if __name__ == "__main__":
    unittest.main()
