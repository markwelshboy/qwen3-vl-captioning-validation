from __future__ import annotations

import copy
import json
import unittest

from pydantic import ValidationError

from qwen_caption_validate.extract_v3_models_x3p3 import ExtractWireX3P3Runtime
from tests.test_extract_v3_pydantic_x3p3 import ExtractV3PydanticX3P3Tests


class ExtractV3X3P3NormalizeTests(unittest.TestCase):
    def _wire_dict(self) -> dict:
        return copy.deepcopy(ExtractV3PydanticX3P3Tests()._wire_dict())

    def test_unanchored_target_hand_is_downgraded_with_dependent_claims(self) -> None:
        data = self._wire_dict()
        data["s"]["bp"] = [
            {
                "p": "right_hand",
                "a": "right",
                "o": "target",
                "v": "full",
                "s": ["palm", "fingers", "wrist"],
                "k": "connected_visible",
                "g": ["palm facing up", "fingers extended"],
                "c": ["hand_touching_neck"],
                "l": "lower_center",
                "q": "h",
            }
        ]
        data["s"]["hf"] = []
        data["s"]["ac"] = [
            {
                "c": "bracelet",
                "d": ["dark", "beaded"],
                "l": "right_wrist",
                "v": "partial",
                "q": "m",
            }
        ]
        data["s"]["ix"] = [
            {
                "k": "contact",
                "p": "right_hand",
                "o": "target",
                "r": "t",
                "x": None,
                "e": "observed",
                "q": "h",
                "c": ["touching neck", "palm against skin"],
            }
        ]
        data["s"]["mk"] = [
            {
                "c": "tattoo",
                "d": ["rose motif"],
                "l": "right_index_finger",
                "v": "partial",
                "q": "h",
            }
        ]

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))

        self.assertEqual(wire.subject.body_parts, [])
        self.assertEqual(len(wire.subject.human_fragments), 1)
        fragment = wire.subject.human_fragments[0]
        self.assertEqual(fragment.part, "hand_fragment")
        self.assertEqual(fragment.ownership, "unknown")
        self.assertEqual(fragment.connectivity, "unknown")
        self.assertIsNone(fragment.visible_count)
        self.assertEqual(fragment.geometry_cues, ["fingers extended"])
        self.assertNotIn("palm facing up", fragment.geometry_cues)
        self.assertEqual(wire.subject.interactions[0].ownership, "unknown")
        self.assertEqual(wire.subject.interactions[0].actor_part, "hand_fragment")
        self.assertEqual(wire.subject.interactions[0].cues, ["touching neck"])
        self.assertEqual(wire.subject.accessories, [])
        self.assertEqual(wire.subject.markings, [])
        self.assertTrue(any("bracelet" in item for item in wire.uncertainties))
        self.assertTrue(any("rose motif" in item for item in wire.uncertainties))

        report = wire.normalization_report()
        rules = [action["rule"] for action in report["actions"]]
        self.assertIn("unanchored_distal_target_part_to_fragment", rules)
        self.assertIn("interaction_actor_ownership_follows_fragment", rules)
        self.assertIn("target_accessory_to_ambiguous_fragment_uncertainty", rules)
        self.assertIn("target_marking_to_ambiguous_fragment_uncertainty", rules)
        self.assertEqual(report["action_count"], 4)
        downgrade = next(
            action
            for action in report["actions"]
            if action["rule"] == "unanchored_distal_target_part_to_fragment"
        )
        self.assertEqual(downgrade["removed_completion_cues"], ["palm facing up"])
        interaction = next(
            action
            for action in report["actions"]
            if action["rule"] == "interaction_actor_ownership_follows_fragment"
        )
        self.assertEqual(interaction["actor_part_from"], "right_hand")
        self.assertEqual(interaction["actor_part_to"], "hand_fragment")

    def test_visible_parent_arm_prevents_hand_downgrade(self) -> None:
        data = self._wire_dict()
        data["s"]["bp"] = [
            {
                "p": "right_arm",
                "a": "right",
                "o": "target",
                "v": "partial",
                "s": ["upper_arm", "forearm"],
                "k": "connected_visible",
                "g": ["arm traceable from shoulder"],
                "c": [],
                "l": "right_center",
                "q": "h",
            },
            {
                "p": "right_hand",
                "a": "right",
                "o": "target",
                "v": "full",
                "s": ["palm", "fingers", "wrist"],
                "k": "connected_visible",
                "g": ["wrist continuous with forearm"],
                "c": [],
                "l": "lower_right",
                "q": "h",
            },
        ]
        data["s"]["hf"] = []

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))
        self.assertEqual(len(wire.subject.body_parts), 2)
        self.assertEqual(wire.subject.human_fragments, [])
        self.assertEqual(wire.normalization_report()["action_count"], 0)

    def test_precision_marking_requires_explicit_palm_visibility(self) -> None:
        data = self._wire_dict()
        data["s"]["bp"] = [
            {
                "p": "left_arm",
                "a": "left",
                "o": "target",
                "v": "full",
                "s": ["shoulder", "upper_arm", "forearm", "hand"],
                "k": "connected_visible",
                "g": ["arm visible from shoulder to hand"],
                "c": [],
                "l": "left_center",
                "q": "h",
            }
        ]
        data["s"]["hf"] = []
        data["s"]["mk"] = [
            {
                "c": "tattoo",
                "d": ["small", "dark", "on_left_hand_palm"],
                "l": "left_hand",
                "v": "partial",
                "q": "h",
            }
        ]

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))
        self.assertEqual(wire.subject.markings, [])
        self.assertTrue(any("palm visibility not established" in item for item in wire.uncertainties))
        report = wire.normalization_report()
        self.assertEqual(report["action_count"], 1)
        self.assertEqual(report["actions"][0]["rule"], "target_marking_requires_visible_subpart")
        self.assertEqual(report["actions"][0]["required_subpart"], "palm")
        self.assertEqual(report["actions"][0]["required_side"], "left")

    def test_precision_marking_kept_when_palm_is_explicitly_visible(self) -> None:
        data = self._wire_dict()
        data["s"]["bp"] = [
            {
                "p": "left_hand",
                "a": "left",
                "o": "target",
                "v": "full",
                "s": ["palm", "fingers", "wrist"],
                "k": "connected_visible",
                "g": ["palm visible"],
                "c": [],
                "l": "left_center",
                "q": "h",
            },
            {
                "p": "left_arm",
                "a": "left",
                "o": "target",
                "v": "partial",
                "s": ["upper_arm", "forearm"],
                "k": "connected_visible",
                "g": ["arm traceable from shoulder"],
                "c": [],
                "l": "left_center",
                "q": "h",
            },
        ]
        data["s"]["hf"] = []
        data["s"]["mk"] = [
            {
                "c": "tattoo",
                "d": ["small", "dark", "on_left_hand_palm"],
                "l": "left_hand",
                "v": "partial",
                "q": "h",
            }
        ]

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))
        self.assertEqual(len(wire.subject.markings), 1)
        self.assertEqual(wire.normalization_report()["action_count"], 0)

    def test_dangling_support_ref_with_description_is_cleared(self) -> None:
        data = self._wire_dict()
        data["h"]["p"]["v"] = "seated"
        data["h"]["sup"] = [
            {
                "r": "seated_on",
                "t": "e5",
                "d": "light_colored_surface",
                "e": "contextual",
                "q": "m",
                "c": ["hips_out_of_crop", "surface_below_subject"],
            }
        ]

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))
        support = wire.hypotheses.support[0]
        self.assertIsNone(support.target_ref)
        self.assertEqual(support.target_description, "light_colored_surface")
        report = wire.normalization_report()
        self.assertEqual(report["action_count"], 1)
        self.assertEqual(report["actions"][0]["rule"], "support_dangling_ref_to_description")

    def test_dangling_support_without_description_remains_hard_failure(self) -> None:
        data = self._wire_dict()
        data["h"]["sup"] = [
            {
                "r": "seated_on",
                "t": "e5",
                "d": None,
                "e": "contextual",
                "q": "m",
                "c": ["ambiguous support"],
            }
        ]
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))

    def test_dangling_relation_is_never_normalized(self) -> None:
        data = self._wire_dict()
        data["r"][0]["o"] = "e5"
        with self.assertRaises(ValidationError):
            ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))

    def test_normalization_does_not_change_generated_schema(self) -> None:
        schema = ExtractWireX3P3Runtime.model_json_schema(by_alias=True)
        self.assertNotIn("_normalization_report", schema.get("properties", {}))
        self.assertIn("hf", schema["$defs"]["WireSubjectX3P3"]["properties"])


if __name__ == "__main__":
    unittest.main()
