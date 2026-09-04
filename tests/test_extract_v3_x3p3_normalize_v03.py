from __future__ import annotations

import copy
import json
import unittest

from qwen_caption_validate.extract_v3_models_x3p3 import ExtractWireX3P3Runtime
from tests.test_extract_v3_pydantic_x3p3 import ExtractV3PydanticX3P3Tests


class ExtractV3X3P3NormalizeV03Tests(unittest.TestCase):
    def _wire_dict(self) -> dict:
        return copy.deepcopy(ExtractV3PydanticX3P3Tests()._wire_dict())

    def test_distal_only_arm_label_is_downgraded(self) -> None:
        data = self._wire_dict()
        data["s"]["bp"] = [
            {
                "p": "right_arm",
                "a": "right",
                "o": "target",
                "v": "partial",
                "s": ["forearm", "hand"],
                "k": "connected_visible",
                "g": ["extended", "angled_downward"],
                "c": ["touching_neck"],
                "l": "bottom_right",
                "q": "h",
            }
        ]
        data["s"]["hf"] = []
        data["s"]["ac"] = [
            {
                "c": "watch",
                "d": ["metallic", "dark band"],
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
                "c": ["fingers_on_skin"],
            }
        ]

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))

        self.assertEqual(wire.subject.body_parts, [])
        self.assertEqual(len(wire.subject.human_fragments), 1)
        self.assertEqual(wire.subject.human_fragments[0].part, "arm_fragment")
        self.assertEqual(wire.subject.human_fragments[0].ownership, "unknown")
        self.assertEqual(wire.subject.human_fragments[0].connectivity, "unknown")
        self.assertEqual(wire.subject.interactions[0].ownership, "unknown")
        self.assertEqual(wire.subject.interactions[0].actor_part, "hand_fragment")
        self.assertEqual(wire.subject.accessories, [])
        self.assertTrue(any("watch" in item for item in wire.uncertainties))

        report = wire.normalization_report()
        self.assertEqual(report["version"], "x3p3-governance-0.3")
        rules = [action["rule"] for action in report["actions"]]
        self.assertIn("unanchored_limb_chain_to_fragment", rules)
        self.assertIn("interaction_actor_follows_unanchored_limb_fragment", rules)
        self.assertIn("target_accessory_follows_unanchored_limb_fragment", rules)

    def test_arm_with_upper_arm_anchor_is_preserved(self) -> None:
        data = self._wire_dict()
        data["s"]["bp"] = [
            {
                "p": "right_arm",
                "a": "right",
                "o": "target",
                "v": "full",
                "s": ["shoulder", "upper_arm", "forearm", "hand"],
                "k": "connected_visible",
                "g": ["visible shoulder-to-hand chain"],
                "c": [],
                "l": "right_center",
                "q": "h",
            }
        ]
        data["s"]["hf"] = []

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))
        self.assertEqual(len(wire.subject.body_parts), 1)
        self.assertEqual(wire.subject.human_fragments, [])
        self.assertEqual(wire.normalization_report()["action_count"], 0)

    def test_distal_only_leg_label_is_downgraded(self) -> None:
        data = self._wire_dict()
        data["s"]["bp"] = [
            {
                "p": "left_leg",
                "a": "left",
                "o": "target",
                "v": "partial",
                "s": ["knee", "shin", "foot"],
                "k": "connected_visible",
                "g": ["lower leg crosses foreground"],
                "c": [],
                "l": "lower_left",
                "q": "h",
            }
        ]
        data["s"]["hf"] = []

        wire = ExtractWireX3P3Runtime.model_validate_json(json.dumps(data))
        self.assertEqual(wire.subject.body_parts, [])
        self.assertEqual(wire.subject.human_fragments[0].part, "leg_fragment")
        self.assertEqual(wire.subject.human_fragments[0].ownership, "unknown")
        self.assertEqual(wire.normalization_report()["version"], "x3p3-governance-0.3")


if __name__ == "__main__":
    unittest.main()
