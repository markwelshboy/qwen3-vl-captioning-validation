from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_141 import (
    _enrich_specific_accessory_states,
    _specific_accessory_states,
)


class ComposeGovernance141Tests(unittest.TestCase):
    def test_sunglasses_and_mask_states_are_preserved_without_anatomical_side(self) -> None:
        analysis = {
            "image_summary": (
                "A woman with sunglasses perched on her head wears a gray t-shirt. "
                "A yellow face mask hangs below her chin."
            )
        }
        self.assertEqual(
            _specific_accessory_states(analysis),
            [
                ("sunglasses", "sunglasses perched on head"),
                ("mask", "yellow face mask below chin"),
            ],
        )

    def test_specific_state_replaces_bare_item_descriptor(self) -> None:
        evidence = {
            "transient_appearance": {
                "descriptors": ["sunglasses", "yellow face mask", "gray t-shirt"]
            }
        }
        analysis = {
            "image_summary": "Sunglasses are perched on her head. A yellow face mask hangs below her chin."
        }
        audit: dict = {"allowed": []}
        _enrich_specific_accessory_states(evidence, analysis, audit)
        self.assertEqual(
            evidence["transient_appearance"]["descriptors"],
            ["gray t-shirt", "sunglasses perched on head", "yellow face mask below chin"],
        )

    def test_wristband_is_recovered_as_transient_accessory(self) -> None:
        analysis = {"image_summary": "Her hand rests on a surface, wearing a white wristband and a ring."}
        self.assertEqual(_specific_accessory_states(analysis), [("wristband", "white wristband")])

    def test_identity_hair_text_is_not_recovered(self) -> None:
        analysis = {"image_summary": "A blonde woman with shoulder-length hair and sunglasses perched on her head."}
        states = _specific_accessory_states(analysis)
        self.assertEqual(states, [("sunglasses", "sunglasses perched on head")])
        self.assertFalse(any("blonde" in value or "hair" in value for _, value in states))


if __name__ == "__main__":
    unittest.main()
