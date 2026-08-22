from __future__ import annotations

import unittest
from pathlib import Path

from qwen_caption_validate.laterality_bilateral_guard import guard_bilateral_sets


def _payload(*, asymmetric: bool = False) -> dict:
    left_geometry = "leg bent" if asymmetric else "leg straight, boot visible"
    return {
        "fusion": {
            "qualified_body_parts": [
                {
                    "part": "legs",
                    "anatomical_side": "left",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": ["thigh", "knee", "lower leg", "boot"],
                    "geometry": left_geometry,
                    "support": "standing on floor",
                    "image_location": "lower center",
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "selection_usable": True,
                        "source_anatomical_side": "left",
                        "qualified_anatomical_side": "right",
                        "laterality_selection_usable": True,
                    },
                },
                {
                    "part": "legs",
                    "anatomical_side": "right",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": ["thigh", "knee", "lower leg", "boot"],
                    "geometry": "leg straight, boot visible",
                    "support": "standing on floor",
                    "image_location": "lower center",
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "selection_usable": True,
                        "source_anatomical_side": "right",
                        "qualified_anatomical_side": "right",
                        "laterality_selection_usable": True,
                    },
                },
            ]
        }
    }


def _dw(*, complete: bool = True) -> dict:
    return {
        "derived": {
            "target": {
                "connectivity": {
                    "left_leg": {"complete": complete},
                    "right_leg": {"complete": complete},
                }
            }
        },
        "raw_pose": {},
    }


class BilateralGuardTests(unittest.TestCase):
    def test_equivalent_leg_records_are_restored_to_bilateral_set(self) -> None:
        out = guard_bilateral_sets(_payload(), _dw(), {}, Path("missing.sam3d.json"))
        parts = out["fusion"]["qualified_body_parts"]
        self.assertEqual(parts[0]["fusion_v2"]["qualified_anatomical_side"], "left")
        self.assertEqual(parts[1]["fusion_v2"]["qualified_anatomical_side"], "right")
        self.assertEqual(len(out["fusion"]["bilateral_equivalence_audit"]["pairs_applied"]), 1)

    def test_asymmetric_semantics_are_not_reassigned_as_unordered_set(self) -> None:
        out = guard_bilateral_sets(_payload(asymmetric=True), _dw(), {}, Path("missing.sam3d.json"))
        parts = out["fusion"]["qualified_body_parts"]
        self.assertEqual(parts[0]["fusion_v2"]["qualified_anatomical_side"], "right")
        self.assertEqual(len(out["fusion"]["bilateral_equivalence_audit"]["pairs_applied"]), 0)

    def test_incomplete_bilateral_chains_do_not_authorize_set(self) -> None:
        out = guard_bilateral_sets(_payload(), _dw(complete=False), {}, Path("missing.sam3d.json"))
        parts = out["fusion"]["qualified_body_parts"]
        self.assertEqual(parts[0]["fusion_v2"]["qualified_anatomical_side"], "right")
        self.assertEqual(len(out["fusion"]["bilateral_equivalence_audit"]["pairs_applied"]), 0)


if __name__ == "__main__":
    unittest.main()
