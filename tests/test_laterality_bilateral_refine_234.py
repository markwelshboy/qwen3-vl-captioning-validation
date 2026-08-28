from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from qwen_caption_validate.laterality_bilateral_refine_234 import (
    _candidate_pairs,
    refine_complementary_bilateral_sets,
)


def _arm_payload(
    *,
    right_geometry: str = "elbow bent, hand holding phone",
    right_location: str = "right center",
) -> dict:
    return {
        "fusion": {
            "schema_version": "analysis-fusion-2.3.3",
            "qualified_body_parts": [
                {
                    "part": "arm",
                    "source_part": "left arm",
                    "anatomical_side": "left",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": ["shoulder", "elbow", "forearm", "hand"],
                    "geometry": "elbow bent, hand holding phone",
                    "contact": "holding smartphone",
                    "image_location": "left center",
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "selection_usable": True,
                        "source_anatomical_side": "left",
                        "qualified_anatomical_side": "unknown",
                        "laterality_selection_usable": False,
                    },
                },
                {
                    "part": "left arm",
                    "source_part": "right arm",
                    "anatomical_side": "right",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": ["shoulder", "elbow", "forearm", "hand"],
                    "geometry": right_geometry,
                    "contact": "holding smartphone",
                    "image_location": right_location,
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "selection_usable": True,
                        "source_anatomical_side": "right",
                        "qualified_anatomical_side": "left",
                        "laterality_selection_usable": True,
                    },
                },
            ],
        }
    }


class BilateralRefine234Tests(unittest.TestCase):
    def test_complementary_frame_locations_form_candidate_pair(self) -> None:
        parts = _arm_payload()["fusion"]["qualified_body_parts"]
        self.assertEqual(_candidate_pairs(parts), [("arm", 0, 1, "complementary")])

    def test_center_overlap_frame_locations_form_candidate_pair(self) -> None:
        parts = _arm_payload(right_location="center")["fusion"]["qualified_body_parts"]
        self.assertEqual(_candidate_pairs(parts), [("arm", 0, 1, "center_overlap")])

    def test_asymmetric_geometry_does_not_form_pair(self) -> None:
        parts = _arm_payload(right_geometry="arm straight at side")["fusion"]["qualified_body_parts"]
        self.assertEqual(_candidate_pairs(parts), [])

    def test_unrelated_frame_locations_do_not_form_pair(self) -> None:
        parts = _arm_payload(right_location="upper center")["fusion"]["qualified_body_parts"]
        self.assertEqual(_candidate_pairs(parts), [])

    def test_complementary_distal_arm_pair_restores_both_sides_and_clears_frame(self) -> None:
        payload = _arm_payload()
        with patch(
            "qwen_caption_validate.laterality_bilateral_refine_234._complete_bilateral_chains",
            return_value=True,
        ), patch(
            "qwen_caption_validate.laterality_bilateral_refine_234._bilateral_hand_support",
            return_value=(True, [{"qualified_side": "left"}, {"qualified_side": "right"}]),
        ):
            out = refine_complementary_bilateral_sets(payload, {}, {}, Path("missing.sam3d.json"))

        parts = out["fusion"]["qualified_body_parts"]
        self.assertEqual(parts[0]["fusion_v2"]["qualified_anatomical_side"], "left")
        self.assertEqual(parts[1]["fusion_v2"]["qualified_anatomical_side"], "right")
        self.assertEqual(parts[0]["part"], "left arm")
        self.assertEqual(parts[1]["part"], "right arm")
        self.assertIsNone(parts[0]["image_location"])
        self.assertIsNone(parts[1]["image_location"])
        self.assertEqual(out["fusion"]["schema_version"], "analysis-fusion-2.3.4")
        self.assertEqual(len(out["fusion"]["bilateral_complementary_frame_audit"]["pairs_applied"]), 1)

    def test_center_overlap_distal_pair_also_clears_frame(self) -> None:
        payload = _arm_payload(right_location="center")
        with patch(
            "qwen_caption_validate.laterality_bilateral_refine_234._complete_bilateral_chains",
            return_value=True,
        ), patch(
            "qwen_caption_validate.laterality_bilateral_refine_234._bilateral_hand_support",
            return_value=(True, [{"qualified_side": "left"}, {"qualified_side": "right"}]),
        ):
            out = refine_complementary_bilateral_sets(payload, {}, {}, Path("missing.sam3d.json"))
        parts = out["fusion"]["qualified_body_parts"]
        self.assertEqual(parts[0]["fusion_v2"]["qualified_anatomical_side"], "left")
        self.assertEqual(parts[1]["fusion_v2"]["qualified_anatomical_side"], "right")
        self.assertIsNone(parts[0]["image_location"])
        self.assertIsNone(parts[1]["image_location"])

    def test_two_observed_hands_are_required_for_distal_pair(self) -> None:
        payload = _arm_payload()
        with patch(
            "qwen_caption_validate.laterality_bilateral_refine_234._complete_bilateral_chains",
            return_value=True,
        ), patch(
            "qwen_caption_validate.laterality_bilateral_refine_234._bilateral_hand_support",
            return_value=(False, [{"qualified_side": "left"}]),
        ):
            out = refine_complementary_bilateral_sets(payload, {}, {}, Path("missing.sam3d.json"))
        parts = out["fusion"]["qualified_body_parts"]
        self.assertEqual(parts[0]["fusion_v2"]["qualified_anatomical_side"], "unknown")
        self.assertEqual(len(out["fusion"]["bilateral_complementary_frame_audit"]["pairs_applied"]), 0)


if __name__ == "__main__":
    unittest.main()
