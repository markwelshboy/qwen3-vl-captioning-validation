from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_135 import (
    _strip_side_bound_geometry,
    _withhold_migrated_side_geometry,
)


class Projection135Tests(unittest.TestCase):
    def test_corrected_shoulder_does_not_migrate_signed_depth_relation(self) -> None:
        payload = {
            "fusion": {
                "qualified_body_parts": [
                    {
                        "part": "shoulders",
                        "source_part": "shoulders",
                        "anatomical_side": "right",
                        "visible_subparts": ["right shoulder"],
                        "geometry": "shoulders level; right shoulder slightly forward and closer to camera",
                        "fusion_v2": {
                            "source_anatomical_side": "right",
                            "qualified_anatomical_side": "left",
                            "selection_usable": True,
                            "laterality_selection_usable": True,
                        },
                    }
                ]
            }
        }
        out, blocked = _withhold_migrated_side_geometry(payload)
        item = out["fusion"]["qualified_body_parts"][0]
        self.assertEqual(item["geometry"], "shoulders level")
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0]["reason"], "side_bound_geometry_cannot_migrate_across_laterality_correction")

    def test_corrected_arm_keeps_non_depth_pose_and_contact_geometry(self) -> None:
        payload = {
            "fusion": {
                "qualified_body_parts": [
                    {
                        "part": "left arm",
                        "source_part": "right_arm",
                        "anatomical_side": "right",
                        "visible_subparts": ["hand"],
                        "geometry": "elbow bent; hand under chin",
                        "contact": "hand touching chin",
                        "support": "hand supporting chin",
                        "fusion_v2": {
                            "source_anatomical_side": "right",
                            "qualified_anatomical_side": "left",
                            "selection_usable": True,
                            "laterality_selection_usable": True,
                        },
                    }
                ]
            }
        }
        out, blocked = _withhold_migrated_side_geometry(payload)
        item = out["fusion"]["qualified_body_parts"][0]
        self.assertEqual(item["geometry"], "elbow bent; hand under chin")
        self.assertEqual(item["support"], "hand supporting chin")
        self.assertEqual(blocked, [])

    def test_strip_side_bound_geometry_drops_whole_relation_when_no_neutral_clause(self) -> None:
        self.assertIsNone(_strip_side_bound_geometry("right shoulder slightly forward and closer to camera"))


if __name__ == "__main__":
    unittest.main()
