from __future__ import annotations

import unittest

from qwen_caption_validate.analysis_v2_normalize import normalize_analysis_v2


class AnalyzeV21NormalizeTests(unittest.TestCase):
    def test_body_part_object_and_array_text_fields_are_compacted(self) -> None:
        analysis = {
            "schema_version": "2.1",
            "target_subject": {
                "visible_body_parts": [
                    {
                        "part": "hand",
                        "anatomical_side": "left",
                        "geometry": {"elbow": "bent", "wrist": "flexed"},
                        "contact": {"type": "holding", "target": "cup"},
                        "support": [],
                    },
                    {
                        "part": "leg",
                        "anatomical_side": "right",
                        "geometry": ["knee bent", "hip flexed"],
                        "contact": ["resistance band"],
                        "support": None,
                    },
                ]
            },
        }

        normalized, actions = normalize_analysis_v2(analysis)
        parts = normalized["target_subject"]["visible_body_parts"]

        self.assertEqual(parts[0]["geometry"], "elbow=bent; wrist=flexed")
        self.assertEqual(parts[0]["contact"], "target=cup; type=holding")
        self.assertIsNone(parts[0]["support"])
        self.assertEqual(parts[1]["geometry"], "knee bent; hip flexed")
        self.assertEqual(parts[1]["contact"], "resistance band")
        self.assertIsNone(parts[1]["support"])
        self.assertEqual(len(actions), 5)

    def test_not_visible_body_part_is_removed_from_visible_only_list(self) -> None:
        analysis = {
            "schema_version": "2.1",
            "target_subject": {
                "geometry_landmark_visibility": {
                    "left_hip": {
                        "visibility": "not_visible",
                        "confidence": 0.99,
                        "evidence": "below crop",
                    }
                },
                "visible_body_parts": [
                    {
                        "part": "left_hand",
                        "anatomical_side": "left",
                        "ownership": "unknown",
                        "visibility": "not_visible",
                        "visible_subparts": [],
                        "connectivity_to_target_chain": "disconnected_in_crop",
                        "geometry": None,
                        "contact": None,
                        "support": None,
                        "foreshortening": "none",
                        "image_location": "not_visible",
                        "confidence": 0.0,
                    },
                    {
                        "part": "right_hand",
                        "anatomical_side": "right",
                        "ownership": "target",
                        "visibility": "partial",
                        "visible_subparts": ["fingers"],
                        "connectivity_to_target_chain": "connected_visible",
                        "geometry": "partly cropped",
                        "contact": None,
                        "support": None,
                        "foreshortening": "none",
                        "image_location": "lower_right",
                        "confidence": 0.9,
                    },
                ],
            },
        }

        normalized, actions = normalize_analysis_v2(analysis)
        parts = normalized["target_subject"]["visible_body_parts"]
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["part"], "right_hand")
        self.assertEqual(
            normalized["target_subject"]["geometry_landmark_visibility"],
            analysis["target_subject"]["geometry_landmark_visibility"],
        )
        self.assertEqual(len(actions), 1)
        self.assertIsNone(actions[0]["to"])

    def test_semantics_are_not_changed(self) -> None:
        analysis = {
            "schema_version": "2.1",
            "target_subject": {
                "geometry_landmark_visibility": {
                    "left_hip": {
                        "visibility": "not_visible",
                        "confidence": 0.99,
                        "evidence": "below crop",
                    }
                },
                "visible_body_parts": [],
            },
        }

        normalized, actions = normalize_analysis_v2(analysis)
        self.assertEqual(normalized["target_subject"]["geometry_landmark_visibility"], analysis["target_subject"]["geometry_landmark_visibility"])
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()
