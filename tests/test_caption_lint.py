from __future__ import annotations

import unittest

from qwen_caption_validate.caption_lint import lint_caption


def _evidence() -> dict:
    return {
        "schema_version": "caption-evidence-1.1",
        "visibility_constraints": {
            "visible": ["head", "left_shoulder", "right_shoulder"],
            "partial": [],
            "not_visible": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
            "unknown": [],
        },
        "visible_subject_parts": [
            {
                "part": "hand",
                "anatomical_side": "unknown",
                "ownership": "target",
                "visible_subparts": ["hand", "fingers"],
                "laterality_qualified": False,
            }
        ],
        "qualified_interactions": [],
        "qualified_3d_geometry": {
            "shoulder_girdle_depth_rotation": {
                "magnitude_band": "very_high",
                "direction": "unsigned",
                "authority": "qualified_component_geometry",
            }
        },
        "required_claims": [
            {
                "id": "shoulder_girdle_depth_rotation",
                "priority": "required",
                "magnitude_band": "very_high",
            }
        ],
    }


class CaptionLintTests(unittest.TestCase):
    def test_clean_caption_passes(self) -> None:
        result = lint_caption(
            "sH1Vx is framed from the chest up, with the shoulders strongly staggered in depth and one visible hand near the torso.",
            _evidence(),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["warning_count"], 0)

    def test_unqualified_anatomical_side_fails(self) -> None:
        result = lint_caption(
            "sH1Vx raises the right hand while the shoulders are strongly staggered in depth.",
            _evidence(),
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any(item["type"] == "unqualified_anatomical_laterality" for item in result["violations"]))

    def test_frame_right_is_not_treated_as_anatomical_side(self) -> None:
        result = lint_caption(
            "sH1Vx looks toward the right side of the frame, with the shoulders staggered in depth.",
            _evidence(),
        )
        self.assertTrue(result["passed"])

    def test_hard_not_visible_hips_fail(self) -> None:
        result = lint_caption(
            "sH1Vx has the shoulders staggered in depth and the hips turned away.",
            _evidence(),
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any(item["type"] == "mentions_hard_not_visible_anatomy" for item in result["violations"]))

    def test_plural_hands_fail_without_two_qualified_hands(self) -> None:
        result = lint_caption(
            "sH1Vx holds a mug with both hands while the shoulders are staggered in depth.",
            _evidence(),
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any(item["type"] == "unsupported_plural_hands" for item in result["violations"]))

    def test_missing_required_geometry_warns(self) -> None:
        result = lint_caption("sH1Vx is shown in a medium close portrait with a neutral expression.", _evidence())
        self.assertTrue(result["passed"])
        self.assertEqual(result["warning_count"], 1)
        self.assertEqual(result["warnings"][0]["type"], "required_claim_not_detected")

    def test_pipeline_meta_language_fails(self) -> None:
        result = lint_caption(
            "sH1Vx has strongly staggered shoulders according to the evidence.",
            _evidence(),
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any(item["type"] == "pipeline_meta_language" for item in result["violations"]))


if __name__ == "__main__":
    unittest.main()
