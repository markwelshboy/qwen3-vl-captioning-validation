from __future__ import annotations

import unittest

from qwen_caption_validate.sam3d_dataset_profile import _authority_status, _band, build_profile


class Sam3DDatasetProfileTests(unittest.TestCase):
    def test_presentation_bands(self) -> None:
        self.assertEqual(_band(0.0), "low")
        self.assertEqual(_band(14.999), "low")
        self.assertEqual(_band(15.0), "moderate")
        self.assertEqual(_band(30.0), "high")
        self.assertEqual(_band(50.0), "very_high")

    def test_provenance_review_does_not_enter_qualified_histogram(self) -> None:
        audit = {
            "shoulder_depth_rotation": {"authority": "qualified_component_geometry"},
            "target_provenance": {"context_risk": "requires_review"},
        }
        self.assertEqual(_authority_status(audit), "pending_target_provenance")

    def test_profile_counts_only_qualified_records(self) -> None:
        profile = build_profile([
            {
                "image": "a.png",
                "angle_deg": 10.0,
                "band": "low",
                "authority_status": "qualified",
                "component_authority": "qualified_component_geometry",
                "support_state": "observed_supported",
                "target_provenance": {},
            },
            {
                "image": "b.png",
                "angle_deg": 62.0,
                "band": "very_high",
                "authority_status": "pending_target_provenance",
                "component_authority": "qualified_component_geometry",
                "support_state": "observed_supported",
                "target_provenance": {"context_risk": "requires_review"},
            },
        ])
        self.assertEqual(profile["qualified_image_count"], 1)
        self.assertEqual(profile["qualified_band_counts"]["low"], 1)
        self.assertEqual(profile["qualified_band_counts"]["very_high"], 0)
        self.assertEqual(profile["pending_provenance_band_counts"]["very_high"], 1)


if __name__ == "__main__":
    unittest.main()
