from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.pose_atlas_v3_compat import (
    _dwpose_target_points_compat,
    _normalize_candidate_array,
)


class PoseAtlasV3CompatTests(unittest.TestCase):
    def _person(self, x_offset: float = 0.0) -> list[list[float]]:
        return [[x_offset + (i / 100.0), i / 200.0] for i in range(18)]

    def test_candidate_dict_shape(self) -> None:
        got = _normalize_candidate_array({"candidate": [self._person()]})
        self.assertEqual(got.shape, (1, 18, 2))

    def test_direct_bodies_list_shape_from_historical_cache(self) -> None:
        # This is the shape that previously crashed because pose_atlas_v3
        # assumed raw_pose.bodies was always a dictionary with .get().
        got = _normalize_candidate_array([self._person()])
        self.assertEqual(got.shape, (1, 18, 2))

    def test_single_person_joint_list_shape(self) -> None:
        got = _normalize_candidate_array(self._person())
        self.assertEqual(got.shape, (1, 18, 2))

    def test_list_of_person_dicts_preserves_target_index(self) -> None:
        record = {
            "raw_pose": {
                "bodies": [
                    {"candidate": self._person(0.0)},
                    {"candidate": self._person(0.5)},
                ]
            },
            "derived": {"target_person_index": 1},
        }
        got = _dwpose_target_points_compat(record, 1000, 500)
        self.assertEqual(got.shape, (18, 2))
        # x=0.5 in normalized coordinates should become 500 pixels.
        self.assertAlmostEqual(float(got[0, 0]), 500.0)

    def test_heterogeneous_candidate_subset_wrapper_prefers_candidate(self) -> None:
        candidate = [self._person()]
        subset = [[0.0] * 20]
        got = _normalize_candidate_array([candidate, subset])
        self.assertEqual(got.shape, (1, 18, 2))

    def test_missing_or_unusable_cache_returns_empty_overlay(self) -> None:
        got = _dwpose_target_points_compat(
            {"raw_pose": {"bodies": []}, "derived": {"target_person_index": 0}},
            1000,
            500,
        )
        self.assertEqual(got.shape, (0, 2))
        self.assertEqual(got.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
