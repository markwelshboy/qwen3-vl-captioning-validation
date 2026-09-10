from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.dwpose_compat import target_points_from_profile_record


class DwposeCompatPointTests(unittest.TestCase):
    def test_historical_single_person_body18_scales_valid_points_and_keeps_missing_sentinel(self) -> None:
        bodies = [[-1.0, -1.0] for _ in range(18)]
        bodies[0] = [0.25, 0.50]
        bodies[1] = [0.50, 0.75]
        record = {
            "raw_pose": {"bodies": bodies},
            "derived": {"target_person_index": 0},
        }
        got = target_points_from_profile_record(record, 400, 200)
        self.assertEqual(got.shape, (18, 2))
        np.testing.assert_allclose(got[0], [100.0, 100.0])
        np.testing.assert_allclose(got[1], [200.0, 150.0])
        np.testing.assert_allclose(got[2], [-1.0, -1.0])

    def test_candidate_mapping_schema_is_supported(self) -> None:
        candidate = np.full((1, 18, 2), -1.0, dtype=np.float64)
        candidate[0, 0] = [0.75, 0.25]
        record = {
            "raw_pose": {"bodies": {"candidate": candidate.tolist()}},
            "derived": {"target_person_index": 0},
        }
        got = target_points_from_profile_record(record, 800, 600)
        np.testing.assert_allclose(got[0], [600.0, 150.0])
        np.testing.assert_allclose(got[1], [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
