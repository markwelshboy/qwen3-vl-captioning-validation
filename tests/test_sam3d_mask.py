from __future__ import annotations

import unittest

import numpy as np

from qwen_caption_validate.sam3d_mask import _entropy_masks


class Sam3DMaskTests(unittest.TestCase):
    def test_subject_zone_expands_core_and_weights_remain_bounded(self) -> None:
        alpha = np.zeros((64, 64), dtype=np.float32)
        alpha[20:44, 24:40] = 1.0

        core, zone, weight, stats = _entropy_masks(
            alpha,
            alpha_threshold=0.01,
            dilate_frac=0.10,
            feather_frac=0.10,
            background_weight=0.35,
        )

        self.assertGreater(int(zone.sum()), int(core.sum()))
        self.assertAlmostEqual(float(weight[32, 32]), 1.0, places=6)
        self.assertGreaterEqual(float(weight.min()), 0.35)
        self.assertLessEqual(float(weight.max()), 1.0)
        self.assertGreater(stats["subject_zone_frame_fraction"], stats["core_frame_fraction"])
        self.assertGreater(stats["dilate_px"], 0)
        self.assertGreater(stats["feather_px"], 0)


if __name__ == "__main__":
    unittest.main()
