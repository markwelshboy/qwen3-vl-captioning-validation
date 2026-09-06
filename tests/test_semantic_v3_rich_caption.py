from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_caption import (
    _approx_prefill_seconds,
    _batch_size_runtime,
    _caption_stats,
)


class SemanticV3RichCaptionTests(unittest.TestCase):
    def test_caption_stats(self):
        stats = _caption_stats("One short sentence. Another useful sentence!")
        self.assertEqual(stats["words"], 7)
        self.assertEqual(stats["sentences"], 2)
        self.assertGreater(stats["characters"], 0)

    def test_prefill_approximation_when_metrics_exist(self):
        self.assertAlmostEqual(
            _approx_prefill_seconds({"ttft_seconds": 1.25, "queue_seconds": 0.20}),
            1.05,
        )
        self.assertIsNone(_approx_prefill_seconds({"ttft_seconds": None, "queue_seconds": None}))

    def test_batch_size_runtime_groups_and_amortizes(self):
        rows = _batch_size_runtime([
            {
                "batch_size": 2,
                "prepare_seconds": 0.10,
                "generation_seconds": 20.0,
                "wall_seconds": 20.2,
                "prompt_tokens": 2000,
                "output_tokens": 500,
            },
            {
                "batch_size": 2,
                "prepare_seconds": 0.08,
                "generation_seconds": 18.0,
                "wall_seconds": 18.2,
                "prompt_tokens": 1800,
                "output_tokens": 450,
            },
            {
                "batch_size": 1,
                "prepare_seconds": 0.04,
                "generation_seconds": 14.0,
                "wall_seconds": 14.1,
                "prompt_tokens": 900,
                "output_tokens": 200,
            },
        ])
        self.assertEqual([row["batch_size"] for row in rows], [1, 2])
        one, two = rows
        self.assertEqual(one["image_count"], 1)
        self.assertAlmostEqual(one["amortized_seconds_per_image"], 14.1)
        self.assertEqual(two["batch_count"], 2)
        self.assertEqual(two["image_count"], 4)
        self.assertAlmostEqual(two["amortized_seconds_per_image"], 9.6)
        self.assertAlmostEqual(two["aggregate_output_tokens_per_second"], 25.0)


if __name__ == "__main__":
    unittest.main()
