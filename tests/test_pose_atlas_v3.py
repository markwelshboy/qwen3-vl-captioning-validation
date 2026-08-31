from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qwen_caption_validate.pose_atlas_v3 import (
    _body_frame_vertices,
    _html_index,
    _load_annotations,
    _normalized_to_pixels,
)


class PoseAtlasV3Tests(unittest.TestCase):
    def test_normalized_points_scale_to_pixels(self) -> None:
        points = np.array([[0.25, 0.5], [1.0, 1.0]], dtype=np.float64)
        got = _normalized_to_pixels(points, 400, 200)
        np.testing.assert_allclose(got, np.array([[100.0, 100.0], [400.0, 200.0]]))

    def test_ndc_points_scale_to_pixels(self) -> None:
        points = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        got = _normalized_to_pixels(points, 400, 200)
        np.testing.assert_allclose(
            got,
            np.array([[0.0, 0.0], [200.0, 100.0], [400.0, 200.0]]),
        )

    def test_pixel_points_are_left_unchanged(self) -> None:
        points = np.array([[100.0, 50.0], [300.0, 180.0]], dtype=np.float64)
        got = _normalized_to_pixels(points, 400, 200)
        np.testing.assert_allclose(got, points)

    def test_body_frame_vertices_reverse_camera_system_flip_at_zero_rotation(self) -> None:
        vertices = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        arrays = {"global_rot": np.zeros(3, dtype=np.float64)}
        got = _body_frame_vertices(vertices, arrays)
        # At zero MHR root rotation, body_to_camera = diag(1,-1,-1), which is self-inverse.
        expected = np.array([[1.0, -2.0, -3.0], [-1.0, 2.0, 3.0]], dtype=np.float64)
        np.testing.assert_allclose(got, expected)

    def test_load_annotations_uses_record_mapping_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "pose-atlas-human-0.1",
                        "records": {
                            "img_a": {"pose_family": "seated"},
                            "bad": "not-a-record",
                        },
                    }
                ),
                encoding="utf-8",
            )
            got = _load_annotations(path)
        self.assertEqual(got, {"img_a": {"pose_family": "seated"}})

    def test_html_index_contains_card_and_summary(self) -> None:
        text = _html_index(
            [
                {
                    "image_key": "img_a",
                    "card_webp": "img_a.pose_atlas.webp",
                    "human_annotation": {"pose_family": "seated"},
                    "sam3d_diagnostic": {
                        "body_camera_relation": {
                            "orientation_band": "three_quarter",
                            "yaw_deg": 42.5,
                        }
                    },
                }
            ]
        )
        self.assertIn("img_a.pose_atlas.webp", text)
        self.assertIn("seated", text)
        self.assertIn("three_quarter", text)
        self.assertIn("42.5", text)


if __name__ == "__main__":
    unittest.main()
