from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_caption_validate import pose_atlas_v3 as base
from qwen_caption_validate.pose_atlas_v3_02 import (
    _draw_3d_pose_panel,
    _masked_dwpose_points,
    _reprojection_residual,
    _resolve_mesh,
)


class PoseAtlasV302Tests(unittest.TestCase):
    def test_masked_dwpose_points_hides_landmarks_not_marked_visible(self) -> None:
        body = [[float(i) / 20.0, float(i) / 20.0] for i in range(18)]
        record = {
            "raw_pose": {"bodies": body},
            "derived": {
                "target_person_index": 0,
                "target": {
                    "visible_body_landmarks": ["nose", "left_shoulder"],
                },
            },
        }
        points, visible = _masked_dwpose_points(record, 200, 100)
        self.assertEqual(visible, {"nose", "left_shoulder"})
        self.assertTrue(base._point_visible(points[base.IDX["nose"]]))
        self.assertTrue(base._point_visible(points[base.IDX["left_shoulder"]]))
        self.assertFalse(base._point_visible(points[base.IDX["right_shoulder"]]))

    def test_reprojection_residual_is_zero_for_matching_common_joints(self) -> None:
        dw = np.full((18, 2), -1.0, dtype=np.float64)
        sam = np.full((70, 2), -1.0, dtype=np.float64)
        dw[base.IDX["nose"]] = [100.0, 50.0]
        dw[base.IDX["left_shoulder"]] = [80.0, 90.0]
        sam[0] = [100.0, 50.0]
        sam[5] = [80.0, 90.0]
        record = {
            "derived": {
                "target": {
                    "keypoint_bbox": {
                        "width_fraction": 0.5,
                        "height_fraction": 0.5,
                    }
                }
            }
        }
        got = _reprojection_residual(
            dw,
            sam,
            {"nose", "left_shoulder"},
            width=200,
            height=100,
            dwpose=record,
        )
        self.assertEqual(got["common_joint_count"], 2)
        self.assertEqual(got["median_px"], 0.0)
        self.assertEqual(got["mean_px"], 0.0)
        self.assertEqual(got["median_fraction_of_dwpose_bbox_diagonal"], 0.0)

    def test_reprojection_residual_reports_known_pixel_offset(self) -> None:
        dw = np.full((18, 2), -1.0, dtype=np.float64)
        sam = np.full((70, 2), -1.0, dtype=np.float64)
        dw[base.IDX["nose"]] = [10.0, 10.0]
        sam[0] = [13.0, 14.0]
        record = {
            "derived": {
                "target": {
                    "keypoint_bbox": {
                        "width_fraction": 1.0,
                        "height_fraction": 1.0,
                    }
                }
            }
        }
        got = _reprojection_residual(
            dw,
            sam,
            {"nose"},
            width=100,
            height=100,
            dwpose=record,
        )
        self.assertEqual(got["common_joint_count"], 1)
        self.assertEqual(got["median_px"], 5.0)
        self.assertEqual(got["mean_px"], 5.0)

    def test_3d_pose_panel_renders_without_mesh(self) -> None:
        points = np.zeros((70, 3), dtype=np.float64)
        points[0] = [0.0, 1.7, 0.0]
        points[5] = [0.25, 1.45, 0.0]
        points[6] = [-0.25, 1.45, 0.0]
        points[7] = [0.45, 1.15, 0.05]
        points[8] = [-0.45, 1.15, -0.05]
        points[9] = [0.15, 0.9, 0.0]
        points[10] = [-0.15, 0.9, 0.0]
        points[11] = [0.15, 0.5, 0.05]
        points[12] = [-0.15, 0.5, -0.05]
        points[13] = [0.15, 0.05, 0.1]
        points[14] = [-0.15, 0.05, -0.1]
        points[41] = [-0.55, 0.95, -0.05]
        points[62] = [0.55, 0.95, 0.05]
        points[69] = [0.0, 1.5, 0.0]
        panel = _draw_3d_pose_panel(
            points,
            np.empty((0, 3), dtype=np.float64),
            (0, 1),
            "test",
        )
        self.assertIsInstance(panel, Image.Image)
        self.assertEqual(panel.size, (base.PANEL_W, base.PANEL_H))

    def test_resolve_mesh_finds_direct_obj(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = root / "img_a.sam3d.obj"
            mesh.write_text("v 0 0 0\n", encoding="utf-8")
            self.assertEqual(_resolve_mesh(root, "img_a"), mesh)

    def test_resolve_mesh_uses_record_basename_when_original_path_moved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            moved = root / "nested" / "img_a.sam3d.obj"
            moved.parent.mkdir(parents=True)
            moved.write_text("v 0 0 0\n", encoding="utf-8")
            (root / "img_a.sam3d.json").write_text(
                json.dumps({"mesh_obj": "/old/machine/path/img_a.sam3d.obj"}),
                encoding="utf-8",
            )
            self.assertEqual(_resolve_mesh(root, "img_a"), moved)


if __name__ == "__main__":
    unittest.main()
