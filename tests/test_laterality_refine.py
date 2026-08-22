from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qwen_caption_validate.laterality_refine import _hand_entities, _target_points, refine_laterality


BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]


def _dwpose(*, left_wrist=True, right_wrist=False, tied_wrists=False) -> dict:
    coords = {
        "right_shoulder": [0.20, 0.65],
        "right_elbow": [0.06, 0.97],
        "left_shoulder": [0.71, 0.69],
        "left_elbow": [0.74, 0.99],
        "left_wrist": [0.61, 0.73],
        "right_wrist": [0.13, 0.83],
    }
    if tied_wrists:
        coords["left_wrist"] = [0.143, 0.832]
        coords["right_wrist"] = [0.134, 0.827]
    bodies = []
    mapping = [-1.0] * 18
    for name, point in coords.items():
        if name == "left_wrist" and not left_wrist:
            continue
        if name == "right_wrist" and not right_wrist:
            continue
        mapping[BODY18.index(name)] = float(len(bodies))
        bodies.append(point)
    return {
        "image_width": 1024,
        "image_height": 1024,
        "derived": {
            "target_person_index": 0,
            "target": {
                "connectivity": {
                    "left_arm": {
                        "visible_count": 3 if left_wrist else 2,
                        "complete": bool(left_wrist),
                    },
                    "right_arm": {
                        "visible_count": 3 if right_wrist else 2,
                        "complete": bool(right_wrist),
                    },
                    "left_leg": {"visible_count": 0, "complete": False},
                    "right_leg": {"visible_count": 0, "complete": False},
                }
            },
        },
        "raw_pose": {
            "bodies": bodies,
            "body_scores": [mapping],
            "hands": [[[0.603, 0.705]]] if not tied_wrists else [[[0.140, 0.826]]],
            "hands_scores": [[0.83]],
        },
    }


def _sam_npz(path: Path, *, left_wrist=(618, 754), right_wrist=(228, 1343), conflict=False) -> dict:
    points = np.zeros((70, 2), dtype=np.float32)
    if conflict:
        points[62] = [900, 900]
        points[41] = [618, 754]
    else:
        points[62] = left_wrist
        points[41] = right_wrist
    points[5] = [744, 715]
    points[6] = [209, 663]
    points[7] = [736, 1010]
    points[8] = [29, 992]
    np.savez(path, pred_keypoints_2d=points)
    return {"arrays_npz": str(path)}


def _fusion_payload() -> dict:
    return {
        "fusion": {
            "deterministic_geometry": {
                "connectivity": {
                    "left_arm": {"visible_count": 3, "complete": True},
                    "right_arm": {"visible_count": 2, "complete": False},
                },
                "hand_candidates": [
                    {
                        "candidate_index": 0,
                        "nearest_visible_target_wrist": "left",
                        "supported_by_nearby_visible_target_wrist": True,
                    }
                ],
            },
            "qualified_body_parts": [
                {
                    "part": "right_arm",
                    "anatomical_side": "right",
                    "ownership": "target",
                    "visible_subparts": ["upper arm", "forearm", "hand"],
                    "geometry": "elbow bent; hand under chin",
                    "contact": "hand touching chin",
                    "support": "hand supporting chin",
                    "image_location": "center-right",
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "unknown",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                        "laterality_reasons": [],
                    },
                },
                {
                    "part": "left_arm",
                    "anatomical_side": "left",
                    "ownership": "target",
                    "visible_subparts": ["upper arm", "shoulder"],
                    "geometry": "arm resting at side",
                    "image_location": "center-left",
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "left",
                        "selection_usable": True,
                        "laterality_selection_usable": True,
                        "laterality_reasons": [],
                    },
                },
            ],
            "qualified_interactions": [
                {
                    "actor_part": "right hand",
                    "actor_ownership": "target",
                    "target": "chin",
                    "evidence_status": "observed",
                    "fusion_v2": {
                        "qualified_actor_ownership": "target",
                        "qualified_actor_anatomical_side": "unknown",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                    },
                }
            ],
        }
    }


class LateralityAuthorityTests(unittest.TestCase):
    def test_dwpose_complete_chain_and_sam_override_analyze_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.sam3d_arrays.npz"
            sam = _sam_npz(path)
            out = refine_laterality(_fusion_payload(), {"framing": {}, "image_summary": ""}, _dwpose(), sam, Path(tmp) / "x.sam3d.json")
        parts = out["fusion"]["qualified_body_parts"]
        self.assertEqual(parts[0]["fusion_v2"]["qualified_anatomical_side"], "left")
        self.assertEqual(parts[0]["part"], "left arm")
        self.assertEqual(parts[1]["fusion_v2"]["qualified_anatomical_side"], "right")
        interaction = out["fusion"]["qualified_interactions"][0]
        self.assertEqual(interaction["actor_part"], "left hand")
        self.assertEqual(interaction["fusion_v2"]["qualified_actor_anatomical_side"], "left")

    def test_sam_reconstruction_without_observed_wrist_cannot_authorize_hand(self) -> None:
        dw = _dwpose(left_wrist=False, right_wrist=False)
        points = _target_points(dw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.sam3d_arrays.npz"
            sam = _sam_npz(path)
            with np.load(path) as arrays:
                entities = _hand_entities(dw, points, arrays["pred_keypoints_2d"])
        self.assertEqual(entities, [])

    def test_near_tied_wrists_prefer_complete_chain_then_sam(self) -> None:
        dw = _dwpose(left_wrist=True, right_wrist=True, tied_wrists=True)
        dw["derived"]["target"]["connectivity"]["right_arm"]["complete"] = False
        dw["derived"]["target"]["connectivity"]["right_arm"]["visible_count"] = 2
        points = _target_points(dw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.sam3d_arrays.npz"
            points2d = np.zeros((70, 2), dtype=np.float32)
            points2d[62] = [112, 839]
            points2d[41] = [500, 1200]
            np.savez(path, pred_keypoints_2d=points2d)
            with np.load(path) as arrays:
                entities = _hand_entities(dw, points, arrays["pred_keypoints_2d"])
        self.assertEqual(entities[0]["qualified_side"], "left")
        self.assertIn("complete_chain", entities[0]["resolution_reason"])

    def test_sam_disagreement_blocks_dwpose_side(self) -> None:
        dw = _dwpose()
        points = _target_points(dw)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.sam3d_arrays.npz"
            sam = _sam_npz(path, conflict=True)
            with np.load(path) as arrays:
                entities = _hand_entities(dw, points, arrays["pred_keypoints_2d"])
        self.assertIsNone(entities[0]["qualified_side"])
        self.assertIn("sam3d_disagrees", entities[0]["resolution_reason"])

    def test_mirror_withholds_corrective_laterality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.sam3d_arrays.npz"
            sam = _sam_npz(path)
            out = refine_laterality(_fusion_payload(), {"framing": {"photographic_archetype": "mirror selfie"}}, _dwpose(), sam, Path(tmp) / "x.sam3d.json")
        self.assertFalse(out["fusion"]["qualified_body_parts"][0]["fusion_v2"]["laterality_selection_usable"])


if __name__ == "__main__":
    unittest.main()
