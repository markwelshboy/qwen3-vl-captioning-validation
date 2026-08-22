from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from qwen_caption_validate.signed_depth_refine import refine_signed_depth


BODY18 = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]


def _dwpose(*, shoulders: bool = True, hips: bool = True) -> dict:
    coords = {
        "left_shoulder": [0.70, 0.40],
        "right_shoulder": [0.30, 0.40],
        "left_hip": [0.66, 0.72],
        "right_hip": [0.34, 0.72],
    }
    bodies: list[list[float]] = []
    mapping = [-1.0] * 18
    for name, point in coords.items():
        if "shoulder" in name and not shoulders:
            continue
        if "hip" in name and not hips:
            continue
        mapping[BODY18.index(name)] = float(len(bodies))
        bodies.append(point)
    return {
        "image_width": 1000,
        "image_height": 1000,
        "derived": {"target_person_index": 0, "target": {"connectivity": {}}},
        "raw_pose": {"bodies": bodies, "body_scores": [mapping]},
    }


def _sam_npz(path: Path, *, conflict_shoulder: bool = False) -> dict:
    points = np.zeros((70, 2), dtype=np.float32)
    if conflict_shoulder:
        points[5] = [300, 400]
        points[6] = [700, 400]
    else:
        points[5] = [700, 400]
        points[6] = [300, 400]
    points[9] = [660, 720]
    points[10] = [340, 720]
    np.savez(path, pred_keypoints_2d=points)
    return {"arrays_npz": str(path)}


def _payload(
    *,
    shoulder_deg: float = 18.0,
    shoulder_fraction: float = 0.31,
    hip_deg: float = 19.0,
    hip_fraction: float = 0.33,
    hip_authority: str = "qualified_component_geometry",
    provenance_risk: bool = False,
) -> dict:
    return {
        "fusion": {
            "schema_version": "analysis-fusion-2.3.2",
            "qualified_body_parts": [
                {
                    "part": "torso",
                    "anatomical_side": "midline",
                    "ownership": "target",
                    "visibility": "full",
                    "visible_subparts": [],
                    "geometry": "upright",
                    "fusion_v2": {
                        "qualified_ownership": "target",
                        "qualified_anatomical_side": "midline",
                        "selection_usable": True,
                        "laterality_selection_usable": False,
                    },
                }
            ],
            "sam3d_geometry_audit": {
                "target_provenance": {
                    "context_risk": "requires_review" if provenance_risk else "no_semantic_multi_subject_risk_detected"
                },
                "shoulder_depth_rotation": {
                    "magnitude_deg": shoulder_deg,
                    "authority": "qualified_component_geometry",
                },
                "hip_depth_rotation": {
                    "magnitude_deg": hip_deg,
                    "authority": hip_authority,
                },
                "signed_depth_diagnostics": {
                    "shoulder_left_to_right": shoulder_fraction,
                    "hip_left_to_right": hip_fraction,
                },
            },
        }
    }


class SignedDepthRefineTests(unittest.TestCase):
    def _run(self, payload: dict, dw: dict | None = None, *, conflict_shoulder: bool = False, mirror: bool = False) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            sam_path = Path(tmp) / "x.sam3d.json"
            npz_path = Path(tmp) / "x.sam3d_arrays.npz"
            sam = _sam_npz(npz_path, conflict_shoulder=conflict_shoulder)
            analysis = {
                "framing": {"photographic_archetype": "mirror selfie" if mirror else "candid"},
                "image_summary": "",
            }
            return refine_signed_depth(payload, analysis, dw or _dwpose(), sam, sam_path)

    def test_positive_signed_fraction_qualifies_left_shoulder_nearer(self) -> None:
        out = self._run(_payload())
        audit = out["fusion"]["signed_depth_authority_audit"]
        shoulder = audit["components"]["shoulder"]
        self.assertEqual(shoulder["action"], "qualified")
        self.assertEqual(shoulder["nearer_anatomical_side"], "left")
        derived = [p for p in out["fusion"]["qualified_body_parts"] if p.get("derived_signed_depth")]
        self.assertEqual(len(derived), 1)
        self.assertEqual(derived[0]["part"], "left_shoulder")
        self.assertIn("rather than square-on", derived[0]["geometry"])

    def test_negative_signed_fraction_qualifies_right_shoulder_nearer(self) -> None:
        out = self._run(_payload(shoulder_fraction=-0.65, hip_fraction=-0.67))
        shoulder = out["fusion"]["signed_depth_authority_audit"]["components"]["shoulder"]
        self.assertEqual(shoulder["nearer_anatomical_side"], "right")
        derived = [p for p in out["fusion"]["qualified_body_parts"] if p.get("derived_signed_depth")]
        self.assertEqual(derived[0]["part"], "right_shoulder")

    def test_low_magnitude_does_not_promote_signed_direction(self) -> None:
        out = self._run(_payload(shoulder_deg=12.0, shoulder_fraction=-0.21, hip_authority="reconstructed_prior_only"))
        shoulder = out["fusion"]["signed_depth_authority_audit"]["components"]["shoulder"]
        self.assertEqual(shoulder["action"], "withheld")
        self.assertEqual(shoulder["reason"], "depth_rotation_below_signed_caption_threshold")
        self.assertFalse(any(p.get("derived_signed_depth") for p in out["fusion"]["qualified_body_parts"]))

    def test_provenance_risk_blocks_signed_direction(self) -> None:
        out = self._run(_payload(provenance_risk=True))
        shoulder = out["fusion"]["signed_depth_authority_audit"]["components"]["shoulder"]
        self.assertEqual(shoulder["action"], "withheld")
        self.assertEqual(shoulder["reason"], "target_provenance_requires_review")

    def test_sam_reconstruction_without_observed_pair_cannot_authorize_direction(self) -> None:
        out = self._run(_payload(), _dwpose(shoulders=False))
        shoulder = out["fusion"]["signed_depth_authority_audit"]["components"]["shoulder"]
        self.assertEqual(shoulder["action"], "withheld")
        self.assertEqual(shoulder["reason"], "bilateral_dwpose_component_not_observed")

    def test_sam_label_conflict_blocks_signed_direction(self) -> None:
        out = self._run(_payload(), conflict_shoulder=True)
        shoulder = out["fusion"]["signed_depth_authority_audit"]["components"]["shoulder"]
        self.assertEqual(shoulder["action"], "withheld")
        self.assertEqual(shoulder["reason"], "sam3d_anatomical_labels_not_correlated_with_observed_dwpose_pair")

    def test_mirror_blocks_signed_direction(self) -> None:
        out = self._run(_payload(), mirror=True)
        audit = out["fusion"]["signed_depth_authority_audit"]
        self.assertTrue(audit["mirror_sensitive"])
        self.assertEqual(audit["components"]["shoulder"]["reason"], "mirror_sensitive")
        self.assertFalse(any(p.get("derived_signed_depth") for p in out["fusion"]["qualified_body_parts"]))

    def test_torso_direction_requires_independent_shoulder_and_hip_agreement(self) -> None:
        agreed = self._run(_payload())
        self.assertEqual(
            agreed["fusion"]["signed_depth_authority_audit"]["torso_direction"]["action"],
            "qualified",
        )
        shoulder_only = self._run(_payload(hip_authority="reconstructed_prior_only"))
        self.assertEqual(
            shoulder_only["fusion"]["signed_depth_authority_audit"]["torso_direction"]["action"],
            "withheld",
        )
        derived = [p for p in shoulder_only["fusion"]["qualified_body_parts"] if p.get("derived_signed_depth")]
        self.assertNotIn("rather than square-on", derived[0]["geometry"])


if __name__ == "__main__":
    unittest.main()
