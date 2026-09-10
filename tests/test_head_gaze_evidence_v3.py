from __future__ import annotations

import unittest

from qwen_caption_validate.head_gaze_evidence_v3 import build_record


class HeadGazeEvidenceV3Tests(unittest.TestCase):
    def _uniface(self, *, quality=0.8, yaw=42.0, pitch=-28.0, roll=6.0):
        return {
            "schema_version": "face-authority-uniface-0.2",
            "status": "ok",
            "face": {"score": 0.999, "bbox_xyxy": [100, 100, 300, 340]},
            "face_state": {"left_eye_open": 0.9, "right_eye_open": 0.9, "sunglasses": 0.0},
            "face_quality": {"score": quality},
            "face_mesh": {"inter_iris_distance_px": 80.0},
            "face_parsing": {"eye_region_diagnostics": {"iris_regions": [
                {"available": True, "fractions": {"hair": 0.0, "background": 0.0}},
                {"available": True, "fractions": {"hair": 0.0, "background": 0.0}},
            ]}},
            "head_pose": {"pitch_deg_raw": pitch, "yaw_deg_raw": yaw, "roll_deg_raw": roll},
        }

    def _pyfeat(self, *, yaw=31.0, pitch=-25.0, roll=-24.0):
        return {
            "schema_version": "gaze-probe-pyfeat-v28-0.3",
            "status": "ok",
            "head_pose": {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll},
        }

    def _l2cs(self):
        return {
            "schema_version": "gaze-probe-l2cs-0.3",
            "status": "ok",
            "gaze": {"pitch_deg": -30.0, "yaw_deg": 40.0, "forward_deviation_deg": 50.0},
        }

    def test_roll_disagreement_does_not_make_head_conflict(self):
        rec = build_record("x", self._uniface(), self._pyfeat(), self._l2cs(), {})
        self.assertNotEqual(rec["head"]["authority"], "conflict")
        self.assertEqual(rec["head"]["axis_authority"]["pitch"]["authority"], "corroborated")
        self.assertIn(rec["head"]["axis_authority"]["yaw"]["authority"], {"corroborated", "corroborated_direction"})
        self.assertEqual(rec["head"]["axis_authority"]["roll"]["authority"], "reduced")

    def test_center_vs_down_pitch_is_reduced_not_conflict(self):
        rec = build_record(
            "x",
            self._uniface(yaw=-4.0, pitch=-15.0, roll=32.0),
            self._pyfeat(yaw=5.5, pitch=-2.5, roll=-29.0),
            self._l2cs(),
            {},
        )
        self.assertEqual(rec["head"]["axis_authority"]["yaw"]["authority"], "corroborated")
        self.assertEqual(rec["head"]["axis_authority"]["pitch"]["authority"], "reduced")
        self.assertEqual(rec["head"]["authority"], "reduced")

    def test_true_left_right_head_disagreement_is_conflict(self):
        rec = build_record(
            "x",
            self._uniface(yaw=35.0, pitch=-5.0),
            self._pyfeat(yaw=-30.0, pitch=-4.0),
            self._l2cs(),
            {},
        )
        self.assertEqual(rec["head"]["axis_authority"]["yaw"]["authority"], "conflict")
        self.assertEqual(rec["head"]["authority"], "conflict")

    def test_true_up_down_head_disagreement_is_conflict(self):
        rec = build_record(
            "x",
            self._uniface(yaw=5.0, pitch=-25.0),
            self._pyfeat(yaw=4.0, pitch=22.0),
            self._l2cs(),
            {},
        )
        self.assertEqual(rec["head"]["axis_authority"]["pitch"]["authority"], "conflict")
        self.assertEqual(rec["head"]["authority"], "conflict")


if __name__ == "__main__":
    unittest.main()
