from __future__ import annotations

import unittest

from qwen_caption_validate.head_gaze_evidence_v2 import build_record


class HeadGazeEvidenceV2Tests(unittest.TestCase):
    def _uniface(self, quality=0.8, yaw=8.0, pitch=-4.0, roll=2.0):
        return {
            "schema_version": "face-authority-uniface-0.2",
            "status": "ok",
            "face": {"score": 0.999, "bbox_xyxy": [100, 100, 300, 340]},
            "face_state": {"left_eye_open": 0.99, "right_eye_open": 0.99, "sunglasses": 0.0},
            "face_quality": {"score": quality},
            "face_mesh": {"inter_iris_distance_px": 90.0},
            "face_parsing": {"eye_region_diagnostics": {"iris_regions": [
                {"available": True, "fractions": {"hair": 0.0, "background": 0.0}},
                {"available": True, "fractions": {"hair": 0.0, "background": 0.0}},
            ]}},
            "head_pose": {"pitch_deg_raw": pitch, "yaw_deg_raw": yaw, "roll_deg_raw": roll},
        }

    def _pyfeat(self, yaw=9.0, pitch=-3.0, roll=-2.0):
        return {
            "schema_version": "gaze-probe-pyfeat-v28-0.2",
            "status": "ok",
            "head_pose": {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll},
        }

    def _l2cs(self):
        return {"schema_version": "gaze-probe-l2cs-0.3", "status": "ok", "gaze": {
            "pitch_deg": -20.0, "yaw_deg": 5.0, "forward_deviation_deg": 20.5,
        }}

    def test_agreeing_clean_head_is_corroborated(self):
        rec = build_record("x", self._uniface(), self._pyfeat(), self._l2cs(), {})
        self.assertEqual(rec["head"]["authority"], "corroborated")

    def test_low_quality_head_is_reduced_even_when_models_agree(self):
        rec = build_record("x", self._uniface(quality=0.2), self._pyfeat(), self._l2cs(), {})
        self.assertEqual(rec["head"]["authority"], "reduced")


if __name__ == "__main__":
    unittest.main()
