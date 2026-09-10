from __future__ import annotations

import unittest

from qwen_caption_validate.head_gaze_evidence import build_record


class HeadGazeEvidenceTests(unittest.TestCase):
    def _base_uniface(self):
        return {
            "schema_version": "face-authority-uniface-0.2",
            "status": "ok",
            "face": {"score": 0.999, "bbox_xyxy": [100, 100, 300, 340]},
            "face_state": {
                "left_eye_open": 0.99,
                "right_eye_open": 0.99,
                "eyeglasses": 0.0,
                "mask": 0.0,
                "sunglasses": 0.0,
            },
            "face_quality": {"score": 0.75},
            "face_mesh": {"inter_iris_distance_px": 90.0},
            "face_parsing": {
                "eye_region_diagnostics": {
                    "iris_regions": [
                        {"available": True, "fractions": {"hair": 0.0, "background": 0.0}},
                        {"available": True, "fractions": {"hair": 0.0, "background": 0.0}},
                    ]
                }
            },
            "head_pose": {"pitch_deg_raw": -4.0, "yaw_deg_raw": 5.0, "roll_deg_raw": 2.0},
        }

    def _pyfeat(self):
        return {
            "schema_version": "gaze-probe-pyfeat-v28-0.2",
            "status": "ok",
            "head_pose": {"pitch_deg": -3.0, "yaw_deg": 6.0, "roll_deg": -2.0},
        }

    def _l2cs(self):
        return {
            "schema_version": "gaze-probe-l2cs-0.3",
            "status": "ok",
            "gaze": {"pitch_deg": -25.0, "yaw_deg": 20.0, "forward_deviation_deg": 31.0},
        }

    def test_frame_direction_signs_are_explicit(self):
        rec = build_record("x", self._base_uniface(), self._pyfeat(), self._l2cs(), {})
        self.assertEqual(rec["gaze"]["horizontal"], "frame_left")
        self.assertEqual(rec["gaze"]["vertical"], "down")
        self.assertEqual(rec["gaze"]["camera_relationship"], "off_camera")

    def test_sunglasses_hard_null_gaze(self):
        uni = self._base_uniface()
        uni["face_state"]["sunglasses"] = 0.99
        rec = build_record("x", uni, self._pyfeat(), self._l2cs(), {})
        self.assertFalse(rec["gaze_observability"]["eligible"])
        self.assertFalse(rec["gaze"]["publishable"])
        self.assertIn("sunglasses_obscure_eye_gaze", rec["gaze_observability"]["reasons"])

    def test_profile_one_good_eye_remains_high_authority(self):
        uni = self._base_uniface()
        uni["head_pose"]["yaw_deg_raw"] = -72.0
        uni["face_state"]["left_eye_open"] = 0.99
        uni["face_state"]["right_eye_open"] = 0.05
        uni["face_mesh"]["inter_iris_distance_px"] = 5.0
        uni["face_parsing"]["eye_region_diagnostics"]["iris_regions"] = [
            {"available": True, "fractions": {"hair": 0.0, "background": 0.0}},
            {"available": True, "fractions": {"hair": 0.0, "background": 0.95}},
        ]
        rec = build_record("x", uni, self._pyfeat(), self._l2cs(), {})
        self.assertTrue(rec["gaze_observability"]["eligible"])
        self.assertEqual(rec["gaze_observability"]["eye_expectation"], "one_clean_eye_sufficient")
        self.assertEqual(rec["head"]["frame_horizontal"], "frame_right")

    def test_frontal_single_weak_eye_reduces_but_does_not_null(self):
        uni = self._base_uniface()
        uni["face_state"]["right_eye_open"] = 0.1
        rec = build_record("x", uni, self._pyfeat(), self._l2cs(), {})
        self.assertTrue(rec["gaze_observability"]["eligible"])
        self.assertEqual(rec["gaze_observability"]["authority"], "reduced")

    def test_both_eye_regions_occluded_nulls_gaze(self):
        uni = self._base_uniface()
        uni["face_parsing"]["eye_region_diagnostics"]["iris_regions"] = [
            {"available": True, "fractions": {"hair": 0.9, "background": 0.0}},
            {"available": True, "fractions": {"hair": 0.7, "background": 0.2}},
        ]
        rec = build_record("x", uni, self._pyfeat(), self._l2cs(), {})
        self.assertFalse(rec["gaze_observability"]["eligible"])
        self.assertIn("all_predicted_eye_regions_occluded", rec["gaze_observability"]["reasons"])


if __name__ == "__main__":
    unittest.main()
