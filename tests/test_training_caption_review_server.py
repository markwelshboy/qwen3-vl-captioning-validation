from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.training_caption_review_server import (
    _export,
    _export_selected_packet,
)


class TrainingCaptionReviewServerTests(unittest.TestCase):
    def test_export_flattens_each_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            index = {
                "schema_version": "training-caption-review-bundle-0.1",
                "records": [
                    {
                        "image_key": "img1",
                        "pose": "sitting",
                        "captions": {
                            "compact": {"text": "Compact caption.", "word_count": 80, "repair_action": "passthrough"},
                            "medium": {"text": "Medium caption.", "word_count": 125, "repair_action": "deterministic"},
                        },
                    }
                ],
            }
            annotations = {
                "records": {
                    "img1": {
                        "reviewed": True,
                        "preferred_profile": "compact",
                        "overall_notes": "Compact is cleaner.",
                        "compact": {
                            "fit": "good",
                            "pose_language": "natural",
                            "issues": [],
                            "pipeline_fix": "no",
                            "notes": "",
                        },
                        "medium": {
                            "fit": "usable_with_issue",
                            "pose_language": "mostly_natural",
                            "issues": ["too_much_low_value_detail"],
                            "pipeline_fix": "no",
                            "notes": "A little busy.",
                        },
                    }
                }
            }
            json_path, csv_path = _export(bundle, index, annotations)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 2)
            self.assertEqual(payload["records"][0]["profile"], "compact")
            self.assertEqual(payload["records"][1]["profile"], "medium")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["issues"], "too_much_low_value_detail")

    def test_selected_packet_is_self_contained_and_keeps_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            (bundle / "media" / "original").mkdir(parents=True)
            (bundle / "media" / "overlay").mkdir(parents=True)
            (bundle / "records").mkdir(parents=True)
            (bundle / "media" / "original" / "img1.png").write_bytes(b"pngbytes")
            (bundle / "media" / "overlay" / "img1.webp").write_bytes(b"webpbytes")
            (bundle / "records" / "img1.json").write_text(json.dumps({"raw": True}), encoding="utf-8")
            index = {
                "records": [
                    {
                        "image_key": "img1",
                        "original": "media/original/img1.png",
                        "overlay": "media/overlay/img1.webp",
                        "raw_json": "records/img1.json",
                        "pose": "crouching",
                        "best_candidate_pose": "crouching",
                        "caption_ready_phrases": ["leans forward in a crouch"],
                        "captions": {
                            "compact": {"text": "Compact caption.", "word_count": 80},
                            "medium": {"text": "Medium caption.", "word_count": 130},
                        },
                    }
                ]
            }
            annotations = {
                "schema_version": "training-caption-review-annotations-0.1",
                "records": {
                    "img1": {
                        "reviewed": True,
                        "preferred_profile": "medium",
                        "compact": {"fit": "good", "pose_language": "natural", "issues": [], "pipeline_fix": "no"},
                        "medium": {"fit": "excellent", "pose_language": "natural", "issues": [], "pipeline_fix": "no"},
                    }
                },
            }
            (bundle / "training_caption_review.index.json").write_text(json.dumps(index), encoding="utf-8")
            (bundle / "training_caption_review_annotations.json").write_text(json.dumps(annotations), encoding="utf-8")

            html_path, json_path = _export_selected_packet(bundle, ["img1"])
            rendered = html_path.read_text(encoding="utf-8")
            packet = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("data:image/png;base64,", rendered)
            self.assertIn("DWPose + SAM3D", rendered)
            self.assertIn("Compact caption.", rendered)
            self.assertEqual(packet["records"][0]["annotation"]["preferred_profile"], "medium")


if __name__ == "__main__":
    unittest.main()
