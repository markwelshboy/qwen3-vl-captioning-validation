from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.training_caption_review_bundle import build_bundle


class TrainingCaptionReviewBundleTests(unittest.TestCase):
    def _write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_bundle_pairs_final_captions_with_pose_review_visuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            captions = root / "captions"
            pose_review = root / "pose-review"
            pose_language = root / "pose-language"
            output = root / "caption-review"
            key = "imageblind-01_00001"

            run_dir.mkdir()
            (pose_review / "media" / "original").mkdir(parents=True)
            (pose_review / "media" / "overlay").mkdir(parents=True)
            (pose_review / "media" / "original" / f"{key}.png").write_bytes(b"original")
            (pose_review / "media" / "overlay" / f"{key}.overlay.webp").write_bytes(b"overlay")
            self._write_json(
                pose_review / "pose_review.index.json",
                {
                    "schema_version": "pose-review-bundle-0.11",
                    "legend": {"dwpose": "#36d7ff", "sam3d": "#ffbf3f"},
                    "records": [
                        {
                            "image_key": key,
                            "original": f"media/original/{key}.png",
                            "overlay": f"media/overlay/{key}.overlay.webp",
                            "pose": "sitting",
                            "best_candidate_pose": "sitting",
                            "crop_support_percent": 82,
                            "reconstruction_match_percent": 91,
                            "posture_modifier_diagnostic": {"lean_severity": "moderate"},
                        }
                    ],
                },
            )
            self._write_json(
                pose_language / f"{key}.pose_language.json",
                {"caption_ready_phrases": ["leans forward while seated"]},
            )
            for profile, words in (("compact", 82), ("medium", 128)):
                self._write_json(
                    captions / f"{key}.{profile}.semantic_repaired_caption.json",
                    {
                        "image_key": key,
                        "profile": profile,
                        "final_caption": f"V3SUBJ final {profile} caption.",
                        "repair_action": "passthrough",
                        "quality_audit": {
                            "rendered_word_count": words,
                            "passes_basic_gate": True,
                        },
                        "final_semantic_warnings": [],
                    },
                )

            index = build_bundle(
                run_dir=run_dir,
                captions_dir=captions,
                pose_review_dir=pose_review,
                pose_language_dir=pose_language,
                output=output,
                overwrite=True,
            )

            self.assertEqual(index["record_count"], 1)
            self.assertEqual(index["missing"], [])
            record = index["records"][0]
            self.assertEqual(record["pose"], "sitting")
            self.assertEqual(record["caption_ready_phrases"], ["leans forward while seated"])
            self.assertEqual(record["captions"]["compact"]["word_count"], 82)
            self.assertEqual(record["captions"]["medium"]["word_count"], 128)
            self.assertTrue((output / record["original"]).is_file())
            self.assertTrue((output / record["overlay"]).is_file())
            raw = json.loads((output / record["raw_json"]).read_text(encoding="utf-8"))
            self.assertEqual(raw["compact"]["final_caption"], "V3SUBJ final compact caption.")
            self.assertTrue((output / "training_caption_review_annotations.json").is_file())

    def test_missing_caption_pair_is_reported_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            captions = root / "captions"
            pose_review = root / "pose-review"
            pose_language = root / "pose-language"
            output = root / "caption-review"
            key = "imageblind-01_00002"
            run_dir.mkdir()
            captions.mkdir()
            pose_language.mkdir()
            (pose_review / "media" / "original").mkdir(parents=True)
            (pose_review / "media" / "overlay").mkdir(parents=True)
            self._write_json(
                pose_review / "pose_review.index.json",
                {"records": [{"image_key": key, "original": "x.png", "overlay": "x.webp"}]},
            )
            self._write_json(
                captions / f"{key}.compact.semantic_repaired_caption.json",
                {"final_caption": "V3SUBJ compact."},
            )

            index = build_bundle(
                run_dir=run_dir,
                captions_dir=captions,
                pose_review_dir=pose_review,
                pose_language_dir=pose_language,
                output=output,
            )
            self.assertEqual(index["record_count"], 0)
            self.assertEqual(index["missing"][0]["reason"], "missing_final_compact_or_medium_caption")


if __name__ == "__main__":
    unittest.main()
