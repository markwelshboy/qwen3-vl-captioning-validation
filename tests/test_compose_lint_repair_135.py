from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.compose_lint_repair_135 import (
    _export_final_captions,
    _needs_repair,
    _render_repair_prompt,
    _safe_relative_caption_path,
)


class ComposeLintRepair135Tests(unittest.TestCase):
    def test_violation_always_requires_repair(self) -> None:
        lint = {"violation_count": 1, "warning_count": 0}
        self.assertTrue(_needs_repair(lint, include_warnings=False))

    def test_warning_repair_is_configurable(self) -> None:
        lint = {"violation_count": 0, "warning_count": 1}
        self.assertTrue(_needs_repair(lint, include_warnings=True))
        self.assertFalse(_needs_repair(lint, include_warnings=False))

    def test_clean_caption_is_not_repaired(self) -> None:
        lint = {"violation_count": 0, "warning_count": 0}
        self.assertFalse(_needs_repair(lint, include_warnings=True))

    def test_repair_prompt_contains_original_evidence_and_findings(self) -> None:
        caption = "TOKEN says the right shoulder is visible."
        evidence = {
            "caption_policy": {"trigger_token": "TOKEN"},
            "pose_orientation": {
                "qualified_laterality": [{"side": "left", "body_family": "shoulder"}]
            },
        }
        lint = {
            "violation_count": 1,
            "violations": [
                {
                    "type": "unqualified_anatomical_laterality",
                    "text": "right shoulder",
                }
            ],
        }
        prompt = _render_repair_prompt(caption, evidence, lint)
        self.assertIn(caption, prompt)
        self.assertIn("unqualified_anatomical_laterality", prompt)
        self.assertIn('"side": "left"', prompt)
        self.assertIn("Do not add or reconstruct visual facts", prompt)
        self.assertIn("Never infer the complementary anatomical side", prompt)

    def test_export_path_uses_original_image_relative_stem(self) -> None:
        self.assertEqual(
            _safe_relative_caption_path("nested/example.webp", "nested__example"),
            Path("nested/example.txt"),
        )

    def test_export_path_rejects_parent_traversal(self) -> None:
        self.assertEqual(
            _safe_relative_caption_path("../outside.png", "safe_key"),
            Path("safe_key.txt"),
        )

    def test_final_caption_export_preserves_relative_layout_and_review_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            target_dir = root / "target"
            export_dir = root / "captions"
            source_dir.mkdir()
            target_dir.mkdir()

            source_meta = source_dir / "nested__example.fusion-safe.json"
            source_meta.write_text("{}\n", encoding="utf-8")
            final_meta = {
                "image": "nested/example.png",
                "caption": "TOKEN stands in a tree-lined park setting.",
                "repair_attempted": True,
                "caption_lint": {
                    "passed": True,
                    "warning_count": 0,
                    "violation_count": 0,
                },
            }
            (target_dir / source_meta.name).write_text(
                json.dumps(final_meta),
                encoding="utf-8",
            )

            index = _export_final_captions(target_dir, [source_meta], export_dir)

            self.assertEqual(
                (export_dir / "nested/example.txt").read_text(encoding="utf-8"),
                "TOKEN stands in a tree-lined park setting.\n",
            )
            self.assertEqual(index["written"], 1)
            self.assertEqual(index["review_required"], 0)
            self.assertEqual(index["records"][0]["status"], "accepted")
            self.assertTrue((export_dir / "caption_export.index.json").is_file())


if __name__ == "__main__":
    unittest.main()
