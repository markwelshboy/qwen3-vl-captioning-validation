from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.compose_lint_repair_140 import REPAIR_PROMPT_140, _write_json_140


class ComposeLintRepair140Tests(unittest.TestCase):
    def test_repair_prompt_preserves_semantic_compression(self) -> None:
        self.assertIn("must not expand the caption into an evidence checklist", REPAIR_PROMPT_140)
        self.assertIn("preferred_scene_entities", REPAIR_PROMPT_140)
        self.assertIn("forearm supports the same target via the hand", REPAIR_PROMPT_140)

    def test_repair_index_reports_projection_140(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lint_repair.index.json"
            _write_json_140(path, {"governance_revision": "1.3.5", "matched": 1})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["governance_revision"], "1.4.0")
        self.assertEqual(payload["matched"], 1)


if __name__ == "__main__":
    unittest.main()
