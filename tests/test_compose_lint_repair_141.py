from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.compose_lint_repair_141 import REPAIR_PROMPT_141, _write_json_141


class ComposeLintRepair141Tests(unittest.TestCase):
    def test_repair_prompt_preserves_accessory_state(self) -> None:
        self.assertIn("sunglasses perched on the head", REPAIR_PROMPT_141)
        self.assertIn("mask below the chin", REPAIR_PROMPT_141)
        self.assertIn("must not expand the caption into an evidence checklist", REPAIR_PROMPT_141)

    def test_repair_index_reports_projection_141(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lint_repair.index.json"
            _write_json_141(path, {"governance_revision": "1.4.0", "matched": 1})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["governance_revision"], "1.4.1")
        self.assertEqual(payload["matched"], 1)


if __name__ == "__main__":
    unittest.main()
