from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.compose_lint_repair_143 import REPAIR_PROMPT_143, _write_json_143


class ComposeLintRepair143Tests(unittest.TestCase):
    def test_repair_prompt_preserves_salient_pose_and_framing(self) -> None:
        self.assertIn("Preserve required framing/subject-extent wording", REPAIR_PROMPT_143)
        self.assertIn("do not invent knee bend/straightness", REPAIR_PROMPT_143)
        self.assertIn("chin-rest gesture", REPAIR_PROMPT_143)

    def test_repair_index_reports_projection_143(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lint_repair.index.json"
            _write_json_143(path, {"governance_revision": "1.4.2", "matched": 1})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["governance_revision"], "1.4.3")
        self.assertEqual(payload["matched"], 1)


if __name__ == "__main__":
    unittest.main()
