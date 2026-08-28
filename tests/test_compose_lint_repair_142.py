from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.compose_lint_repair_142 import REPAIR_PROMPT_142, _write_json_142


class ComposeLintRepair142Tests(unittest.TestCase):
    def test_repair_prompt_preserves_relative_pose_and_removed_contact(self) -> None:
        self.assertIn("upper_torso_depth_relation", REPAIR_PROMPT_142)
        self.assertIn("head_torso_relation", REPAIR_PROMPT_142)
        self.assertIn("Do not recreate body-to-body contact/support", REPAIR_PROMPT_142)
        self.assertIn("head faces forward", REPAIR_PROMPT_142)

    def test_repair_index_reports_projection_142(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lint_repair.index.json"
            _write_json_142(path, {"governance_revision": "1.4.1", "matched": 1})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["governance_revision"], "1.4.2")
        self.assertEqual(payload["matched"], 1)


if __name__ == "__main__":
    unittest.main()
