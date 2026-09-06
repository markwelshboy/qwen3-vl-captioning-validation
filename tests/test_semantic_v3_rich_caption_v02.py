from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qwen_caption_validate import semantic_v3_rich_caption_v02 as v02


class RichCaptionV02Tests(unittest.TestCase):
    def test_version_and_prompt_are_v02(self) -> None:
        self.assertEqual(v02.ARTIFACT_VERSION, "semantic-v3-rich-caption-0.2")
        self.assertEqual(v02.RUN_VERSION, "semantic-v3-rich-caption-0.2-run")
        self.assertEqual(v02.DEFAULT_PROMPT.name, "semantic_v3_rich_caption_v02.txt")

    def test_default_output_is_v02_tree(self) -> None:
        argv = ["semantic_v3_rich_caption_v02", "/tmp/example-run", "--model", "32b-fp8"]
        with patch.object(sys, "argv", argv):
            v02._inject_default_output_dir()
            self.assertIn("--output-dir", sys.argv)
            output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
        self.assertEqual(output.parent.parent.name, "rich-caption-v0.2")
        self.assertEqual(output.parent.parent.parent.name, "semantic-v3")


if __name__ == "__main__":
    unittest.main()
