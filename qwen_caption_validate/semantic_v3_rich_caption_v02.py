from __future__ import annotations

import sys
from pathlib import Path

from . import semantic_v3_rich_caption as v01
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "semantic_v3_rich_caption_v02.txt"
ARTIFACT_VERSION = "semantic-v3-rich-caption-0.2"
RUN_VERSION = "semantic-v3-rich-caption-0.2-run"
DEFAULT_OUTPUT_SUBDIR = "rich-caption-v0.2"


def _inject_default_output_dir() -> None:
    """Keep v0.2 artifacts isolated without changing the calibrated v0.1 runtime."""
    if "--output-dir" in sys.argv or len(sys.argv) < 2:
        return
    run_dir = Path(sys.argv[1]).expanduser().resolve()
    model_arg = "32b-fp8"
    if "--model" in sys.argv:
        index = sys.argv.index("--model")
        if index + 1 < len(sys.argv):
            model_arg = sys.argv[index + 1]
    slug = model_slug(resolve_model_id(model_arg))
    output_dir = run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / slug
    sys.argv.extend(["--output-dir", str(output_dir)])


def main() -> int:
    _inject_default_output_dir()
    v01.DEFAULT_PROMPT = DEFAULT_PROMPT
    v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    v01.RUN_VERSION = RUN_VERSION
    return v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
