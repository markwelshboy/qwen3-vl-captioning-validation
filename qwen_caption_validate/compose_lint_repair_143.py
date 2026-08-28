from __future__ import annotations

from pathlib import Path
from typing import Any

from . import compose_lint_repair_142 as _base
from .caption_projection_143 import lint_caption


REPAIR_PROMPT_143 = _base.REPAIR_PROMPT_142.replace(
    "Projection 1.4.2 pose-consistency, semantic-compression, and accessory-state governance",
    "Projection 1.4.3 semantic-salience, framing, and pose-consistency governance",
).replace(
    "- Use natural sentences rather than one long comma-separated list.\n",
    "- Preserve required framing/subject-extent wording and high-confidence pose interactions; repair must not silently drop them unless the lint finding specifically invalidates them.\n"
    "- If standing is qualified, retain standing even when feet are cropped, but do not invent knee bend/straightness or exact foot/ground contact when those details are absent from governed evidence.\n"
    "- If a chin-rest gesture is governed, describe the recognizable chin-on-hand/fist gesture naturally rather than as finger geometry.\n"
    "- Use natural sentences rather than one long comma-separated list.\n",
)

_ORIGINAL_WRITE_JSON = _base._ORIGINAL_WRITE_JSON


def _write_json_143(path: Path, value: dict[str, Any]) -> None:
    if path.name == "lint_repair.index.json":
        value = dict(value)
        value["governance_revision"] = "1.4.3"
    _ORIGINAL_WRITE_JSON(path, value)


def main() -> int:
    _base._base._base._base.lint_caption = lint_caption
    _base._base._base._base.REPAIR_PROMPT = REPAIR_PROMPT_143
    _base._base._base._base._write_json = _write_json_143
    return _base._base._base._base.main()


if __name__ == "__main__":
    raise SystemExit(main())
