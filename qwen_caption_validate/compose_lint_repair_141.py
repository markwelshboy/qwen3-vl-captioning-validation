from __future__ import annotations

from pathlib import Path
from typing import Any

from . import compose_lint_repair_140 as _base
from .caption_projection_141 import lint_caption


REPAIR_PROMPT_141 = _base.REPAIR_PROMPT_140.replace(
    "Projection 1.4.0 semantic-compression governance",
    "Projection 1.4.1 semantic-compression and accessory-state governance",
).replace(
    "- Use natural sentences rather than one long comma-separated list.\n",
    "- Preserve explicit transient accessory states such as sunglasses perched on the head or a mask below the chin; do not flatten them back to bare item names.\n"
    "- Use natural sentences rather than one long comma-separated list.\n",
)

_ORIGINAL_WRITE_JSON = _base._ORIGINAL_WRITE_JSON


def _write_json_141(path: Path, value: dict[str, Any]) -> None:
    if path.name == "lint_repair.index.json":
        value = dict(value)
        value["governance_revision"] = "1.4.1"
    _ORIGINAL_WRITE_JSON(path, value)


def main() -> int:
    # compose_lint_repair_140 delegates into the 1.3.5 implementation, so patch
    # the underlying globals directly rather than calling 1.4.0's main(), which
    # would restore the 1.4.0 linter.
    _base._base.lint_caption = lint_caption
    _base._base.REPAIR_PROMPT = REPAIR_PROMPT_141
    _base._base._write_json = _write_json_141
    return _base._base.main()


if __name__ == "__main__":
    raise SystemExit(main())
