from __future__ import annotations

from pathlib import Path
from typing import Any

from . import compose_lint_repair_141 as _base
from .caption_projection_142 import lint_caption


REPAIR_PROMPT_142 = _base.REPAIR_PROMPT_141.replace(
    "Projection 1.4.1 semantic-compression and accessory-state governance",
    "Projection 1.4.2 pose-consistency, semantic-compression, and accessory-state governance",
).replace(
    "- Use natural sentences rather than one long comma-separated list.\n",
    "- If `upper_torso_depth_relation` is present, describe the torso/body as strongly turned in depth or near side-on; do not call it frontal/square-on.\n"
    "- If `head_torso_relation` is present, explicitly say the head/face turns toward the camera relative to the torso; do not use the ambiguous phrase `head faces forward`.\n"
    "- Do not recreate body-to-body contact/support that Fusion 2.3.5 removed.\n"
    "- Use natural sentences rather than one long comma-separated list.\n",
)

_ORIGINAL_WRITE_JSON = _base._ORIGINAL_WRITE_JSON


def _write_json_142(path: Path, value: dict[str, Any]) -> None:
    if path.name == "lint_repair.index.json":
        value = dict(value)
        value["governance_revision"] = "1.4.2"
    _ORIGINAL_WRITE_JSON(path, value)


def main() -> int:
    # Delegate through the existing one-shot repair implementation, but replace
    # the linter/prompt/index writer with Projection 1.4.2 behavior.
    _base._base._base.lint_caption = lint_caption
    _base._base._base.REPAIR_PROMPT = REPAIR_PROMPT_142
    _base._base._base._write_json = _write_json_142
    return _base._base._base.main()


if __name__ == "__main__":
    raise SystemExit(main())
