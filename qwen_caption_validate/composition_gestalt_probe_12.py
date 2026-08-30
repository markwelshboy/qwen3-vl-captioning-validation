from __future__ import annotations

import sys
from pathlib import Path

from . import composition_gestalt_probe as base
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "composition_gestalt_v1_2.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "composition_gestalt_v1_2.schema.json"


def _option_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def _has_option(argv: list[str], name: str) -> bool:
    return name in argv


def main() -> int:
    argv = list(sys.argv[1:])
    if not argv:
        # Delegate argparse's normal usage/error handling.
        return base.main()

    run_dir = Path(argv[0]).expanduser().resolve()
    model_arg = _option_value(argv, "--model", "32b-fp8") or "32b-fp8"
    slug = model_slug(resolve_model_id(model_arg))

    if not _has_option(argv, "--prompt"):
        argv.extend(["--prompt", str(DEFAULT_PROMPT)])
    if not _has_option(argv, "--schema"):
        argv.extend(["--schema", str(DEFAULT_SCHEMA)])
    if not _has_option(argv, "--output-dir"):
        argv.extend(["--output-dir", str(run_dir / "composition-gestalt-v1.2" / slug)])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        return base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
