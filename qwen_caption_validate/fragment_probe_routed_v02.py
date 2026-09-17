from __future__ import annotations

from pathlib import Path

from . import fragment_probe_routed_v01 as base

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "fragment-probe-routed-v0.2"
DEFAULT_CONFIGURATION_PROMPT = PACKAGE_ROOT / "prompts" / "fragment_probe_route_configuration_v02.txt"
SCHEMA_VERSION = "fragment-probe-routed-0.2"


def main() -> int:
    old_policy = base.DEFAULT_POLICY_SUBDIR
    old_output = base.DEFAULT_OUTPUT_SUBDIR
    old_configuration_prompt = base.DEFAULT_CONFIGURATION_PROMPT
    old_schema = base.SCHEMA_VERSION
    try:
        base.DEFAULT_POLICY_SUBDIR = DEFAULT_POLICY_SUBDIR
        base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
        base.DEFAULT_CONFIGURATION_PROMPT = DEFAULT_CONFIGURATION_PROMPT
        base.SCHEMA_VERSION = SCHEMA_VERSION
        return base.main()
    finally:
        base.DEFAULT_POLICY_SUBDIR = old_policy
        base.DEFAULT_OUTPUT_SUBDIR = old_output
        base.DEFAULT_CONFIGURATION_PROMPT = old_configuration_prompt
        base.SCHEMA_VERSION = old_schema


if __name__ == "__main__":
    raise SystemExit(main())
