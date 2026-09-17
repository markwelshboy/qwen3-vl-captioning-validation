from __future__ import annotations

from pathlib import Path

from . import fact_sheet_normalizer_v01 as base

# Routing/acquisition successor.  The typed Phase-4A schema itself is unchanged;
# only the authoritative production sources move from perception/body v0.1 to
# the validated v0.2 routing path.
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_BODY_SUBDIR = Path("semantic-v3") / "fragment-probe-routed-v0.2"
DEFAULT_GESTALT_SUBDIR = Path("semantic-v3") / "routed-gestalt-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.1-routed-v0.2"
SCHEMA_VERSION = base.SCHEMA_VERSION


def main() -> int:
    old = (
        base.DEFAULT_POLICY_SUBDIR,
        base.DEFAULT_BODY_SUBDIR,
        base.DEFAULT_GESTALT_SUBDIR,
        base.DEFAULT_OUTPUT_SUBDIR,
    )
    try:
        base.DEFAULT_POLICY_SUBDIR = DEFAULT_POLICY_SUBDIR
        base.DEFAULT_BODY_SUBDIR = DEFAULT_BODY_SUBDIR
        base.DEFAULT_GESTALT_SUBDIR = DEFAULT_GESTALT_SUBDIR
        base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
        return base.main()
    finally:
        (
            base.DEFAULT_POLICY_SUBDIR,
            base.DEFAULT_BODY_SUBDIR,
            base.DEFAULT_GESTALT_SUBDIR,
            base.DEFAULT_OUTPUT_SUBDIR,
        ) = old


if __name__ == "__main__":
    raise SystemExit(main())
