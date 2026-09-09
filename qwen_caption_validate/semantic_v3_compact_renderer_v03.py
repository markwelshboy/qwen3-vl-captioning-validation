from __future__ import annotations

from . import semantic_v3_compact_renderer_v01 as v01
from . import semantic_v3_compact_renderer_v02 as v02


ARTIFACT_VERSION = "semantic-v3-compact-renderer-0.3"
RUN_VERSION = "semantic-v3-compact-renderer-0.3-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-v0.3"
# v0.2's 220-token ceiling truncated 5/87 medium blind renders.  v0.3 changes
# only the generation ceiling; the calibrated renderer prompt/policy is unchanged.
DEFAULT_MAX_TOKENS = 320


def main() -> int:
    v02.ARTIFACT_VERSION = ARTIFACT_VERSION
    v02.RUN_VERSION = RUN_VERSION
    v02.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    v02.DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS

    # v0.2 main delegates mechanics to v0.1, so keep the delegated globals aligned.
    v01.DEFAULT_MAX_TOKENS = DEFAULT_MAX_TOKENS
    return v02.main()


if __name__ == "__main__":
    raise SystemExit(main())
