from __future__ import annotations

import re

from . import semantic_v3_compact_renderer_semantic_repair_v02 as base


ARTIFACT_VERSION = "semantic-v3-compact-renderer-semantic-repair-0.3"
RUN_VERSION = "semantic-v3-compact-renderer-semantic-repair-0.3-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-semantic-repair-v0.3"


def _cleanup_after_local_deletion(text: str) -> str:
    """Turn a surviving hair participle into a finite verb after leak deletion.

    Example:
      Their tousled hair [falls loosely around their shoulders], partially framing ...
    becomes:
      Their tousled hair partially frames ...

    This keeps the transient face-framing state while removing only the protected
    length/location implication and avoids leaving a sentence fragment.
    """
    text = re.sub(
        r"\bhair\s*,\s*partially\s+framing\b",
        "hair partially frames",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bhair\s*,\s*framing\b",
        "hair frames",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# Patch only the narrow local grammar function and provenance constants. All other
# deterministic-first/fallback behavior remains v0.2.
base._cleanup_after_local_deletion = _cleanup_after_local_deletion
base.ARTIFACT_VERSION = ARTIFACT_VERSION
base.RUN_VERSION = RUN_VERSION
base.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR

# Re-export the calibrated helpers for tests/reuse.
deterministic_local_repair = base.deterministic_local_repair
build_model_repair_input = base.build_model_repair_input


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
