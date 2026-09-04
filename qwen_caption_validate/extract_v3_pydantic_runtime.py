from __future__ import annotations

"""x3p2 runtime entrypoint with structural hard gates and semantic warnings."""

from . import extract_v3_pydantic as base
from .extract_v3_models_runtime import ExtractWireV1Runtime


# The base runner deliberately resolves these through module globals.  Swap in
# the corrected runtime contract without changing the VLM-facing field layout.
base.ExtractWireV1 = ExtractWireV1Runtime

_base_expand_extract_wire = base.expand_extract_wire


def _expand_with_semantic_warnings(wire: ExtractWireV1Runtime):
    canonical, metadata = _base_expand_extract_wire(wire)
    warnings = list(metadata.get("warnings") or [])
    warnings.extend(wire.semantic_warnings())
    metadata["warnings"] = warnings
    return canonical, metadata


base.expand_extract_wire = _expand_with_semantic_warnings


if __name__ == "__main__":
    raise SystemExit(base.main())
