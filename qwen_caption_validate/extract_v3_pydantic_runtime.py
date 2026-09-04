from __future__ import annotations

"""x3p3 runtime entrypoint.

The base runner is reused for batching/performance/provenance.  This module
swaps in the x3p3 Pydantic wire contract, fragment-aware deterministic expander,
prompt, version label and isolated output tree.
"""

from . import extract_v3_pydantic as base
from .extract_v3_models_x3p3 import ExtractWireX3P3Runtime
from .extract_v3_wire_contract_x3p3 import expand_extract_wire


base.ExtractWireV1 = ExtractWireX3P3Runtime
base.expand_extract_wire = expand_extract_wire
base.WIRE_VERSION = "x3p3"
base.DEFAULT_PROMPT = base.PACKAGE_ROOT / "prompts" / "extract_v3_wire_x3p3.txt"

_base_parse_args = base.parse_args


def _parse_args():
    args = _base_parse_args()
    # Keep every wire calibration revision isolated.  This also overrides the
    # x3p2 default path hard-coded in the generic runner without changing old
    # artifacts or requiring a second copy of the batching implementation.
    if args.output_dir is None:
        run_dir = args.run_dir.expanduser().resolve()
        model_id = base.resolve_model_id(args.model)
        slug = base.model_slug(model_id)
        args.output_dir = run_dir / "extract-v3-pydantic.3" / slug
    return args


base.parse_args = _parse_args


if __name__ == "__main__":
    raise SystemExit(base.main())
