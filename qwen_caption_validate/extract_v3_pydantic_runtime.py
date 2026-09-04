from __future__ import annotations

"""x3p3 runtime entrypoint.

The base runner is reused for batching/performance/provenance. This module
swaps in the x3p3 Pydantic wire contract, fragment-aware deterministic expander,
prompt, version label and an isolated calibration output tree.

Revision ``extract-v3-pydantic.5`` keeps the x3p3 wire schema unchanged while
isolating the proximal-limb-chain prompt plus governance v0.3 from the prior
``extract-v3-pydantic.4`` calibration. The safety ceiling is 4500 tokens after
one fresh x3p3 control reached the former 4000-token ceiling mid-JSON.
"""

from . import extract_v3_pydantic as base
from .extract_v3_models_x3p3 import ExtractWireX3P3Runtime
from .extract_v3_wire_contract_x3p3 import expand_extract_wire


base.ExtractWireV1 = ExtractWireX3P3Runtime
base.expand_extract_wire = expand_extract_wire
base.WIRE_VERSION = "x3p3"
base.DEFAULT_PROMPT = base.PACKAGE_ROOT / "prompts" / "extract_v3_wire_x3p3.txt"
base.DEFAULT_MAX_TOKENS = 4500

_base_parse_args = base.parse_args


def _parse_args():
    args = _base_parse_args()
    # Keep every calibration revision isolated. The generic runner otherwise
    # reuses an existing artifact solely by path, so sharing a tree across prompt
    # revisions would make an old raw response look like a test of new guidance.
    if args.output_dir is None:
        run_dir = args.run_dir.expanduser().resolve()
        model_id = base.resolve_model_id(args.model)
        slug = base.model_slug(model_id)
        args.output_dir = run_dir / "extract-v3-pydantic.5" / slug
    return args


base.parse_args = _parse_args


if __name__ == "__main__":
    raise SystemExit(base.main())
