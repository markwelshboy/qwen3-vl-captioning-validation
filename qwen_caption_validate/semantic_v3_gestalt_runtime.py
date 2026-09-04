from __future__ import annotations

from typing import Any

from . import semantic_v3_gestalt as base
from .runner import validate_analysis as strict_validate_analysis
from .semantic_v3_gestalt_normalize import normalize_gestalt_representation


_ORIGINAL_GOVERN_GESTALT = base.govern_gestalt


def _validate_with_representation_normalization(
    payload: dict[str, Any] | None,
    schema: dict[str, Any],
) -> list[str]:
    """Validate the canonical representation without mutating raw model output."""
    if isinstance(payload, dict):
        payload, _ = normalize_gestalt_representation(payload)
    return strict_validate_analysis(payload, schema)


def _govern_with_representation_normalization(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized, normalization = normalize_gestalt_representation(payload)
    governed, governance = _ORIGINAL_GOVERN_GESTALT(normalized)
    governance["representation_normalization"] = normalization
    return governed, governance


def install_runtime_adapter() -> None:
    """Install narrow runtime adapters around the stable Gestalt v1.4 implementation."""
    base.validate_analysis = _validate_with_representation_normalization
    base.govern_gestalt = _govern_with_representation_normalization


def main() -> int:
    install_runtime_adapter()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
