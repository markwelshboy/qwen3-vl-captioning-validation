from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import semantic_v3_gestalt as base
from .runner import model_slug, resolve_model_id
from .semantic_v3_gestalt_runtime import install_runtime_adapter


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_V02 = PACKAGE_ROOT / "prompts" / "semantic_v3_gestalt_from_extract_v0_2.txt"
OUTPUT_VERSION_V02 = "semantic-v3-gestalt-from-extract-0.2"
OUTPUT_SUBDIR_V02 = "gestalt-from-extract-v0.2"
INDEX_VERSION_V02 = "semantic-v3-gestalt-from-extract-run-0.2"

_ORIGINAL_PARSE_ARGS = base.parse_args


def _copy_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def build_gestalt_evidence_v02(extract: dict[str, Any]) -> dict[str, Any]:
    """Project observation-bearing Extract fields only.

    Extract.hypotheses is intentionally excluded. Gestalt must aggregate the immutable
    observations rather than echoing Extract's own camera/posture/orientation/support
    interpretations back to Fusion as a second-looking semantic result.
    """
    target = extract.get("target_subject") if isinstance(extract.get("target_subject"), dict) else {}
    scene = extract.get("scene") if isinstance(extract.get("scene"), dict) else {}

    return {
        "source_schema_version": extract.get("schema_version"),
        "projection_policy": "observation_only; Extract hypotheses intentionally omitted",
        "framing": _copy_dict(extract.get("framing")),
        "subject_evidence": {
            "visible_body_parts": _copy_list(target.get("visible_body_parts")),
            "geometry_landmark_visibility": _copy_dict(target.get("geometry_landmark_visibility")),
            "orientation_cues": _copy_list(target.get("orientation_cues")),
            "gaze": _copy_dict(target.get("gaze")),
            "interactions": _copy_list(target.get("interactions")),
        },
        "entities": _copy_list(extract.get("entities")),
        "relations": _copy_list(extract.get("relations")),
        "scene": {
            "environment_candidate": deepcopy(scene.get("environment_candidate")),
            "background_regions": _copy_list(scene.get("background_regions")),
        },
        "composition_observations": _copy_list(extract.get("composition_observations")),
        "uncertainties": _copy_list(extract.get("uncertainties")),
    }


def _parse_args_v02():
    args = _ORIGINAL_PARSE_ARGS()
    if args.output_dir is None:
        model_id = resolve_model_id(args.model)
        slug = model_slug(model_id)
        args.output_dir = args.run_dir.expanduser().resolve() / "semantic-v3" / OUTPUT_SUBDIR_V02 / slug
    return args


def install_v02() -> None:
    base.DEFAULT_PROMPT = DEFAULT_PROMPT_V02
    base.OUTPUT_VERSION = OUTPUT_VERSION_V02
    base.build_gestalt_evidence = build_gestalt_evidence_v02
    base.parse_args = _parse_args_v02


def _refresh_index_v02(output_dir: Path) -> None:
    index_path = output_dir / "gestalt_from_extract.index.json"
    if not index_path.exists():
        return
    value = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        return
    value["schema_version"] = INDEX_VERSION_V02
    value["evidence_policy"] = (
        "one image-conditioned Extract pass; Gestalt consumes observation-bearing canonical Extract fields only; "
        "Extract hypotheses are intentionally omitted"
    )
    value["gestalt_projection_version"] = "semantic-v3-gestalt-observation-projection-0.2"
    index_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    install_runtime_adapter()
    install_v02()
    result = base.main()

    # Reconstruct the default/explicit output directory from the now-patched parser inputs
    # without loading a model. base.main has already written the index at this point.
    # The output tree itself is version-isolated, so a best-effort scan is sufficient here.
    # Explicit --output-dir callers remain supported by base.main.
    import sys

    argv = sys.argv[1:]
    if argv:
        run_dir = Path(argv[0]).expanduser().resolve()
        candidates = list((run_dir / "semantic-v3" / OUTPUT_SUBDIR_V02).glob("*/gestalt_from_extract.index.json"))
        for index_path in candidates:
            _refresh_index_v02(index_path.parent)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
