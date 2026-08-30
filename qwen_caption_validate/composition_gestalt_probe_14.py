from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from . import composition_gestalt_probe as base
from .runner import model_slug, resolve_model_id


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "composition_gestalt_v1_4.txt"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "composition_gestalt_v1_4.schema.json"

_BODY_SIDE_RE = re.compile(
    r"\b(?:left|right)\s+(?=(?:hand|fist|forearm|arm|elbow|wrist|shoulder|leg|knee|foot|hip|thigh)\b)",
    re.I,
)


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


def _scrub_body_laterality(text: Any) -> tuple[Any, bool]:
    if not isinstance(text, str):
        return text, False
    clean, count = _BODY_SIDE_RE.subn("", text)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean, bool(count)


def _govern_output_file(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    gestalt = payload.get("gestalt")
    if not isinstance(gestalt, dict):
        return

    scrubbed: list[dict[str, Any]] = []
    for index, item in enumerate(gestalt.get("salient_body_configuration") or []):
        if not isinstance(item, dict):
            continue
        original = item.get("description")
        clean, changed = _scrub_body_laterality(original)
        if changed:
            item["description"] = clean
            scrubbed.append({"field": f"salient_body_configuration[{index}].description", "original": original, "governed": clean})

    original_summary = gestalt.get("composition_summary")
    clean_summary, changed = _scrub_body_laterality(original_summary)
    if changed:
        gestalt["composition_summary"] = clean_summary
        scrubbed.append({"field": "composition_summary", "original": original_summary, "governed": clean_summary})

    support_audit: list[dict[str, Any]] = []
    for index, item in enumerate(gestalt.get("support_context") or []):
        if not isinstance(item, dict):
            continue
        ownership = item.get("target_ownership")
        evidence_status = item.get("evidence_status")
        eligible = ownership == "external_scene" and evidence_status in {"observed", "contextual"}
        support_audit.append({
            "index": index,
            "target": item.get("target"),
            "target_ownership": ownership,
            "evidence_status": evidence_status,
            "external_support_candidate": eligible,
        })

    orientation = gestalt.get("subject_orientation") if isinstance(gestalt.get("subject_orientation"), dict) else {}
    orientation_audit = {
        "body_orientation": orientation.get("body_orientation"),
        "body_faces_frame": orientation.get("body_faces_frame"),
        "body_confidence": orientation.get("body_confidence"),
        "torso_evidence_quality": orientation.get("torso_evidence_quality"),
        "head_relative_body": orientation.get("head_relative_body"),
        "posture_independent": True,
        "frame_direction_not_anatomical_laterality": True,
    }

    payload["gestalt"] = gestalt
    payload["governance_v14"] = {
        "anatomical_laterality_policy": "composition probe is side-neutral for body parts; body_faces_frame is image-frame direction only",
        "body_laterality_scrubbed": scrubbed,
        "support_target_policy": "support is not an external-support candidate unless target_ownership=external_scene and evidence_status is observed/contextual",
        "support_context_audit": support_audit,
        "subject_orientation_audit": orientation_audit,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    argv = list(sys.argv[1:])
    if not argv:
        return base.main()

    run_dir = Path(argv[0]).expanduser().resolve()
    model_arg = _option_value(argv, "--model", "32b-fp8") or "32b-fp8"
    slug = model_slug(resolve_model_id(model_arg))
    output_dir = Path(_option_value(argv, "--output-dir", str(run_dir / "composition-gestalt-v1.4" / slug)) or "").expanduser().resolve()

    if not _has_option(argv, "--prompt"):
        argv.extend(["--prompt", str(DEFAULT_PROMPT)])
    if not _has_option(argv, "--schema"):
        argv.extend(["--schema", str(DEFAULT_SCHEMA)])
    if not _has_option(argv, "--output-dir"):
        argv.extend(["--output-dir", str(output_dir)])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        rc = base.main()
    finally:
        sys.argv = old_argv

    if rc == 0 and output_dir.is_dir():
        for path in output_dir.glob("*.composition_gestalt.json"):
            _govern_output_file(path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
