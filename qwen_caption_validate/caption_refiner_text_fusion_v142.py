from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import caption_refiner_text_fusion_v14 as v14
from . import caption_refiner_text_fusion_v141 as v141
from .caption_refiner_specialist_facts import _axis_reliable


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "caption_refiner_v142_text_fusion_guarded.txt"
ARTIFACT_VERSION = "caption-refiner-text-fusion-0.14.2"
RUN_VERSION = "caption-refiner-text-fusion-0.14.2-run"

_FRAME_TERM_RE = re.compile(
    r"(?:\b(?:image|frame|photo(?:graph)?|picture)[- ](?:left|right)\b|"
    r"\b(?:left|right)\s+(?:side|half)\s+of\s+(?:the\s+)?(?:image|frame|photo(?:graph)?|picture)\b|"
    r"\b(?:toward|towards|to|on)\s+(?:the\s+)?(?:left|right)\s+(?:side\s+)?of\s+(?:the\s+)?(?:image|frame|photo(?:graph)?|picture)\b)",
    re.IGNORECASE,
)
_CAMERA_ANY_RE = re.compile(
    r"\b(?:off[- ]camera|away\s+from\s+(?:the\s+)?camera|toward(?:s)?\s+(?:the\s+)?camera|"
    r"at\s+(?:the\s+)?camera|looking\s+(?:at|toward(?:s)?)\s+(?:the\s+)?camera|eye[- ]?contact)\b",
    re.IGNORECASE,
)
_CAMERA_OFF_RE = re.compile(r"\b(?:off[- ]camera|away\s+from\s+(?:the\s+)?camera)\b", re.IGNORECASE)
_CAMERA_TOWARD_RE = re.compile(
    r"\b(?:toward(?:s)?\s+(?:the\s+)?camera|at\s+(?:the\s+)?camera|"
    r"looking\s+(?:at|toward(?:s)?)\s+(?:the\s+)?camera|eye[- ]?contact)\b",
    re.IGNORECASE,
)
_GAZE_RE = re.compile(r"\b(?:gaze|gazing|eyes?|looking|looks?|eye[- ]?contact)\b", re.IGNORECASE)
_HEAD_RE = re.compile(r"\b(?:head|face|chin)\b", re.IGNORECASE)
_FRAME_HORIZONTAL_RE = re.compile(r"\b(?:image|frame)[- ](?:left|right)\b|\b(?:left|right)\s+side\s+of\s+the\s+frame\b", re.IGNORECASE)
_VERTICAL_RE = re.compile(r"\b(?:upward|upwards|downward|downwards|tilted\s+up|tilted\s+down|raised|lowered)\b", re.IGNORECASE)
_BODY_RE = re.compile(
    r"\b(?:torso|trunk|shoulders?|hips?|arms?|elbows?|wrists?|hands?|legs?|knees?|ankles?|feet|foot|upper\s+body)\b",
    re.IGNORECASE,
)

_JOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "shoulder": ("shoulder",),
    "elbow": ("elbow",),
    "wrist": ("wrist", "hand"),
    "hip": ("hip",),
    "knee": ("knee",),
    "ankle": ("ankle", "foot", "feet"),
}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _fact_is_explicitly_checkable(fact: dict[str, Any], reference_text: str) -> bool:
    anatomical = str(fact.get("anatomical_side") or "")
    joint = str(fact.get("joint") or "").lower()
    if anatomical not in {"subject-left", "subject-right"} or joint not in _JOINT_ALIASES:
        return False
    side = "left" if anatomical == "subject-left" else "right"
    aliases = _JOINT_ALIASES[joint]

    side_joint_patterns = [
        re.compile(rf"\b(?:her|his|their)\s+{side}\s+{re.escape(alias)}\b", re.IGNORECASE)
        for alias in aliases
    ]
    side_joint_patterns += [
        re.compile(rf"\bsubject[- ]{side}\s+{re.escape(alias)}\b", re.IGNORECASE)
        for alias in aliases
    ]
    side_joint_patterns += [
        re.compile(rf"\b{side}\s+{re.escape(alias)}\b", re.IGNORECASE)
        for alias in aliases
    ]

    for sentence in _sentences(reference_text):
        if not _FRAME_TERM_RE.search(sentence):
            continue
        if any(pattern.search(sentence) for pattern in side_joint_patterns):
            return True
    return False


def strict_caption_safe_laterality_facts(
    facts: Any,
    *,
    current_caption: str,
    accepted_pose_candidate: str | None,
) -> list[dict[str, Any]]:
    """Expose 2-D laterality only when there is an explicit frame-location claim to check."""
    if not isinstance(facts, list):
        return []
    reference = current_caption
    if accepted_pose_candidate:
        reference += " " + accepted_pose_candidate

    safe: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("inside_frame") is not True:
            continue
        if not _fact_is_explicitly_checkable(fact, reference):
            continue
        anatomical = str(fact.get("anatomical_side") or "")
        joint = str(fact.get("joint") or "")
        frame = str(fact.get("frame_side") or "")
        if anatomical not in {"subject-left", "subject-right"} or not joint:
            continue
        safe.append({
            "joint": joint,
            "anatomical_side": anatomical,
            "frame_side": frame,
            "source": fact.get("source"),
        })
    return safe


def _semantic_veto_reasons(
    correction: str,
    *,
    head_gaze: dict[str, Any],
    pose_gate: dict[str, Any],
    laterality_facts: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    head = head_gaze.get("head") if isinstance(head_gaze.get("head"), dict) else {}
    gaze = head_gaze.get("gaze") if isinstance(head_gaze.get("gaze"), dict) else {}

    if _GAZE_RE.search(correction):
        if not gaze.get("available") or not gaze.get("publishable"):
            reasons.append("gaze_language_without_publishable_gaze")

    camera = str(gaze.get("camera_relationship") or "")
    if _CAMERA_ANY_RE.search(correction):
        if camera not in {"toward_camera", "off_camera"}:
            reasons.append("camera_relationship_not_authoritative")
        elif _CAMERA_OFF_RE.search(correction) and camera != "off_camera":
            reasons.append("camera_relationship_direction_conflict")
        elif _CAMERA_TOWARD_RE.search(correction) and camera != "toward_camera":
            reasons.append("camera_relationship_direction_conflict")

    if _HEAD_RE.search(correction) and _FRAME_HORIZONTAL_RE.search(correction):
        if not head.get("available") or not _axis_reliable(head, "yaw"):
            reasons.append("head_yaw_language_without_reliable_yaw")

    if _HEAD_RE.search(correction) and _VERTICAL_RE.search(correction):
        if not head.get("available") or not _axis_reliable(head, "pitch"):
            reasons.append("head_pitch_language_without_reliable_pitch")

    if _BODY_RE.search(correction) and pose_gate.get("status") != "accepted" and not laterality_facts:
        reasons.append("body_geometry_without_governed_body_evidence")

    return sorted(set(reasons))


def _read_json(path: Path | None) -> dict[str, Any]:
    return v14._read_json(path)


def _write_json(path: Path, value: Any) -> None:
    v14._write_json(path, value)


def _resolve_output_dir(argv: list[str]) -> tuple[list[str], Path]:
    for i, value in enumerate(argv):
        if value == "--output-dir" and i + 1 < len(argv):
            return argv, Path(argv[i + 1]).expanduser().resolve()
        if value.startswith("--output-dir="):
            return argv, Path(value.split("=", 1)[1]).expanduser().resolve()
    if len(argv) < 2 or argv[1].startswith("-"):
        raise SystemExit("run_dir positional argument is required")
    output = Path(argv[1]).expanduser().resolve() / "caption-refiner-text-fusion-v0.14.2"
    return argv + ["--output-dir", str(output)], output


def _postprocess(output_dir: Path) -> None:
    old_index_path = output_dir / "caption_refiner_text_fusion_v141.index.json"
    index = _read_json(old_index_path)
    if not index:
        return

    index["schema_version"] = RUN_VERSION
    index["semantic_fail_closed"] = True
    records = index.get("records")
    if not isinstance(records, list):
        records = []

    for record in records:
        if not isinstance(record, dict):
            continue
        key = str(record.get("image_key") or "")
        per_path = output_dir / f"{key}.text_refiner.json"
        payload = _read_json(per_path)
        target = payload if payload else record

        model_delta = copy.deepcopy(target.get("correction_delta") or {})
        target["model_correction_delta"] = model_delta
        final_delta = copy.deepcopy(model_delta)
        veto_reasons: list[str] = []

        if model_delta.get("decision") == "CORRECTION":
            if not model_delta.get("contract_ok"):
                veto_reasons.append(f"model_contract:{model_delta.get('contract_reason') or 'invalid'}")
            text = str(model_delta.get("text") or "").strip()
            hg_path_raw = target.get("head_gaze_evidence_source")
            hg_path = Path(hg_path_raw) if isinstance(hg_path_raw, str) and hg_path_raw else None
            hg = _read_json(hg_path)
            gate = target.get("pose_candidate_gate") if isinstance(target.get("pose_candidate_gate"), dict) else {}
            laterality = target.get("laterality_facts_supplied") if isinstance(target.get("laterality_facts_supplied"), list) else []
            veto_reasons.extend(
                _semantic_veto_reasons(
                    text,
                    head_gaze=hg,
                    pose_gate=gate,
                    laterality_facts=laterality,
                )
            )

        if veto_reasons:
            final_delta = {
                "decision": "NO_CORRECTION",
                "text": "NO_CORRECTION",
                "contract_ok": True,
                "contract_reason": None,
                "semantic_vetoed": True,
                "semantic_veto_reasons": sorted(set(veto_reasons)),
            }
        else:
            final_delta["semantic_vetoed"] = False
            final_delta["semantic_veto_reasons"] = []

        target["schema_version"] = ARTIFACT_VERSION
        target["correction_delta"] = final_delta
        target["semantic_fail_closed"] = True
        if payload:
            _write_json(per_path, target)

        record.update({
            "schema_version": ARTIFACT_VERSION,
            "laterality_facts_supplied": target.get("laterality_facts_supplied", record.get("laterality_facts_supplied")),
            "laterality_text_supplied": target.get("laterality_text_supplied", record.get("laterality_text_supplied")),
            "model_correction_delta": model_delta,
            "correction_delta": final_delta,
            "semantic_fail_closed": True,
        })

    new_index_path = output_dir / "caption_refiner_text_fusion_v142.index.json"
    _write_json(new_index_path, index)


def main() -> int:
    original_argv = list(sys.argv)
    adjusted, output_dir = _resolve_output_dir(original_argv)
    sys.argv[:] = adjusted

    # Install v0.14.2 behavior only for this process; older experiment paths remain unchanged.
    v14.caption_safe_laterality_facts = strict_caption_safe_laterality_facts
    v141.DEFAULT_PROMPT = DEFAULT_PROMPT
    v141.ARTIFACT_VERSION = ARTIFACT_VERSION
    v141.RUN_VERSION = RUN_VERSION

    rc = v141.main()
    if rc == 0:
        _postprocess(output_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
