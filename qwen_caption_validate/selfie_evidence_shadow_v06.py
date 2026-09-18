from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import camera_composition_shadow_v01 as camera_shadow
from . import selfie_evidence_shadow_v02 as selfie_v02
from . import selfie_evidence_shadow_v03 as selfie_v03
from . import selfie_evidence_shadow_v05 as selfie_v05

SCHEMA_VERSION = "selfie-evidence-shadow-0.6"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_FRAMING_SUBDIR = Path("semantic-v3") / "framing-semantics-v1.0"
DEFAULT_GESTALT_SUBDIR = Path("semantic-v3") / "routed-gestalt-v0.2"
DEFAULT_MESH_ARM_SUBDIR = Path("semantic-v3") / "mesh-arm-occupancy-shadow-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.6"

_EXPLICIT_MIRROR_SELFIE_RE = re.compile(
    r"\bmirror[- ]selfie\b"
    r"|\bselfie(?:[- ]style)?\s+(?:photo\s+)?(?:in|using|through)\s+(?:a\s+|the\s+)?mirror\b"
    r"|\bselfie[- ]style\s+mirror\s+(?:reflection|photo|image)\b",
    re.I,
)
_GENERIC_SELFIE_RE = re.compile(
    r"\bselfie\b|\bselfie[- ]style\b|\bself[- ]portrait\b",
    re.I,
)
_MIRROR_RE = re.compile(
    r"\bmirror\b|\bmirrored\b|\breflection\b|\breflective\s+(?:surface|wall|panel)\b",
    re.I,
)
_PHONE_RE = re.compile(
    r"\bsmartphone\b|\bcell(?:ular)?\s+phone\b|\bmobile\s+phone\b|\bphone\b",
    re.I,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _flatten_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        text = " ".join(value.split())
        if text:
            out.append(text)
    elif isinstance(value, list):
        for item in value:
            out.extend(_flatten_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(_flatten_strings(item))
    return out


def _neutral_mirror_selfie_semantic(
    gestalt_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(gestalt_record, dict):
        return {
            "grade": "none",
            "supported": False,
            "reason": "neutral_gestalt_record_missing",
            "explicit_mirror_selfie_text": [],
            "generic_selfie_text": [],
            "mirror_text": [],
            "phone_text": [],
        }

    acquisition = (
        gestalt_record.get("acquisition")
        if isinstance(gestalt_record.get("acquisition"), dict)
        else {}
    )
    strings = _flatten_strings(acquisition)

    explicit = [s for s in strings if _EXPLICIT_MIRROR_SELFIE_RE.search(s)]
    generic_selfie = [s for s in strings if _GENERIC_SELFIE_RE.search(s)]
    mirror = [s for s in strings if _MIRROR_RE.search(s)]
    phone = [s for s in strings if _PHONE_RE.search(s)]

    if explicit:
        grade = "strong"
        supported = True
        reason = "explicit_mirror_selfie_language_in_neutral_gestalt"
    elif generic_selfie and mirror:
        grade = "strong"
        supported = True
        reason = "neutral_selfie_semantic_plus_independent_mirror_or_reflection_semantic"
    elif mirror and phone:
        grade = "moderate"
        supported = False
        reason = (
            "mirror_or_reflection_plus_phone_present_but_no_neutral_selfie_semantic; "
            "withhold_for_precision"
        )
    elif mirror:
        grade = "weak"
        supported = False
        reason = "mirror_or_reflection_semantic_without_selfie_semantic"
    else:
        grade = "none"
        supported = False
        reason = "no_neutral_mirror_selfie_evidence"

    return {
        "grade": grade,
        "supported": supported,
        "explicit_mirror_selfie_text": explicit,
        "generic_selfie_text": generic_selfie,
        "mirror_text": mirror,
        "phone_text": phone,
        "reason": reason,
        "authority": "unprompted_qwen_gestalt_mirror_semantic",
        "precision_policy": (
            "Only explicit mirror-selfie language, or neutral selfie language plus "
            "an independent mirror/reflection cue, is publishable in v0.6."
        ),
    }


def _mirror_surface_policy(mirror_supported: bool) -> dict[str, Any]:
    if not mirror_supported:
        return {
            "active": False,
            "spatial_surface_mode": None,
        }

    return {
        "active": True,
        "spatial_surface_mode": "depicted_frame",
        "directional_reference_system": "frame_relative_only",
        "composer_anatomical_laterality": "withhold",
        "composer_reflection_physical_laterality": "withhold",
        "allowed_directional_terms": [
            "frame left",
            "frame right",
            "upper frame",
            "lower frame",
            "foreground",
            "background",
        ],
        "surface_rules": [
            "Describe the reflected image as depicted in the training frame.",
            "Use frame-left/frame-right for directional limb, phone and body placement.",
            "Do not expose anatomical left/right for hands, arms or shoulders by default.",
            "Do not ask the composer to invert a reflection or recover physical-world laterality.",
            "Torso turn direction may remain frame-relative.",
            "Internal anatomical geometry may still be retained for consistency checks.",
        ],
        "note": (
            "Mirror-selfie captioning uses one visible-image coordinate system to avoid "
            "mixing anatomical, physical-camera and reflected-frame laterality."
        ),
    }


def _capture_style(
    mirror: dict[str, Any],
    direct_decision: dict[str, Any],
) -> dict[str, Any]:
    if mirror.get("supported"):
        return {
            "family": "selfie",
            "subtype": "mirror_selfie",
            "composer_text": "mirror selfie",
            "authority": "neutral_mirror_selfie_semantic",
            "spatial_surface_mode": "depicted_frame",
        }
    if direct_decision.get("publishable_selfie"):
        return {
            "family": "selfie",
            "subtype": "direct_selfie",
            "composer_text": "selfie-style capture",
            "authority": "multi_family_direct_selfie_evidence",
            "spatial_surface_mode": "direct_camera_frame",
        }
    return {
        "family": None,
        "subtype": None,
        "composer_text": None,
        "authority": None,
        "spatial_surface_mode": None,
    }


def evaluate(
    policy: dict[str, Any],
    framing: dict[str, Any],
    gestalt: dict[str, Any] | None,
    mesh_arm: dict[str, Any] | None,
    arrays: dict[str, Any],
    dwpose: dict[str, Any],
) -> dict[str, Any]:
    geometry = camera_shadow.build_subject_geometry(arrays, dwpose)
    camera_viewpoint = camera_shadow._camera_viewpoint(geometry)
    shoulder = camera_shadow._shoulder_depth(arrays, dwpose)
    portrait = camera_shadow._portrait_context(framing, policy)

    neutral = selfie_v02._neutral_selfie_semantic(gestalt)
    mirror = _neutral_mirror_selfie_semantic(gestalt)
    camera = selfie_v02._camera_evidence(camera_viewpoint)
    partition = selfie_v05._foreground_arm_partition(mesh_arm, shoulder)
    direct_arm = partition["selfie_arm_evidence"]

    direct_decision = selfie_v03._decision(
        neutral,
        camera,
        direct_arm,
        shoulder,
        portrait,
    )
    direct_decision["decision_rule_version"] = (
        "v05_direct_selfie_separate_composition_arm_from_selfie_arm"
    )

    capture_style = _capture_style(mirror, direct_decision)
    mirror_policy = _mirror_surface_policy(bool(mirror.get("supported")))

    if mirror.get("supported"):
        final_decision = {
            "status": "selfie_supported",
            "publishable_selfie": True,
            "capture_subtype": "mirror_selfie",
            "reason": mirror.get("reason"),
            "decision_rule_version": "v06_high_precision_neutral_mirror_semantic",
            "promoted_fact_candidates": ["mirror selfie"],
            "direct_selfie_decision_retained_for_diagnostics": direct_decision,
        }
    else:
        final_decision = {
            **direct_decision,
            "capture_subtype": (
                "direct_selfie"
                if direct_decision.get("publishable_selfie")
                else None
            ),
            "mirror_candidate_grade": mirror.get("grade"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": policy.get("image_key"),
        "portrait_context": portrait,
        "capture_style": capture_style,
        "mirror_caption_surface_policy": mirror_policy,
        "evidence": {
            "neutral_semantic": neutral,
            "mirror_selfie_semantic": mirror,
            "camera_viewpoint": camera,
            "foreground_arm": direct_arm,
            "foreground_arm_composition": {
                "candidates": partition["composition_arm_candidates"],
                "all_candidates": partition["all_arm_candidates"],
                "role": "visible_composition_facts_not_automatically_selfie_evidence",
            },
            "nearer_shoulder": {
                **shoulder,
                "role": (
                    "internal_geometry_only_for_mirror_selfie"
                    if mirror.get("supported")
                    else "supportive_context_and_selfie_arm_binding_anchor"
                ),
            },
        },
        "decision": final_decision,
        "direct_selfie_diagnostic": direct_decision,
        "invariants": {
            "dedicated_selfie_or_mirror_prompt_is_not_used": True,
            "mirror_selfie_requires_high_precision_neutral_semantic_evidence": True,
            "mirror_plus_phone_without_selfie_semantic_is_not_publishable": True,
            "mirror_selfie_surface_uses_depicted_frame_coordinates_only": True,
            "mirror_selfie_anatomical_laterality_remains_internal": True,
            "mirror_selfie_does_not_require_direct_selfie_camera_geometry": True,
            "mirror_selfie_does_not_require_foreground_selfie_arm_geometry": True,
            "direct_selfie_rule_is_unchanged_from_v05": True,
            "misses_are_preferred_to_mirror_false_positives": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Selfie evidence shadow with separate high-precision mirror-selfie "
            "semantics and frame-relative mirror caption policy."
        )
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--framing-dir", type=Path)
    p.add_argument("--gestalt-dir", type=Path)
    p.add_argument("--mesh-arm-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = (
        args.policy_dir.expanduser().resolve()
        if args.policy_dir
        else run_dir / DEFAULT_POLICY_SUBDIR
    )
    framing_dir = (
        args.framing_dir.expanduser().resolve()
        if args.framing_dir
        else run_dir / DEFAULT_FRAMING_SUBDIR
    )
    gestalt_dir = (
        args.gestalt_dir.expanduser().resolve()
        if args.gestalt_dir
        else run_dir / DEFAULT_GESTALT_SUBDIR
    )
    mesh_arm_dir = (
        args.mesh_arm_dir.expanduser().resolve()
        if args.mesh_arm_dir
        else run_dir / DEFAULT_MESH_ARM_SUBDIR
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / DEFAULT_OUTPUT_SUBDIR
    )

    required = (run_dir, policy_dir, framing_dir, gestalt_dir, mesh_arm_dir)
    if not all(p.is_dir() for p in required):
        print(
            "Required input missing: "
            f"run={run_dir} policy={policy_dir} framing={framing_dir} "
            f"gestalt={gestalt_dir} mesh_arm={mesh_arm_dir}",
            file=sys.stderr,
        )
        return 2

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [
            p
            for p in paths
            if p.name.removesuffix(".perception_policy.json") in requested
        ]
    if not paths:
        print("No matching policy records.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for policy_path in paths:
        policy = _read_json(policy_path)
        key = str(
            policy.get("image_key")
            or policy_path.name.removesuffix(".perception_policy.json")
        )
        out_path = output_dir / f"{key}.selfie_evidence.json"

        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
            records.append(record)
            continue

        framing_path = framing_dir / f"{key}.framing.json"
        gestalt_path = gestalt_dir / f"{key}.gestalt.json"
        mesh_path = mesh_arm_dir / f"{key}.mesh_arm_occupancy.json"
        sources = (
            policy.get("sources")
            if isinstance(policy.get("sources"), dict)
            else {}
        )
        arrays_path = Path(str(sources.get("sam3d_arrays") or "")).expanduser()
        dwpose_path = Path(str(sources.get("dwpose") or "")).expanduser()

        if (
            not framing_path.is_file()
            or not gestalt_path.is_file()
            or not mesh_path.is_file()
            or not arrays_path.is_file()
            or not dwpose_path.is_file()
        ):
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "error": "missing_required_selfie_v06_input",
                "sources": {
                    "policy": str(policy_path),
                    "framing": str(framing_path),
                    "neutral_gestalt": str(gestalt_path),
                    "mesh_arm": str(mesh_path),
                    "sam3d_arrays": str(arrays_path),
                    "dwpose": str(dwpose_path),
                },
            }
        else:
            record = evaluate(
                policy,
                _read_json(framing_path),
                _read_json(gestalt_path),
                _read_json(mesh_path),
                _load_npz(arrays_path),
                _read_json(dwpose_path),
            )
            record["sources"] = {
                "policy": str(policy_path),
                "framing": str(framing_path),
                "neutral_gestalt": str(gestalt_path),
                "mesh_arm": str(mesh_path),
                "sam3d_arrays": str(arrays_path),
                "dwpose": str(dwpose_path),
            }

        _write_json(out_path, record)
        records.append(record)

        if record.get("status") == "ok":
            print(
                f"{key}: "
                f"mirror={record['evidence']['mirror_selfie_semantic']['grade']} "
                f"capture={record['capture_style'].get('subtype') or '-'} "
                f"=> {record['decision']['status']}"
            )
        else:
            print(f"{key}: error")

    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    decision_counts = Counter(
        str((r.get("decision") or {}).get("status") or "unknown")
        for r in records
        if r.get("status") == "ok"
    )
    subtype_counts = Counter(
        str((r.get("capture_style") or {}).get("subtype") or "none")
        for r in records
        if r.get("status") == "ok"
    )
    mirror_grade_counts = Counter(
        str(
            (((r.get("evidence") or {}).get("mirror_selfie_semantic") or {}).get("grade"))
            or "none"
        )
        for r in records
        if r.get("status") == "ok"
    )

    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "capture_subtype_counts": dict(sorted(subtype_counts.items())),
        "mirror_evidence_grade_counts": dict(sorted(mirror_grade_counts.items())),
        "publishable_selfie_count": sum(
            bool((r.get("decision") or {}).get("publishable_selfie"))
            for r in records
            if r.get("status") == "ok"
        ),
        "records": records,
    }
    _write_json(output_dir / "selfie_evidence.index.json", index)
    print(f"Index: {output_dir / 'selfie_evidence.index.json'}")
    return 1 if status_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
