from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import camera_composition_shadow_v01 as camera_shadow

SCHEMA_VERSION = "selfie-evidence-shadow-0.2"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.2"
DEFAULT_FRAMING_SUBDIR = Path("semantic-v3") / "framing-semantics-v1.0"
DEFAULT_GESTALT_SUBDIR = Path("semantic-v3") / "routed-gestalt-v0.2"
DEFAULT_MESH_ARM_SUBDIR = Path("semantic-v3") / "mesh-arm-occupancy-shadow-v0.1"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "selfie-evidence-shadow-v0.2"

_SELFIE_RE = re.compile(
    r"\bselfie\b|\bselfie[- ]style\b|\bself[- ]portrait\b|\btaking (?:a )?selfie\b",
    re.I,
)

PRIMARY_FAMILIES = ("neutral_semantic", "camera_viewpoint", "foreground_arm")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def _neutral_selfie_semantic(gestalt_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(gestalt_record, dict):
        return {
            "grade": "none",
            "matched_text": [],
            "reason": "neutral_gestalt_record_missing",
        }

    acquisition = (
        gestalt_record.get("acquisition")
        if isinstance(gestalt_record.get("acquisition"), dict)
        else {}
    )
    candidates: list[str] = []
    for field in ("expression_action", "gestalt", "uncertainties"):
        candidates.extend(_flatten_strings(acquisition.get(field)))

    matched = [text for text in candidates if _SELFIE_RE.search(text)]
    if matched:
        return {
            "grade": "strong",
            "matched_text": matched,
            "reason": "selfie_language_appeared_naturally_in_neutral_gestalt_acquisition",
            "authority": "unprompted_qwen_gestalt_semantic",
        }
    return {
        "grade": "none",
        "matched_text": [],
        "reason": "neutral_gestalt_did_not_naturally_call_image_selfie",
        "authority": "unprompted_qwen_gestalt_semantic",
    }


def _camera_evidence(camera_viewpoint: dict[str, Any]) -> dict[str, Any]:
    classification = str(camera_viewpoint.get("classification") or "withheld")
    if classification == "elevated_downward":
        grade = "strong"
    elif classification == "downward_aimed":
        grade = "moderate"
    else:
        grade = "none"
    return {
        "grade": grade,
        "classification": classification,
        "vertical_vs_eye": camera_viewpoint.get("vertical_vs_eye"),
        "optical_axis_pitch_deg": camera_viewpoint.get("optical_axis_pitch_deg"),
        "composer_text": camera_viewpoint.get("composer_text") if grade != "none" else None,
        "reason": camera_viewpoint.get("reason"),
        "authority": "sam3d_subject_relative_camera_geometry_shadow",
    }


def _mesh_arm_evidence(mesh_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mesh_record, dict) or mesh_record.get("status") != "ok":
        return {
            "grade": "none",
            "selected_arm": None,
            "reason": "mesh_arm_occupancy_unavailable",
        }

    arms = mesh_record.get("arms") if isinstance(mesh_record.get("arms"), dict) else {}
    ranked = {"strong": 3, "moderate": 2, "weak": 1, "insufficient": 0}
    candidates: list[tuple[int, float, str, dict[str, Any]]] = []
    for side in ("left", "right"):
        arm = arms.get(side) if isinstance(arms.get(side), dict) else {}
        grade = str(arm.get("evidence_grade") or "insufficient")
        occupancy = arm.get("visible_mesh_area_fraction")
        occupancy = float(occupancy) if isinstance(occupancy, (int, float)) else 0.0
        candidates.append((ranked.get(grade, 0), occupancy, side, arm))

    candidates.sort(reverse=True)
    _, _, side, arm = candidates[0] if candidates else (0, 0.0, "", {})
    grade = str(arm.get("evidence_grade") or "insufficient")
    if grade not in {"strong", "moderate", "weak"}:
        grade = "none"

    return {
        "grade": grade,
        "selected_arm": side or None,
        "visible_mesh_area_fraction": arm.get("visible_mesh_area_fraction"),
        "occupancy_band": arm.get("occupancy_band"),
        "frame_region": arm.get("frame_region"),
        "composer_text": arm.get("composer_text_side_neutral"),
        "dwpose_observation_support": arm.get("dwpose_observation_support"),
        "reason": (
            "largest_supported_projected_arm_mesh_occupancy"
            if grade != "none"
            else "no_arm_cleared_mesh_plus_dwpose_evidence_threshold"
        ),
        "authority": "mesh_projection_plus_dwpose_observation_shadow",
    }


def _grade_rank(value: str) -> int:
    return {"none": 0, "weak": 1, "moderate": 2, "strong": 3}.get(str(value), 0)


def _decision(
    neutral: dict[str, Any],
    camera: dict[str, Any],
    arm: dict[str, Any],
    shoulder: dict[str, Any],
    portrait: dict[str, Any],
) -> dict[str, Any]:
    grades = {
        "neutral_semantic": str(neutral.get("grade") or "none"),
        "camera_viewpoint": str(camera.get("grade") or "none"),
        "foreground_arm": str(arm.get("grade") or "none"),
    }
    qualifying = [
        family for family, grade in grades.items()
        if _grade_rank(grade) >= _grade_rank("moderate")
    ]
    strong = [
        family for family, grade in grades.items()
        if grade == "strong"
    ]

    eligible = bool(portrait.get("eligible"))
    supported = bool(eligible and len(qualifying) >= 2 and len(strong) >= 1)

    if supported:
        status = "selfie_supported"
        reason = "at_least_two_independent_primary_cue_families_with_at_least_one_strong"
    elif len(strong) >= 1 or len(qualifying) >= 2:
        status = "candidate_not_publishable"
        reason = "interesting_selfie_evidence_but_not_enough_independent_support"
    else:
        status = "insufficient"
        reason = "not_enough_independent_selfie_evidence"

    promoted: list[str] = []
    if supported:
        if neutral.get("grade") == "strong":
            promoted.append("selfie-style capture")
        if camera.get("grade") in {"strong", "moderate"} and camera.get("composer_text"):
            promoted.append(str(camera["composer_text"]))
        if arm.get("grade") in {"strong", "moderate"} and arm.get("composer_text"):
            promoted.append(str(arm["composer_text"]))
        if (
            shoulder.get("publishable_candidate")
            and shoulder.get("composer_text")
            and (
                arm.get("grade") in {"strong", "moderate"}
                or neutral.get("grade") == "strong"
            )
        ):
            promoted.append(str(shoulder["composer_text"]))

    return {
        "status": status,
        "publishable_selfie": supported,
        "qualifying_primary_families": qualifying,
        "strong_primary_families": strong,
        "primary_family_grades": grades,
        "nearer_shoulder_is_supportive_not_primary": True,
        "promoted_fact_candidates": promoted,
        "reason": reason,
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

    neutral = _neutral_selfie_semantic(gestalt)
    camera = _camera_evidence(camera_viewpoint)
    arm = _mesh_arm_evidence(mesh_arm)
    decision = _decision(neutral, camera, arm, shoulder, portrait)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": policy.get("image_key"),
        "portrait_context": portrait,
        "evidence": {
            "neutral_semantic": neutral,
            "camera_viewpoint": camera,
            "foreground_arm": arm,
            "nearer_shoulder": {
                **shoulder,
                "role": "supportive_context_only",
            },
        },
        "decision": decision,
        "invariants": {
            "dedicated_selfie_prompt_is_not_used": True,
            "neutral_gestalt_selfie_language_is_high_precision_but_not_required": True,
            "selfie_requires_at_least_two_independent_primary_cue_families": True,
            "at_least_one_primary_cue_must_be_strong": True,
            "nearer_shoulder_never_counts_as_primary_selfie_evidence": True,
            "mesh_is_reconstruction_and_requires_dwpose_observation_support": True,
            "foreground_arm_caption_text_is_frame_relative_and_side_neutral": True,
            "selfie_label_does_not_imply_arm_holds_camera": True,
            "misses_are_preferred_to_single_cue_false_positives": True,
        },
    }


def _load_npz(path: Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grade independent selfie evidence from neutral gestalt, camera geometry and mesh arm occupancy."
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
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    framing_dir = args.framing_dir.expanduser().resolve() if args.framing_dir else run_dir / DEFAULT_FRAMING_SUBDIR
    gestalt_dir = args.gestalt_dir.expanduser().resolve() if args.gestalt_dir else run_dir / DEFAULT_GESTALT_SUBDIR
    mesh_arm_dir = args.mesh_arm_dir.expanduser().resolve() if args.mesh_arm_dir else run_dir / DEFAULT_MESH_ARM_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

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
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in requested]
    if not paths:
        print("No matching policy records.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for policy_path in paths:
        policy = _read_json(policy_path)
        key = str(policy.get("image_key") or policy_path.name.removesuffix(".perception_policy.json"))
        out_path = output_dir / f"{key}.selfie_evidence.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
            records.append(record)
            continue

        framing_path = framing_dir / f"{key}.framing.json"
        gestalt_path = gestalt_dir / f"{key}.gestalt.json"
        mesh_path = mesh_arm_dir / f"{key}.mesh_arm_occupancy.json"
        sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
        arrays_path = Path(str(sources.get("sam3d_arrays") or "")).expanduser()
        dwpose_path = Path(str(sources.get("dwpose") or "")).expanduser()

        if not framing_path.is_file() or not mesh_path.is_file() or not arrays_path.is_file() or not dwpose_path.is_file():
            record = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "image_key": key,
                "error": "missing_required_selfie_evidence_input",
            }
        else:
            record = evaluate(
                policy,
                _read_json(framing_path),
                _read_json(gestalt_path) if gestalt_path.is_file() else None,
                _read_json(mesh_path),
                _load_npz(arrays_path),
                _read_json(dwpose_path),
            )
            record["sources"] = {
                "policy": str(policy_path),
                "framing": str(framing_path),
                "neutral_gestalt": str(gestalt_path) if gestalt_path.is_file() else None,
                "mesh_arm": str(mesh_path),
                "sam3d_arrays": str(arrays_path),
                "dwpose": str(dwpose_path),
            }

        _write_json(out_path, record)
        records.append(record)
        if record.get("status") == "ok":
            ev = record["evidence"]
            dec = record["decision"]
            print(
                f"{key}: "
                f"neutral={ev['neutral_semantic']['grade']} "
                f"camera={ev['camera_viewpoint']['grade']} "
                f"arm={ev['foreground_arm']['grade']} "
                f"shoulder={ev['nearer_shoulder'].get('anatomical_side_nearer') or '-'} "
                f"=> {dec['status']}"
            )
        else:
            print(f"{key}: error")

    status_counts = Counter(str(r.get("status") or "unknown") for r in records)
    decision_counts = Counter(
        str((r.get("decision") or {}).get("status") or "unknown")
        for r in records
        if r.get("status") == "ok"
    )
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
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
