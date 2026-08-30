from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_07 as v07
from . import pose_semantics_08 as v08
from . import pose_semantics_10 as v10
from .pose_semantics_05 import _safe_float
from .runner import model_slug, resolve_model_id


_RESTORABLE_SUPPORT_CLASSES = {"supported_lean", "surface_support"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _gestalt_root(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("gestalt")
    return nested if isinstance(nested, dict) else payload


def _qualified_label(result: dict[str, Any]) -> str | None:
    posture = result.get("posture") or {}
    if posture.get("status") != "qualified":
        return None
    return str(posture.get("label") or "") or None


def _supporting_segment_for_gesture(semantic_v07: dict[str, Any], gesture: dict[str, Any]) -> tuple[Any, Any]:
    details = gesture.get("details") or {}
    kind = details.get("class")
    side = details.get("actor_side")
    if kind == "surface_support":
        return side, details.get("part")
    if kind == "supported_lean":
        graph = semantic_v07.get("support_graph") or {}
        chain = next(
            (
                item
                for item in graph.get("support_chains") or []
                if isinstance(item, dict) and item.get("side") == side
            ),
            None,
        )
        return side, (chain or {}).get("support_part")
    return side, None


def _restorable_support_gestures(semantic_v07: dict[str, Any], dwpose: dict[str, Any]) -> list[dict[str, Any]]:
    """Return top-down support primitives whose claimed supporting limb is observed.

    v0.10 intentionally required a duplicate non-gestalt contact/support edge. That
    prevented false support completion, but it also erased a useful image-conditioned
    semantic observation in tight crops. v0.11 uses DWPose only as an observation
    gate: the pose gestalt supplies the semantic support relation, while DWPose must
    independently show the relevant elbow/forearm/wrist chain. Reconstruction is not
    used to manufacture contact.
    """
    out: list[dict[str, Any]] = []
    for gesture in semantic_v07.get("gestures") or []:
        if not isinstance(gesture, dict) or not gesture.get("caption_preferred"):
            continue
        details = gesture.get("details") or {}
        if details.get("class") not in _RESTORABLE_SUPPORT_CLASSES:
            continue
        side, part = _supporting_segment_for_gesture(semantic_v07, gesture)
        if not v08._segment_visible(dwpose, side, part):
            continue
        restored = copy.deepcopy(gesture)
        restored.setdefault("support", []).append("v11_dwpose_supporting_segment_observation_gate")
        restored.setdefault("details", {})["semantic_authority"] = "image_conditioned_pose_gestalt_with_observed_supporting_segment"
        restored["caption_preferred"] = True
        out.append(restored)
    return out


def _merge_support_gestures(result: dict[str, Any], restored: list[dict[str, Any]]) -> None:
    if not restored:
        return
    gestures = [copy.deepcopy(item) for item in result.get("gestures") or [] if isinstance(item, dict)]
    restored_classes = {(item.get("details") or {}).get("class") for item in restored}
    restored_sides = {(item.get("details") or {}).get("actor_side") for item in restored}

    kept: list[dict[str, Any]] = []
    for item in gestures:
        details = item.get("details") or {}
        # A restored supported-lean/surface-support primitive is the semantic unit.
        # Do not also emit its lower-level head/hand or same-side surface ingredients.
        if details.get("class") in {"head_support", "surface_support", "supported_lean"}:
            if details.get("actor_side") in restored_sides:
                continue
        kept.append(item)

    merged = [*restored, *kept]
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in merged:
        label = str(item.get("label") or "")
        if label and label in seen:
            continue
        if label:
            seen.add(label)
        deduped.append(item)
    result["gestures"] = deduped
    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in deduped if item.get("caption_preferred")
    ]


def _restore_cropped_seated_semantics(
    result: dict[str, Any],
    semantic_v07: dict[str, Any],
    dwpose: dict[str, Any],
    gestalt_payload: dict[str, Any] | None,
) -> None:
    gestalt = _gestalt_root(gestalt_payload)
    probe_posture = str(gestalt.get("posture") or "").lower()
    probe_conf = _safe_float(gestalt.get("posture_confidence"), 0.0)
    probe_basis = str(gestalt.get("posture_basis") or "unknown").lower()
    current_label = _qualified_label(result)
    old_label = _qualified_label(semantic_v07)
    old_probe = semantic_v07.get("pose_gestalt_probe") or {}
    old_audit = semantic_v07.get("pose_gestalt_corroboration") or {}

    audit = {
        "eligible": False,
        "promoted": False,
        "probe_posture": probe_posture or None,
        "probe_confidence": round(probe_conf, 3),
        "probe_basis": probe_basis,
        "v10_qualified_posture": current_label,
        "v07_qualified_posture": old_label,
        "v07_corroboration_valid": bool(old_audit.get("valid")),
        "geometric_posture_conflict": bool(old_probe.get("contradicted_by_existing_geometric_posture")),
        "restored_support_gestures": [],
        "policy": (
            "A dedicated image-conditioned pose gestalt may establish a recognizable cropped seated/support configuration when confidence is strong, "
            "v0.7's non-circular support corroboration succeeded, no different v0.10 posture is qualified, and DWPose independently observes the supporting limb segment. "
            "DWPose is an observation gate, not contact authority."
        ),
    }

    eligible = (
        probe_posture in {"seated", "sitting"}
        and probe_conf >= 0.85
        and probe_basis in {"contextual", "mixed", "geometric"}
        and old_label == "seated"
        and bool(old_audit.get("valid"))
        and not bool(old_probe.get("contradicted_by_existing_geometric_posture"))
        and current_label in {None, "seated"}
    )
    audit["eligible"] = eligible
    if not eligible:
        result["top_down_cropped_seated_semantics_v11"] = audit
        return

    restored = _restorable_support_gestures(semantic_v07, dwpose)
    audit["restored_support_gestures"] = [item.get("label") for item in restored]

    # Posture recognition and support description are related but not identical.
    # The dedicated visual probe can establish seated as the human-level pose when
    # its earlier non-circular validation succeeded; caption-facing arm/table detail
    # is restored only when the relevant support segment is independently observed.
    if current_label is None:
        source_posture = copy.deepcopy(semantic_v07.get("posture") or {})
        source_posture.update(
            {
                "status": "qualified",
                "label": "seated",
                "primitive_id": "posture_seated_top_down_v11",
                "support_score": round(probe_conf, 3),
                "confidence_band": "strong",
                "caption_preferred": True,
                "authority": "v11_dedicated_image_pose_gestalt_plus_non_circular_support_validation",
                "limitations": list(source_posture.get("limitations") or [])
                + [
                    "lower body may be cropped; seated is a governed image-semantic fact rather than a claim reconstructed from hidden hips/legs"
                ],
            }
        )
        result["posture"] = source_posture
        result.setdefault("preferred_pose", {})["posture"] = "seated"
        result["posture_candidate"] = None
        audit["promoted"] = True

    _merge_support_gestures(result, restored)

    result["pose_gestalt_corroboration_v10"] = copy.deepcopy(result.get("pose_gestalt_corroboration") or {})
    result["pose_gestalt_corroboration"] = {
        "valid": True,
        "route": "v11_top_down_cropped_seated_semantics",
        "probe_posture": "seated",
        "probe_confidence": round(probe_conf, 3),
        "restored_support_gesture_count": len(restored),
        "restored_support_gestures": [item.get("label") for item in restored],
        "policy": audit["policy"],
    }
    probe = result.get("pose_gestalt_probe") or {}
    probe["caption_preferred"] = True
    probe["promotion_reason"] = "v11_top_down_cropped_seated_semantics"
    result["pose_gestalt_probe"] = probe
    result["top_down_cropped_seated_semantics_v11"] = audit


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = v10.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)
    semantic_v07 = v07.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)

    _restore_cropped_seated_semantics(result, semantic_v07, dwpose, gestalt_payload)

    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in result.get("gestures") or [] if item.get("caption_preferred")
    ]
    result["human_summary"] = base._human_summary(
        result.get("posture") or {},
        result.get("torso_orientation") or {},
        result.get("gestures") or [],
        result.get("head_and_gaze") or [],
        result.get("framing") or {},
    )
    result["schema_version"] = "pose-semantics-0.11"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "v0.10 reclining and false-surface-support hardening remains the baseline.",
        "For strongly recognized cropped seated poses, v0.11 re-admits the dedicated image-conditioned pose gestalt as semantic evidence when v0.7's non-circular corroboration succeeded and no conflicting v0.10 posture exists.",
        "Caption-facing forearm/arm-to-surface support is restored only when DWPose independently observes the claimed supporting segment; DWPose does not manufacture contact/support.",
        "This implements recognize-first/anatomize-second: bottom-up anatomy is used to veto or gate semantic precision, not to require a duplicate proof of every obvious human-level pose relation.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-11",
        description="Pose semantics v0.11: v0.10 safety plus governed recovery of cropped top-down seated/support semantics.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--fusion-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--gestalt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    model_id = resolve_model_id(args.model)
    slug = model_slug(model_id)
    fusion_dir = (args.fusion_dir or (run_dir / "fusion-v2.3.5" / slug)).expanduser().resolve()
    dwpose_dir = (args.dwpose_dir or (run_dir / "dwpose")).expanduser().resolve()
    analysis_dir = run_dir / slug
    gestalt_dir = (args.gestalt_dir or (run_dir / "pose-gestalt-v1" / slug)).expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.11" / slug)).expanduser().resolve()

    for path, label in (
        (fusion_dir, "Fusion"),
        (dwpose_dir, "DWPose"),
        (analysis_dir, "Analyze"),
        (gestalt_dir, "Pose gestalt"),
    ):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_paths = sorted(fusion_dir.glob("*.fused_v2_3.json"))
    if args.only:
        needles = tuple(args.only)
        fusion_paths = [path for path in fusion_paths if any(needle in path.name for needle in needles)]

    records: list[dict[str, Any]] = []
    for fusion_path in fusion_paths:
        key = fusion_path.name.removesuffix(".fused_v2_3.json")
        out_json = output_dir / f"{key}.pose_semantics.json"
        out_txt = output_dir / f"{key}.pose_semantics.txt"
        if out_json.exists() and out_txt.exists() and not args.overwrite:
            result = _read(out_json)
            records.append({"image_key": key, "status": "reused", "human_summary": result.get("human_summary")})
            continue

        dw_path = dwpose_dir / f"{key}.dwpose.json"
        analysis_path = analysis_dir / f"{key}.analysis.json"
        gestalt_path = gestalt_dir / f"{key}.pose_gestalt.json"
        if not dw_path.is_file() or not analysis_path.is_file() or not gestalt_path.is_file():
            records.append({"image_key": key, "status": "missing_source"})
            continue

        result = build_pose_semantics(
            _read(dw_path),
            _read(fusion_path),
            _read(analysis_path),
            _read(gestalt_path),
        )
        result.update(
            {
                "image_key": key,
                "source_paths": {
                    "fusion": str(fusion_path),
                    "dwpose": str(dw_path),
                    "analysis": str(analysis_path),
                    "pose_gestalt": str(gestalt_path),
                },
            }
        )
        _write(out_json, result)
        out_txt.write_text(str(result.get("human_summary") or "") + "\n", encoding="utf-8")
        v11_audit = result.get("top_down_cropped_seated_semantics_v11") or {}
        records.append(
            {
                "image_key": key,
                "status": "written",
                "posture": (result.get("preferred_pose") or {}).get("posture"),
                "gestures": (result.get("preferred_pose") or {}).get("gestures") or [],
                "v11_eligible": bool(v11_audit.get("eligible")),
                "v11_promoted": bool(v11_audit.get("promoted")),
                "restored_support_gestures": v11_audit.get("restored_support_gestures") or [],
                "human_summary": result.get("human_summary"),
            }
        )

    index = {
        "schema_version": "pose-semantics-0.11-run",
        "run_dir": str(run_dir),
        "analysis_model": model_id,
        "fusion_dir": str(fusion_dir),
        "dwpose_dir": str(dwpose_dir),
        "analysis_dir": str(analysis_dir),
        "gestalt_dir": str(gestalt_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
    }
    _write(output_dir / "pose_semantics.index.json", index)
    print(f"Pose semantics v0.11: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
