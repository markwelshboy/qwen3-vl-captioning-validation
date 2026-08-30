from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_11 as v11
from .runner import model_slug, resolve_model_id


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _restored_by_v11(item: dict[str, Any]) -> bool:
    details = item.get("details") or {}
    return details.get("semantic_authority") == "image_conditioned_pose_gestalt_with_observed_supporting_segment"


def _target_neutral_supported_lean(item: dict[str, Any]) -> dict[str, Any]:
    """Preserve the visible head/hand/forearm configuration without inventing a surface.

    v0.11 proved only that the claimed supporting limb segment was observed. That is
    sufficient to retain the recognizable head-on-hand + forearm configuration, but
    it is not sufficient to assert that the forearm is resting on a table, desk,
    counter, or other external support target.
    """
    details = item.get("details") or {}
    side = details.get("actor_side")
    side_text = f"{side} " if side in {"left", "right"} else ""
    label = f"chin resting on the {side_text}hand, with the {side_text}forearm held beneath it"
    support = []
    for value in item.get("support") or []:
        text = str(value)
        if "head/chin is supported by" in text or "v11_dwpose_supporting_segment_observation_gate" in text:
            support.append(value)
    return {
        "id": f"gesture_head_hand_forearm_configuration_{side or 'unknown'}_v12",
        "label": label,
        "support_score": item.get("support_score"),
        "confidence_band": item.get("confidence_band"),
        "caption_preferred": True,
        "support": support,
        "limitations": [
            "external support target withheld: observing the forearm does not verify a table/desk/surface contact target"
        ],
        "subsumes": [
            "head-on-hand contact",
            "visible supporting forearm configuration",
            "unverified external support-target prose",
        ],
        "details": {
            "class": "head_support_with_forearm_configuration",
            "actor_side": side,
            "surface_target": None,
            "surface_target_status": "withheld_unverified",
            "semantic_authority": "governed_head_hand_relation_plus_dwpose_forearm_observation",
        },
    }


def _apply_support_target_firewall(result: dict[str, Any]) -> None:
    """Separate limb observation from support-target verification.

    A top-down image model may correctly recognize that a forearm/hand is important
    while hallucinating what it rests on (for example, mistaking the subject's lap or
    clothing for a table). DWPose can verify that the limb exists and its side, but it
    cannot verify the semantic identity/ownership of the contact target. Therefore:

    * v11-restored generic surface_support is withheld entirely unless a future
      target-verification stage explicitly re-admits it.
    * v11-restored supported_lean is downgraded to the target-neutral visible
      head/hand/forearm configuration. This preserves useful pose structure without
      claiming a table/desk/surface outside the evidence.
    """
    gestures = [copy.deepcopy(item) for item in result.get("gestures") or [] if isinstance(item, dict)]
    rewritten: list[dict[str, Any]] = []
    withheld: list[str] = []
    downgraded: list[dict[str, Any]] = []

    for item in gestures:
        if not _restored_by_v11(item):
            rewritten.append(item)
            continue

        details = item.get("details") or {}
        kind = details.get("class")
        if kind == "surface_support":
            item["caption_preferred"] = False
            item.setdefault("limitations", []).append(
                "v12 support-target firewall: observed limb does not verify an external support target"
            )
            withheld.append(str(item.get("label") or ""))
            rewritten.append(item)
            continue

        if kind == "supported_lean":
            neutral = _target_neutral_supported_lean(item)
            downgraded.append(
                {
                    "from": item.get("label"),
                    "to": neutral.get("label"),
                    "reason": "external support target not independently verified",
                }
            )
            rewritten.append(neutral)
            continue

        rewritten.append(item)

    # Dedupe by label while preferring caption-facing entries.
    by_label: dict[str, dict[str, Any]] = {}
    for item in rewritten:
        label = str(item.get("label") or "")
        previous = by_label.get(label)
        if previous is None or (item.get("caption_preferred") and not previous.get("caption_preferred")):
            by_label[label] = item
    result["gestures"] = list(by_label.values())
    result.setdefault("preferred_pose", {})["gestures"] = [
        item.get("label") for item in result["gestures"] if item.get("caption_preferred")
    ]
    result["support_target_firewall_v12"] = {
        "withheld_surface_support": withheld,
        "downgraded_supported_lean": downgraded,
        "policy": (
            "DWPose/SAM3D-derived limb observation or laterality may corroborate the actor limb, but neither proves that the contact target is an external table/desk/surface. "
            "External support-target semantics require their own governed verification."
        ),
    }


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = v11.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)
    _apply_support_target_firewall(result)

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
    result["schema_version"] = "pose-semantics-0.12"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "v0.12 separates ACTOR-LIMB verification from SUPPORT-TARGET verification.",
        "A DWPose-observed forearm can corroborate that the forearm exists and its governed laterality; it cannot prove the forearm is resting on an external table/desk/surface.",
        "Top-down surface-support-only gestures restored by v0.11 are withheld pending target verification.",
        "A supported head-on-hand configuration may retain target-neutral head/hand/forearm structure while the external support target is withheld.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-12",
        description="Pose semantics v0.12: v0.11 cropped-pose recovery plus a separate support-target firewall.",
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
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.12" / slug)).expanduser().resolve()

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
        records.append(
            {
                "image_key": key,
                "status": "written",
                "posture": (result.get("preferred_pose") or {}).get("posture"),
                "gestures": (result.get("preferred_pose") or {}).get("gestures") or [],
                "support_target_firewall_v12": result.get("support_target_firewall_v12") or {},
                "human_summary": result.get("human_summary"),
            }
        )

    index = {
        "schema_version": "pose-semantics-0.12-run",
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
    print(f"Pose semantics v0.12: {output_dir}")
    for record in records:
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
