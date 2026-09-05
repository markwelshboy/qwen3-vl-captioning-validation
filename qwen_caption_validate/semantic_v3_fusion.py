from __future__ import annotations

import argparse
import fnmatch
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .semantic_v3_pose_adapter import adapt_pose_v016


FUSION_VERSION = "semantic-fusion-3.0"
RUN_VERSION = "semantic-fusion-3.0-run"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _matches(key: str, only: list[str]) -> bool:
    if not only:
        return True
    return any(fnmatch.fnmatch(key, pattern) or pattern == key or pattern in key for pattern in only)


def _posture_from_sources(analyze: dict[str, Any], pose: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conflicts: list[dict[str, Any]] = []
    a = _dict(analyze.get("posture"))
    p_public = _dict(pose.get("public"))
    p_recon = _dict(pose.get("reconstruction"))
    recovery = _dict(pose.get("semantic_recovery"))

    av = str(a.get("value") or "unknown")
    a_conf = float(a.get("confidence") or 0.0)
    a_assessment = str(a.get("assessment") or "unknown")
    pv = str(p_public.get("value") or "unknown")
    rv = str(p_recon.get("best_candidate") or "unknown")

    if pv != "unknown" and p_public.get("authority_state") == "asserted":
        result = {
            "value": pv,
            "status": "asserted",
            "confidence": p_public.get("selected_path_authority") or p_public.get("crop_support") or 0.0,
            "authority": "pose_v0.16_public",
            "semantic_value": av,
            "semantic_confidence": a_conf,
            "semantic_assessment": a_assessment,
            "pose_reconstruction": rv,
            "rationale": ["Pose v0.16 published this broad posture through its assertion-authority policy."],
        }
        if av != "unknown" and av != pv:
            conflicts.append({
                "type": "posture_disagreement",
                "semantic": {"value": av, "confidence": a_conf, "assessment": a_assessment},
                "pose_public": pv,
                "resolution": "pose_public",
            })
        return result, conflicts

    # Public geometry withheld. Recovery may use independent semantic interpretation,
    # but Pose reconstruction is provenance only and never becomes a vote.
    recovery_allowed = recovery.get("needed") is True
    semantic_strong = av != "unknown" and a_assessment == "supported" and a_conf >= 0.80
    if recovery_allowed and semantic_strong:
        result = {
            "value": av,
            "status": "recovered",
            "confidence": a_conf,
            "authority": "semantic_recovery_via_analyze",
            "semantic_value": av,
            "semantic_confidence": a_conf,
            "semantic_assessment": a_assessment,
            "pose_reconstruction": rv,
            "rationale": [
                "Pose v0.16 withheld public posture and explicitly requested semantic recovery.",
                "Analyze independently supplied a supported high-confidence broad posture.",
            ],
        }
        if rv != "unknown" and rv != av:
            conflicts.append({
                "type": "withheld_reconstruction_disagrees_with_semantic_recovery",
                "semantic": av,
                "pose_reconstruction": rv,
                "resolution": "semantic_recovery",
                "note": "reconstruction is not an independent vote because public Pose authority was withheld",
            })
        return result, conflicts

    rationale = ["Pose v0.16 did not publish a broad posture."]
    if recovery_allowed:
        rationale.append("Semantic recovery was requested, but Analyze did not provide a supported >=0.80 posture claim.")
    else:
        rationale.append("No semantic-recovery path was authorized by Pose v0.16.")
    return {
        "value": "unknown",
        "status": "withheld" if p_public.get("authority_state") == "withheld" else "unknown",
        "confidence": 0.0,
        "authority": "withheld",
        "semantic_value": av,
        "semantic_confidence": a_conf,
        "semantic_assessment": a_assessment,
        "pose_reconstruction": rv,
        "rationale": rationale,
    }, conflicts


def _hand_head_topology_conflict(pose: dict[str, Any]) -> dict[str, Any] | None:
    relation = _dict(_dict(pose.get("relations")).get("head_supported_by_hand"))
    if relation.get("value") is not False:
        return None
    reason = str(relation.get("rejection_reason") or "")
    proximal = _dict(relation.get("v14_proximal_chain_guard"))
    if proximal.get("geometry_match") is False or "proximal" in reason or "wrist_palm" in reason or "finger_proximity" in reason:
        return relation
    return None


def _is_hand_head_semantic_interaction(item: dict[str, Any]) -> bool:
    actor = str(item.get("actor_part") or "").lower()
    if not any(token in actor for token in ("hand", "finger", "fist")):
        return False
    target = " ".join(
        str(item.get(k) or "").lower()
        for k in ("target_ref", "target_text", "interpretation")
    )
    return any(token in target for token in ("target_subject", "head", "face", "chin", "neck"))


def _govern_interactions_and_ownership(analyze: dict[str, Any], pose: dict[str, Any]) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    interactions = deepcopy(_list(analyze.get("interactions")))
    ownership = deepcopy(_list(analyze.get("ownership_assessments")))
    conflicts: list[dict[str, Any]] = []
    topology = _hand_head_topology_conflict(pose)
    downgraded_parts: set[str] = set()

    if topology is not None:
        for index, item in enumerate(interactions):
            if not isinstance(item, dict) or not _is_hand_head_semantic_interaction(item):
                continue
            if item.get("actor_ownership") != "target":
                continue
            original = deepcopy(item)
            item["actor_ownership"] = "unknown"
            item["confidence"] = min(float(item.get("confidence") or 0.0), 0.60)
            item["limitations"] = list(item.get("limitations") or []) + [
                "Fusion withheld target ownership for this hand/head relation because governed Pose rejects the proximal palm/wrist support chain."
            ]
            part = str(item.get("actor_part") or "")
            if part:
                downgraded_parts.add(part)
            conflicts.append({
                "type": "semantic_hand_head_chain_vs_pose_topology",
                "interaction_index": index,
                "semantic_original": original,
                "pose_relation": deepcopy(topology),
                "resolution": "withhold_actor_ownership_for_relation",
            })

    if downgraded_parts:
        for index, item in enumerate(ownership):
            if not isinstance(item, dict):
                continue
            part = str(item.get("part") or "")
            if part not in downgraded_parts or item.get("ownership") != "target":
                continue
            original = deepcopy(item)
            item["ownership"] = "unknown"
            item["confidence"] = min(float(item.get("confidence") or 0.0), 0.50)
            item["limitations"] = list(item.get("limitations") or []) + [
                "Fusion withheld whole-part target ownership because the related governed Pose topology does not support the claimed proximal chain."
            ]
            conflicts.append({
                "type": "semantic_whole_part_ownership_withheld",
                "ownership_index": index,
                "semantic_original": original,
                "resolution": "unknown",
            })

    return interactions, ownership, conflicts


def fuse_semantic_v3(
    *,
    image_key: str,
    extract_wrapper: dict[str, Any],
    analyze_artifact: dict[str, Any],
    gestalt_artifact: dict[str, Any],
    pose_record: dict[str, Any],
) -> dict[str, Any]:
    analyze = _dict(analyze_artifact.get("analyze"))
    gestalt = _dict(gestalt_artifact.get("gestalt"))
    pose = adapt_pose_v016(pose_record)

    posture, posture_conflicts = _posture_from_sources(analyze, pose)
    interactions, ownership, physical_conflicts = _govern_interactions_and_ownership(analyze, pose)

    head_support = _dict(_dict(pose.get("relations")).get("head_supported_by_hand"))
    if head_support:
        head_support_value = head_support.get("value")
        head_support_canonical = {
            "value": head_support_value,
            "authority": "pose_v0.16_relation_topology",
            "reason": head_support.get("rejection_reason"),
            "support_class": head_support.get("support_class"),
        }
    else:
        head_support_canonical = {"value": None, "authority": "unavailable", "reason": None, "support_class": None}

    canonical = {
        "posture": posture,
        "actions": deepcopy(_list(analyze.get("actions"))),
        "interactions": interactions,
        "ownership_assessments": ownership,
        "support_context": deepcopy(_list(analyze.get("support_context"))),
        "composition": {
            "framing": deepcopy(gestalt.get("framing")),
            "environment": deepcopy(gestalt.get("environment")),
            "background_regions": deepcopy(_list(gestalt.get("background_regions"))),
            "foreground_relations": deepcopy(_list(gestalt.get("foreground_relations"))),
        },
        "physical_relations": {
            "head_supported_by_hand": head_support_canonical,
        },
        "pose_modifiers": deepcopy(pose.get("modifiers")),
    }

    return {
        "schema_version": FUSION_VERSION,
        "image_key": image_key,
        "canonical": canonical,
        "conflicts": posture_conflicts + physical_conflicts,
        "pose_adapter": pose,
        "semantic_provenance": {
            "extract_schema_version": _dict(extract_wrapper.get("extract")).get("schema_version"),
            "extract_source_sha256": analyze_artifact.get("source_extract_sha256"),
            "analyze_schema_version": analyze.get("schema_version"),
            "gestalt_schema_version": gestalt.get("schema_version"),
        },
        "policy": {
            "semantic_evidence_family": "Extract + Analyze + Gestalt are one semantic evidence family and are never counted as independent votes.",
            "pose_evidence_family": "Pose v0.16 public assertions and governed relations are the physical/geometry authority surface.",
            "reconstruction": "Pose reconstruction remains provenance when public assertion authority is withheld.",
            "semantic_recovery": "A withheld public posture may be recovered only from supported Analyze semantics when Pose v0.16 explicitly requests semantic recovery.",
            "missing_evidence": "missing evidence is not negative evidence",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="qwen-semantic-v3-fusion", description="Fuse Extract, Analyze V3, Gestalt V0.3 and governed Pose v0.16 without loading a model.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--analyze-dir", type=Path, required=True)
    parser.add_argument("--gestalt-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    extract_dir = args.extract_dir.expanduser().resolve()
    analyze_dir = args.analyze_dir.expanduser().resolve()
    gestalt_dir = args.gestalt_dir.expanduser().resolve()
    pose_dir = args.pose_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "semantic-v3" / "fusion-v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for analyze_path in sorted(analyze_dir.glob("*.analyze_v3.json")):
        key = analyze_path.name.removesuffix(".analyze_v3.json")
        if not _matches(key, args.only):
            continue
        out_path = output_dir / f"{key}.fusion_v3.json"
        if out_path.exists() and not args.overwrite:
            rows.append({"image_key": key, "status": "reused", "output": str(out_path)})
            continue

        extract_path = extract_dir / f"{key}.extract.json"
        gestalt_path = gestalt_dir / f"{key}.gestalt_v3.json"
        pose_path = pose_dir / f"{key}.sam3d_relational_pose.json"
        missing = [str(path) for path in (extract_path, gestalt_path, pose_path) if not path.exists()]
        if missing:
            rows.append({"image_key": key, "status": "missing_inputs", "missing": missing})
            continue

        extract_wrapper = _read_json(extract_path)
        analyze_artifact = _read_json(analyze_path)
        gestalt_artifact = _read_json(gestalt_path)
        pose_record = _read_json(pose_path)
        if analyze_artifact.get("schema_valid") is False or gestalt_artifact.get("schema_valid") is False:
            rows.append({"image_key": key, "status": "invalid_semantic_input"})
            continue

        fused = fuse_semantic_v3(
            image_key=key,
            extract_wrapper=extract_wrapper,
            analyze_artifact=analyze_artifact,
            gestalt_artifact=gestalt_artifact,
            pose_record=pose_record,
        )
        payload = {
            "schema_version": FUSION_VERSION,
            "image_key": key,
            "source_paths": {
                "extract": str(extract_path),
                "analyze": str(analyze_path),
                "gestalt": str(gestalt_path),
                "pose": str(pose_path),
            },
            "fusion": fused,
        }
        _write_json(out_path, payload)
        posture = _dict(_dict(fused.get("canonical")).get("posture"))
        rows.append({
            "image_key": key,
            "status": "written",
            "posture": posture.get("value"),
            "posture_status": posture.get("status"),
            "conflict_count": len(_list(fused.get("conflicts"))),
            "output": str(out_path),
        })
        print(f"{key}: posture={posture.get('value')} status={posture.get('status')} conflicts={len(_list(fused.get('conflicts')))}")

    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "extract_dir": str(extract_dir),
        "analyze_dir": str(analyze_dir),
        "gestalt_dir": str(gestalt_dir),
        "pose_dir": str(pose_dir),
        "output_dir": str(output_dir),
        "GPU/model_load": "none",
        "records": rows,
    }
    _write_json(output_dir / "fusion_v3.index.json", index)
    print(f"Semantic V3 Fusion: {output_dir}")
    print(f"Records: {len(rows)}; GPU/model load: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
