from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from . import extract_v3
from .semantic_v3_pose_adapter import adapt_pose_v016


FUSION_VERSION = "semantic-v3-kiss-fusion-0.1"
RUN_VERSION = "semantic-v3-kiss-fusion-0.1-run"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def fuse_kiss(
    *,
    image_key: str,
    packet_artifact: dict[str, Any],
    pose_record: dict[str, Any],
) -> dict[str, Any]:
    """Fuse one sparse semantic packet with frozen governed Pose v0.16.

    KISS v0.1 intentionally does very little:
    - semantic packet text is not interrogated or expanded;
    - governed public Pose is the only structured posture authority;
    - withheld reconstruction remains provenance only;
    - only positively asserted governed physical relations enter canonical truth.
    """

    packet = _dict(packet_artifact.get("packet"))
    pose = adapt_pose_v016(pose_record)
    public = _dict(pose.get("public"))
    reconstruction = _dict(pose.get("reconstruction"))
    modifiers = _dict(pose.get("modifiers"))

    public_value = str(public.get("value") or "unknown")
    authority_state = str(public.get("authority_state") or "unknown")
    if authority_state == "asserted" and public_value != "unknown":
        posture = {
            "value": public_value,
            "status": "asserted",
            "authority": "pose_v0.16_public",
            "pose_joint_authority": public.get("crop_support"),
            "withheld_reason": None,
        }
    elif authority_state == "withheld":
        posture = {
            "value": "unknown",
            "status": "withheld",
            "authority": "pose_v0.16_public",
            "pose_joint_authority": public.get("crop_support"),
            "withheld_reason": public.get("withheld_reason"),
        }
    else:
        posture = {
            "value": "unknown",
            "status": "unknown",
            "authority": "pose_v0.16_public",
            "pose_joint_authority": public.get("crop_support"),
            "withheld_reason": public.get("withheld_reason"),
        }

    modifier_family = str(modifiers.get("pose_family") or "unknown")
    if posture["status"] == "asserted" and modifier_family == posture["value"]:
        canonical_modifiers: dict[str, Any] | None = {
            "pose_family": modifier_family,
            "lean_severity": modifiers.get("lean_severity"),
            "lean_direction": modifiers.get("lean_direction"),
            "shoulder_line_tilt_severity": modifiers.get("shoulder_line_tilt_severity"),
            "suggested_modifiers": deepcopy(modifiers.get("suggested_modifiers") or []),
            "suggested_compound_pose_modifier": modifiers.get("suggested_compound_pose_modifier"),
            "authority": "pose_v0.16_public_family",
        }
    else:
        canonical_modifiers = None

    positive_relations: dict[str, Any] = {}
    for name, relation_value in _dict(pose.get("relations")).items():
        relation = _dict(relation_value)
        if relation.get("value") is True:
            positive_relations[name] = {
                "value": True,
                "side": relation.get("side"),
                "support_class": relation.get("support_class"),
                "crop_support": relation.get("crop_support"),
                "authority": "pose_v0.16_governed_relation",
            }

    canonical = {
        "semantic": {
            "subject": deepcopy(_list(packet.get("subject"))),
            "body": deepcopy(_list(packet.get("body"))),
            "interactions": deepcopy(_list(packet.get("interactions"))),
            "scene": deepcopy(_list(packet.get("scene"))),
            "composition": deepcopy(_list(packet.get("composition"))),
            "uncertainties": deepcopy(_list(packet.get("uncertainties"))),
        },
        "posture": posture,
        "pose_modifiers": canonical_modifiers,
        "physical_relations": positive_relations,
    }

    return {
        "schema_version": FUSION_VERSION,
        "image_key": image_key,
        "canonical": canonical,
        "policy": {
            "architecture": "one image VLM semantic packet + governed Pose v0.16 + deterministic Fusion",
            "semantic_packet": (
                "Sparse image-conditioned observations are preserved without confidence, evidence, "
                "counterevidence, or clarification passes."
            ),
            "pose_authority": (
                "Governed public Pose is the only structured posture authority. Reconstruction is provenance only."
            ),
            "physical_relations": (
                "Only positively asserted governed Pose relations enter canonical truth in KISS v0.1; "
                "negative/unknown relations remain provenance."
            ),
            "semantic_recovery": (
                "KISS v0.1 does not perform a second LLM recovery pass and does not promote a withheld reconstruction."
            ),
        },
        "provenance": {
            "packet_schema_version": packet_artifact.get("packet_schema_version"),
            "packet_artifact_schema_version": packet_artifact.get("schema_version"),
            "pose_adapter": pose,
            "pose_reconstruction_provenance": {
                "best_candidate": reconstruction.get("best_candidate"),
                "score": reconstruction.get("score"),
                "winner_margin": reconstruction.get("winner_margin"),
                "support_class": reconstruction.get("support_class"),
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="semantic-v3-kiss-fusion",
        description="Deterministically fuse sparse KISS semantic packets with governed Pose v0.16.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    packet_dir = args.packet_dir.expanduser().resolve()
    pose_dir = args.pose_dir.expanduser().resolve()
    output_dir = (
        args.output_dir or (run_dir / "semantic-v3" / "kiss-fusion-v0.1")
    ).expanduser().resolve()

    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    if not packet_dir.is_dir():
        print(f"Packet directory not found: {packet_dir}", file=sys.stderr)
        return 2
    if not pose_dir.is_dir():
        print(f"Pose directory not found: {pose_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only)
    packet_paths = sorted(packet_dir.glob("*.semantic_packet.json"))
    if only:
        packet_paths = [
            path for path in packet_paths
            if path.name.removesuffix(".semantic_packet.json") in only
        ]
    if not packet_paths:
        print("No matching KISS semantic packets found.", file=sys.stderr)
        return 2

    print("Semantic V3 KISS Fusion: sparse packet + governed Pose v0.16 -> canonical truth")
    print("Model/GPU load: NONE")
    print("Policy: no Analyze/Gestalt calls; no semantic self-voting; Pose reconstruction is provenance only")

    records: list[dict[str, Any]] = []
    for packet_path in packet_paths:
        packet_artifact = _read_json(packet_path)
        image_key = str(packet_artifact.get("image_key") or packet_path.name.removesuffix(".semantic_packet.json"))
        if not packet_artifact.get("schema_valid") or not isinstance(packet_artifact.get("packet"), dict):
            print(f"ERROR: invalid packet artifact for {image_key}: {packet_path}", file=sys.stderr)
            return 1

        pose_path = pose_dir / f"{image_key}.sam3d_relational_pose.json"
        if not pose_path.is_file():
            print(f"ERROR: governed Pose v0.16 record missing for {image_key}: {pose_path}", file=sys.stderr)
            return 1

        out_path = output_dir / f"{image_key}.kiss_fusion.json"
        if out_path.exists() and not args.overwrite:
            fused = _read_json(out_path)
            reused = True
        else:
            fused = fuse_kiss(
                image_key=image_key,
                packet_artifact=packet_artifact,
                pose_record=_read_json(pose_path),
            )
            extract_v3._write_json(out_path, fused)
            reused = False

        posture = _dict(_dict(fused.get("canonical")).get("posture"))
        records.append({
            "image_key": image_key,
            "output": str(out_path),
            "reused": reused,
            "posture": posture.get("value"),
            "posture_status": posture.get("status"),
            "positive_physical_relations": sorted(
                _dict(_dict(fused.get("canonical")).get("physical_relations")).keys()
            ),
        })
        print(
            f"{image_key}: posture={posture.get('value')} "
            f"status={posture.get('status')} "
            f"positive_relations={len(records[-1]['positive_physical_relations'])}"
        )

    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "packet_dir": str(packet_dir),
        "pose_dir": str(pose_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
    }
    extract_v3._write_json(output_dir / "kiss_fusion.index.json", index)
    print(f"KISS Fusion: {output_dir}")
    print(f"Records: {len(records)}; GPU/model load: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
