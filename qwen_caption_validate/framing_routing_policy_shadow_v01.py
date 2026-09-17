from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "caption-perception-policy-framing-routing-shadow-0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.1"
DEFAULT_SHADOW_SUBDIR = Path("semantic-v3") / "framing-semantics-shadow-v0.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-perception-policy-framing-routing-shadow-v0.1"

MODE_RELEVANCE = {
    "framing_only": "negligible",
    "configuration": "low",
    "pose_allowed": "medium",
    "pose_guided": "high",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def apply_overlay(policy: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(policy)
    gate = shadow.get("pose_gate_shadow") if isinstance(shadow.get("pose_gate_shadow"), dict) else {}
    proposed = str(gate.get("proposed_mode") or "")
    if proposed not in MODE_RELEVANCE:
        raise ValueError(f"invalid proposed shadow mode: {proposed!r}")

    original_mode = str((policy.get("policy") or {}).get("mode") or "")
    original_relevance = policy.get("pose_relevance")
    original_broad = bool((policy.get("visibility") or {}).get("broad_pose_supported")) if isinstance(policy.get("visibility"), dict) else False
    proposed_broad = bool(gate.get("broad_pose_supported"))

    out["schema_version"] = SCHEMA_VERSION
    out.setdefault("policy", {})["mode"] = proposed
    out["pose_relevance"] = MODE_RELEVANCE[proposed]
    if isinstance(out.get("visibility"), dict):
        out["visibility"]["broad_pose_supported"] = proposed_broad

    out["routing_shadow"] = {
        "source": "framing_semantics_shadow_v02",
        "source_schema_version": shadow.get("schema_version"),
        "original": {
            "mode": original_mode,
            "pose_relevance": original_relevance,
            "broad_pose_supported": original_broad,
        },
        "proposed": {
            "mode": proposed,
            "pose_relevance": MODE_RELEVANCE[proposed],
            "broad_pose_supported": proposed_broad,
            "local_configuration_supported": bool(gate.get("local_configuration_supported")),
        },
        "local_configuration_gate": copy.deepcopy(gate.get("local_configuration_gate")),
        "note": (
            "Validation-only overlay. It changes route eligibility for downstream shadow acquisition "
            "without replacing production caption-perception-policy-v0.1."
        ),
    }

    reasons = list(policy.get("reasons") or [])
    reasons.append(
        "SHADOW ROUTING: broad-pose observability and local-configuration observability were evaluated independently."
    )
    if proposed == "configuration" and not proposed_broad:
        reasons.append(
            "SHADOW ROUTING: broad posture is withheld, but a directly visible local body relationship justifies the configuration-only observer."
        )
    out["reasons"] = reasons
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build validation-only perception-policy overlays from framing semantics shadow v0.2.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--shadow-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    shadow_dir = args.shadow_dir.expanduser().resolve() if args.shadow_dir else run_dir / DEFAULT_SHADOW_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    for label, path in (("run", run_dir), ("policy", policy_dir), ("shadow", shadow_dir)):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.only)
    policies = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        policies = [p for p in policies if p.name.removesuffix(".perception_policy.json") in requested]
    if not policies:
        print(f"No matching policies found in {policy_dir}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for policy_path in policies:
        key = policy_path.name.removesuffix(".perception_policy.json")
        shadow_path = shadow_dir / f"{key}.framing_shadow.json"
        if not shadow_path.is_file():
            missing.append(key)
            continue
        out_path = output_dir / f"{key}.perception_policy.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            try:
                record = apply_overlay(_read_json(policy_path), _read_json(shadow_path))
            except ValueError as exc:
                print(f"{key}: {exc}", file=sys.stderr)
                return 2
            _write_json(out_path, record)
        records.append(record)
        routing = record.get("routing_shadow") if isinstance(record.get("routing_shadow"), dict) else {}
        proposed = routing.get("proposed") if isinstance(routing.get("proposed"), dict) else {}
        original = routing.get("original") if isinstance(routing.get("original"), dict) else {}
        print(
            f"{key}: {original.get('mode')} -> {proposed.get('mode')} "
            f"broad={proposed.get('broad_pose_supported')} local={proposed.get('local_configuration_supported')}"
        )

    counts = Counter(str((r.get("policy") or {}).get("mode") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "shadow_dir": str(shadow_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "mode_counts": dict(sorted(counts.items())),
        "missing_shadow_records": missing,
        "records": [
            {
                "image_key": r.get("image_key"),
                "mode": (r.get("policy") or {}).get("mode"),
                "pose_relevance": r.get("pose_relevance"),
                "routing_shadow": r.get("routing_shadow"),
            }
            for r in records
        ],
    }
    _write_json(output_dir / "caption_perception_policy_framing_routing_shadow.index.json", index)
    print(f"Index: {output_dir / 'caption_perception_policy_framing_routing_shadow.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
