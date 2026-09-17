from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

from .caption_perception_policy import _dwpose_points

SCHEMA_VERSION = "local-configuration-semantics-shadow-0.1"
DEFAULT_FRAGMENT_SUBDIR = Path("semantic-v3") / "fragment-probe-framing-routing-shadow-v0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-framing-routing-shadow-v0.1"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "local-configuration-semantics-shadow-v0.1"

ANATOMICAL_LATERALITY_RE = re.compile(
    r"\b(left|right)\s+(hand|arm|forearm|wrist|elbow|shoulder|hip|knee|leg|ankle|foot)\b",
    re.I,
)
HEAD_SUPPORT_RE = re.compile(
    r"(?:\b(?:fist|hand)\b.{0,30}\b(?:under|beneath|support(?:ing|s)?)\b.{0,30}\b(?:chin|head)\b"
    r"|\b(?:chin|head)\b.{0,30}\b(?:rest(?:ing|s)?|support(?:ed|ing)?)\b.{0,30}\b(?:fist|hand)\b"
    r"|\b(?:hand|fist)\s+support(?:ing|s)?\s+(?:the\s+)?(?:chin|head)\b)",
    re.I,
)
# A generic "forearm held ..." phrase is not evidence that the forearm supports
# the head/chin arrangement.  Require an actual support/beneath/under semantic.
FOREARM_SUPPORT_RE = re.compile(
    r"\bforearm\b.{0,40}\b(?:beneath|under|support(?:ing|s|ed)?)\b",
    re.I,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _neutralize_laterality(text: str) -> tuple[str, list[str]]:
    leaks: list[str] = []

    def repl(match: re.Match[str]) -> str:
        leaks.append(match.group(0))
        return match.group(2)

    neutral = ANATOMICAL_LATERALITY_RE.sub(repl, text)
    return " ".join(neutral.split()).strip(" ,;"), leaks


def _relationships(fragment: dict[str, Any]) -> list[str]:
    extraction = fragment.get("extraction") if isinstance(fragment.get("extraction"), dict) else {}
    raw = extraction.get("body_relationships") if isinstance(extraction.get("body_relationships"), list) else []
    out: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            text = _clean(item.get("text"))
        else:
            text = _clean(item)
        if text:
            out.append(text)
    return out


def _head_support_semantics(neutralized: list[str]) -> dict[str, Any]:
    support_sources = [text for text in neutralized if HEAD_SUPPORT_RE.search(text)]
    forearm_sources = [text for text in neutralized if FOREARM_SUPPORT_RE.search(text)]
    if not support_sources:
        return {
            "status": "not_present",
            "source_relations": [],
            "forearm_source_relations": [],
            "canonical_side_neutral": None,
            "uses_fist_semantics": False,
            "forearm_support_explicit": False,
        }

    uses_fist = any(re.search(r"\bfist\b", text, re.I) for text in support_sources)
    forearm_explicit = bool(forearm_sources) or any(
        re.search(r"\bforearm\b.{0,40}\b(?:beneath|under|support(?:ing|s|ed)?)\b", text, re.I)
        for text in support_sources
    )
    hand_noun = "fist" if uses_fist else "hand"
    canonical = f"chin resting on a {hand_noun}"
    if forearm_explicit:
        canonical += ", with the forearm beneath/supporting the pose"

    return {
        "status": "candidate",
        "source_relations": support_sources,
        "forearm_source_relations": forearm_sources,
        "canonical_side_neutral": canonical,
        "uses_fist_semantics": uses_fist,
        "forearm_support_explicit": forearm_explicit,
    }


def _load_points(policy: dict[str, Any]) -> tuple[dict[str, tuple[float, float] | None] | None, str | None]:
    sources = policy.get("sources") if isinstance(policy.get("sources"), dict) else {}
    source = sources.get("dwpose")
    size = policy.get("image_size")
    if not source:
        return None, "dwpose_source_missing"
    if not isinstance(size, list) or len(size) < 2:
        return None, "image_size_missing"
    path = Path(str(source)).expanduser()
    if not path.is_file():
        return None, "dwpose_source_not_found"
    try:
        return _dwpose_points(_read_json(path), int(size[0]), int(size[1])), None
    except Exception as exc:
        return None, f"dwpose_load_failed:{type(exc).__name__}"


def _head_support_side_binding(
    points: dict[str, tuple[float, float] | None] | None,
    semantics: dict[str, Any],
) -> dict[str, Any]:
    if semantics.get("status") != "candidate":
        return {
            "status": "not_applicable",
            "anatomical_side": None,
            "authority": None,
            "reason": "no_head_support_semantic_candidate",
        }
    if not points:
        return {
            "status": "unresolved",
            "anatomical_side": None,
            "authority": None,
            "reason": "dwpose_points_unavailable",
        }

    visible_wrists = [side for side in ("left", "right") if points.get(f"{side}_wrist") is not None]
    details = {
        side: {
            "shoulder": points.get(f"{side}_shoulder") is not None,
            "elbow": points.get(f"{side}_elbow") is not None,
            "wrist": points.get(f"{side}_wrist") is not None,
        }
        for side in ("left", "right")
    }

    if len(visible_wrists) == 1:
        side = visible_wrists[0]
        proximal = bool(details[side]["shoulder"] or details[side]["elbow"])
        if proximal:
            return {
                "status": "bound",
                "anatomical_side": side,
                "authority": "single_observed_dwpose_wrist_with_same_side_proximal_arm_support",
                "reason": "head_support_semantic_requires_a_visible_hand_and_only_one_anatomically_named_wrist_is_observed",
                "observed_arm_chains": details,
                "forearm_side_publishable": bool(details[side]["elbow"] and details[side]["wrist"]),
            }

    return {
        "status": "unresolved",
        "anatomical_side": None,
        "authority": None,
        "reason": "head_support_side_requires_unambiguous_observed_wrist_binding",
        "observed_arm_chains": details,
        "visible_wrist_sides": visible_wrists,
        "forearm_side_publishable": False,
    }


def evaluate(
    fragment: dict[str, Any],
    policy: dict[str, Any],
    *,
    points: dict[str, tuple[float, float] | None] | None = None,
    points_error: str | None = None,
) -> dict[str, Any]:
    raw = _relationships(fragment)
    neutralized: list[str] = []
    leaks: list[dict[str, Any]] = []
    for text in raw:
        neutral, found = _neutralize_laterality(text)
        neutralized.append(neutral)
        if found:
            leaks.append({"source_text": text, "removed_tokens": found, "side_neutral_text": neutral})

    semantics = _head_support_semantics(neutralized)
    if points is None and points_error is None:
        points, points_error = _load_points(policy)
    binding = _head_support_side_binding(points, semantics)

    side = binding.get("anatomical_side") if binding.get("status") == "bound" else None
    canonical = semantics.get("canonical_side_neutral")
    composer_text = canonical
    if side and canonical:
        noun = "fist" if semantics.get("uses_fist_semantics") else "hand"
        composer_text = f"chin resting on the {side} {noun}"
        if semantics.get("forearm_support_explicit"):
            if binding.get("forearm_side_publishable"):
                composer_text += f", with the {side} forearm beneath/supporting the pose"
            else:
                composer_text += ", with the forearm beneath/supporting the pose"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": fragment.get("image_key") or policy.get("image_key"),
        "policy_mode": fragment.get("policy_mode") or ((policy.get("policy") or {}).get("mode") if isinstance(policy.get("policy"), dict) else None),
        "raw_relationships": raw,
        "side_neutral_relationships": neutralized,
        "laterality_leakage": leaks,
        "head_support": {
            **copy.deepcopy(semantics),
            "laterality_binding": binding,
            "composer_text": composer_text,
        },
        "dwpose_error": points_error,
        "invariants": {
            "qwen_laterality_is_never_trusted": True,
            "head_support_semantics_can_survive_without_broad_pose": True,
            "single_observed_wrist_can_bind_hand_side_only_when_same_side_proximal_arm_is_observed": True,
            "forearm_side_requires_same_side_elbow_and_wrist": True,
            "generic_forearm_held_language_does_not_create_head_support": True,
            "external_support_target_is_not_created": True,
            "no_model_calls": True,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow normalization for local configuration semantics and conservative head-support laterality binding.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--fragment-dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    fragment_dir = args.fragment_dir.expanduser().resolve() if args.fragment_dir else run_dir / DEFAULT_FRAGMENT_SUBDIR
    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR

    for label, path in (("run", run_dir), ("fragment", fragment_dir), ("policy", policy_dir)):
        if not path.is_dir():
            print(f"{label} directory not found: {path}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.only)
    fragment_paths = sorted(fragment_dir.glob("*.routed_fragments.json"))
    if requested:
        fragment_paths = [p for p in fragment_paths if p.name.removesuffix(".routed_fragments.json") in requested]
    if not fragment_paths:
        print(f"No matching routed fragments found in {fragment_dir}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    for fragment_path in fragment_paths:
        key = fragment_path.name.removesuffix(".routed_fragments.json")
        policy_path = policy_dir / f"{key}.perception_policy.json"
        if not policy_path.is_file():
            print(f"Missing policy for {key}: {policy_path}", file=sys.stderr)
            return 2
        out_path = output_dir / f"{key}.local_configuration_shadow.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            record = evaluate(_read_json(fragment_path), _read_json(policy_path))
            _write_json(out_path, record)
        records.append(record)
        hs = record.get("head_support") if isinstance(record.get("head_support"), dict) else {}
        binding = hs.get("laterality_binding") if isinstance(hs.get("laterality_binding"), dict) else {}
        print(
            f"{key}: head_support={hs.get('status')} side={binding.get('anatomical_side') or '-'} "
            f"laterality_leaks={len(record.get('laterality_leakage') or [])}"
        )

    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "fragment_dir": str(fragment_dir),
        "policy_dir": str(policy_dir),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "records": records,
    }
    _write_json(output_dir / "local_configuration_semantics_shadow.index.json", index)
    print(f"Index: {output_dir / 'local_configuration_semantics_shadow.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
