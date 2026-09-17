from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "framing-semantics-shadow-0.1"
DEFAULT_POLICY_SUBDIR = Path("semantic-v3") / "caption-perception-policy-v0.1"
DEFAULT_FACT_SUBDIRS = (
    Path("semantic-v3") / "caption-fact-sheet-v0.2.14",
    Path("semantic-v3") / "caption-fact-sheet-v0.2.13",
)
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "framing-semantics-shadow-v0.1"

TIER_ORDER = ("head", "shoulders", "hips", "knees", "ankles")
TIER_LABELS = {
    "head": "head",
    "shoulders": "shoulders",
    "hips": "hips",
    "knees": "knees",
    "ankles": "ankles",
}

# These thresholds are intentionally used only for the OPTIONAL photographic
# shot-scale candidate.  The canonical anatomical span below does not depend on
# them.  The purpose of this shadow is to let us validate whether conventional
# terms such as close-up / medium are trustworthy before promoting them.
EXTREME_CLOSE_DOMINANCE = 0.55
CLOSE_DOMINANCE = 0.62
MEDIUM_CLOSE_MIN_HEIGHT = 0.38
MEDIUM_MIN_HEIGHT = 0.45
MEDIUM_WIDE_MIN_HEIGHT = 0.45


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


def _landmarks(policy: dict[str, Any]) -> set[str]:
    visibility = policy.get("visibility") if isinstance(policy.get("visibility"), dict) else {}
    raw = visibility.get("observed_landmarks")
    if not isinstance(raw, list):
        return set()
    return {str(x) for x in raw if x}


def _tier_evidence(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    visible = _landmarks(policy)
    face_names = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}
    face_count = len(face_names & visible)
    head_any = bool(face_count or "neck" in visible)

    def paired(left: str, right: str) -> dict[str, Any]:
        names = [name for name in (left, right) if name in visible]
        count = len(names)
        state = "strong" if count == 2 else ("partial" if count == 1 else "absent")
        return {"state": state, "count": count, "landmarks": names}

    # A neck point by itself is not a strong observation of the head.  At least
    # two actual facial landmarks are required before the head can anchor the
    # upper end of the published anatomical span.
    head_state = "strong" if face_count >= 2 else ("partial" if head_any else "absent")
    head_landmarks = [name for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck") if name in visible]

    return {
        "head": {"state": head_state, "count": len(head_landmarks), "face_landmark_count": face_count, "landmarks": head_landmarks},
        "shoulders": paired("left_shoulder", "right_shoulder"),
        "hips": paired("left_hip", "right_hip"),
        "knees": paired("left_knee", "right_knee"),
        "ankles": paired("left_ankle", "right_ankle"),
    }


def _core_span(tiers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    states = {tier: str((tiers.get(tier) or {}).get("state") or "absent") for tier in TIER_ORDER}
    strong_indices = [i for i, tier in enumerate(TIER_ORDER) if states[tier] == "strong"]
    if not strong_indices:
        partial = [tier for tier in TIER_ORDER if states[tier] == "partial"]
        return {
            "status": "unavailable",
            "upper_anchor": None,
            "lower_anchor": None,
            "upper_partial": None,
            "lower_partial": None,
            "noncontiguous_observations": partial,
            "composer_text": None,
            "reason": "no_strong_anatomical_anchor",
        }

    start = strong_indices[0]
    end = start
    for idx in range(start + 1, len(TIER_ORDER)):
        tier = TIER_ORDER[idx]
        if states[tier] != "strong":
            break
        end = idx

    upper = TIER_ORDER[start]
    lower = TIER_ORDER[end]
    upper_partial = TIER_ORDER[start - 1] if start > 0 and states[TIER_ORDER[start - 1]] == "partial" else None
    lower_partial = TIER_ORDER[end + 1] if end + 1 < len(TIER_ORDER) and states[TIER_ORDER[end + 1]] == "partial" else None

    noncontiguous: list[str] = []
    for idx in range(end + 1, len(TIER_ORDER)):
        tier = TIER_ORDER[idx]
        if tier == lower_partial:
            continue
        if states[tier] in {"strong", "partial"}:
            noncontiguous.append(tier)

    if upper == lower:
        if upper == "head":
            text = "framed tightly around the head and face"
        else:
            text = f"framed around the {TIER_LABELS[upper]}"
    else:
        prefix = "framed from the" if upper == "head" else "framed from around the"
        text = f"{prefix} {TIER_LABELS[upper]} through the {TIER_LABELS[lower]}"

    return {
        "status": "available",
        "upper_anchor": upper,
        "lower_anchor": lower,
        "upper_partial": upper_partial,
        "lower_partial": lower_partial,
        "noncontiguous_observations": noncontiguous,
        "composer_text": text,
        "reason": "deepest_contiguous_strong_anatomical_span",
    }


def _bbox(policy: dict[str, Any]) -> dict[str, float | None]:
    visibility = policy.get("visibility") if isinstance(policy.get("visibility"), dict) else {}
    raw = visibility.get("observed_bbox") if isinstance(visibility.get("observed_bbox"), dict) else {}

    def number(key: str) -> float | None:
        value = raw.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return {"width_fraction": number("width_fraction"), "height_fraction": number("height_fraction")}


def _authoritative_pose_text(fact: dict[str, Any] | None) -> str | None:
    if not isinstance(fact, dict):
        return None
    facts = fact.get("facts") if isinstance(fact.get("facts"), dict) else {}
    body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    adjudication = body.get("broad_pose_adjudication") if isinstance(body.get("broad_pose_adjudication"), dict) else {}
    if adjudication.get("composer_authoritative"):
        text = _clean(adjudication.get("canonical_pose_text"))
        if text:
            return text
    pose = body.get("pose_candidate") if isinstance(body.get("pose_candidate"), dict) else {}
    status = str(pose.get("promotion_status") or "")
    if status.startswith("accepted_specialist_"):
        for key in ("composer_text", "normalized_text", "text"):
            text = _clean(pose.get(key))
            if text:
                return text
    return None


def _pose_family(text: str | None) -> str | None:
    if not text:
        return None
    folded = text.casefold()
    if "stand" in folded:
        return "standing"
    if "seat" in folded or "sitting" in folded or " sits" in f" {folded}":
        return "seated"
    if "squat" in folded:
        return "squatting"
    if "crouch" in folded:
        return "crouching"
    if "kneel" in folded:
        return "kneeling"
    if "reclin" in folded:
        return "reclining"
    if "lying" in folded or "lies" in folded:
        return "lying"
    return None


def _shot_scale_candidate(
    tiers: dict[str, dict[str, Any]],
    span: dict[str, Any],
    bbox: dict[str, float | None],
    *,
    authoritative_pose: str | None,
) -> dict[str, Any]:
    states = {tier: str((tiers.get(tier) or {}).get("state") or "absent") for tier in TIER_ORDER}
    h = bbox.get("height_fraction")
    w = bbox.get("width_fraction")
    dominance = max([value for value in (h, w) if value is not None], default=None)
    pose_family = _pose_family(authoritative_pose)

    def candidate(label: str, basis: list[str]) -> dict[str, Any]:
        surface = {
            "extreme_close_up": "extreme close-up",
            "close_up": "close-up",
            "medium_close_up": "medium close-up",
            "medium": "medium shot",
            "medium_wide": "medium-wide shot",
            "full_body": "full-body shot",
        }[label]
        return {
            "status": "candidate",
            "label": label,
            "composer_text": surface,
            "basis": basis,
            "pose_family": pose_family,
            "note": "Optional photographic language only; canonical crop truth remains the anatomical span.",
        }

    if states["head"] != "strong":
        return {
            "status": "withheld",
            "label": None,
            "composer_text": None,
            "basis": ["head_not_strong"],
            "pose_family": pose_family,
            "note": "Conventional shot scale withheld; use anatomical span instead.",
        }

    if states["hips"] == "absent" and states["shoulders"] in {"absent", "partial"}:
        if dominance is not None and dominance >= EXTREME_CLOSE_DOMINANCE:
            return candidate("extreme_close_up", ["head_strong", "shoulders_not_strong", f"landmark_bbox_dominance={dominance:.3f}"])
        if dominance is not None and dominance >= 0.35:
            return candidate("close_up", ["head_strong", "shoulders_not_strong", f"landmark_bbox_dominance={dominance:.3f}"])

    if states["hips"] == "absent" and states["shoulders"] == "strong":
        if dominance is not None and dominance >= CLOSE_DOMINANCE:
            return candidate("close_up", ["head_and_shoulders_strong", "hips_absent", f"landmark_bbox_dominance={dominance:.3f}"])
        if h is not None and h >= MEDIUM_CLOSE_MIN_HEIGHT:
            return candidate("medium_close_up", ["head_and_shoulders_strong", "hips_absent", f"landmark_bbox_height={h:.3f}"])

    if states["hips"] == "strong" and states["knees"] == "absent" and h is not None and h >= MEDIUM_MIN_HEIGHT:
        return candidate("medium", ["head_through_hips_strong", "knees_absent", f"landmark_bbox_height={h:.3f}"])

    if states["knees"] == "strong" and states["ankles"] == "absent" and h is not None and h >= MEDIUM_WIDE_MIN_HEIGHT:
        return candidate("medium_wide", ["head_through_knees_strong", "ankles_absent", f"landmark_bbox_height={h:.3f}"])

    # "Full body" is especially pose-dependent.  Only surface it in shadow
    # when the complete head->ankle chain is strong AND an authoritative broad
    # pose says the person is standing.  Non-standing images keep the literal
    # anatomical span instead of being forced into upright portrait vocabulary.
    if (
        span.get("upper_anchor") == "head"
        and span.get("lower_anchor") == "ankles"
        and pose_family == "standing"
    ):
        return candidate("full_body", ["head_through_both_ankles_strong", "authoritative_pose_family=standing"])

    return {
        "status": "withheld",
        "label": None,
        "composer_text": None,
        "basis": ["no_conservative_standard_scale_rule_matched"],
        "pose_family": pose_family,
        "note": "Conventional shot scale withheld; use anatomical span instead.",
    }


def _pose_gate_shadow(policy: dict[str, Any], tiers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hips_strong = (tiers.get("hips") or {}).get("state") == "strong"
    knees_strong = (tiers.get("knees") or {}).get("state") == "strong"
    broad = bool(hips_strong and knees_strong)

    geometry = policy.get("geometry") if isinstance(policy.get("geometry"), dict) else {}
    sam = policy.get("sam3d") if isinstance(policy.get("sam3d"), dict) else {}
    config_score = int(geometry.get("configuration_score") or 0)
    complexity = int(geometry.get("pose_complexity_score") or 0)
    guidance_usable = bool(sam.get("available") and int(sam.get("projected_selected_joint_count") or 0) >= 8)

    if broad:
        mode = "pose_guided" if complexity >= 2 and guidance_usable else "pose_allowed"
    else:
        mode = "configuration" if config_score >= 2 else "framing_only"

    legacy_visibility = policy.get("visibility") if isinstance(policy.get("visibility"), dict) else {}
    legacy_broad = bool(legacy_visibility.get("broad_pose_supported"))
    legacy_mode = str((policy.get("policy") or {}).get("mode") or "")
    return {
        "broad_pose_supported": broad,
        "proposed_mode": mode,
        "changed_from_legacy": broad != legacy_broad or mode != legacy_mode,
        "legacy_broad_pose_supported": legacy_broad,
        "legacy_mode": legacy_mode,
        "reason": (
            "requires_both_hips_and_both_knees_as_strong_observed_crop_evidence"
            if broad
            else "withholds_broad_pose_without_bilateral_hip_and_knee_visibility"
        ),
    }


def evaluate(policy: dict[str, Any], fact: dict[str, Any] | None = None) -> dict[str, Any]:
    tiers = _tier_evidence(policy)
    span = _core_span(tiers)
    bbox = _bbox(policy)
    pose_text = _authoritative_pose_text(fact)
    scale = _shot_scale_candidate(tiers, span, bbox, authoritative_pose=pose_text)
    pose_gate = _pose_gate_shadow(policy, tiers)

    span_text = _clean(span.get("composer_text"))
    scale_text = _clean(scale.get("composer_text"))
    if scale_text and span_text:
        opening = f"[[trigger]] is shown in a {scale_text}, {span_text}"
    elif span_text:
        if span_text.startswith("framed "):
            opening = "[[trigger]] is " + span_text
        else:
            opening = "[[trigger]] is shown " + span_text
    elif scale_text:
        opening = f"[[trigger]] is shown in a {scale_text}"
    else:
        opening = None

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "image_key": policy.get("image_key"),
        "legacy": {
            "extent_hint": (policy.get("visibility") or {}).get("extent_hint") if isinstance(policy.get("visibility"), dict) else None,
            "broad_pose_supported": bool((policy.get("visibility") or {}).get("broad_pose_supported")) if isinstance(policy.get("visibility"), dict) else False,
            "mode": (policy.get("policy") or {}).get("mode") if isinstance(policy.get("policy"), dict) else None,
        },
        "tier_evidence": tiers,
        "observed_bbox": bbox,
        "anatomical_span": span,
        "standard_shot_scale": scale,
        "authoritative_pose_text": pose_text,
        "pose_gate_shadow": pose_gate,
        "proposed_opening_template": opening,
        "invariants": {
            "canonical_framing_is_anatomical_span_not_portrait_bucket": True,
            "single_outlier_joint_cannot_extend_core_span": True,
            "standard_shot_scale_is_optional": True,
            "standard_shot_scale_never_overrides_anatomical_span": True,
            "broad_pose_gate_is_independent_of_framing_label": True,
            "no_model_calls": True,
        },
    }


def _fact_dir(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_dir() else None
    for rel in DEFAULT_FACT_SUBDIRS:
        candidate = run_dir / rel
        if candidate.is_dir():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shadow probe for anatomical-span framing and conservative standard shot-scale language.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--policy-dir", type=Path)
    p.add_argument("--fact-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    policy_dir = args.policy_dir.expanduser().resolve() if args.policy_dir else run_dir / DEFAULT_POLICY_SUBDIR
    if not policy_dir.is_dir():
        print(f"Perception-policy directory not found: {policy_dir}", file=sys.stderr)
        return 2

    fact_dir = _fact_dir(run_dir, args.fact_dir)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.only)
    paths = sorted(policy_dir.glob("*.perception_policy.json"))
    if requested:
        paths = [p for p in paths if p.name.removesuffix(".perception_policy.json") in requested]
    if not paths:
        print(f"No matching perception-policy records found in {policy_dir}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    for policy_path in paths:
        key = policy_path.name.removesuffix(".perception_policy.json")
        out_path = output_dir / f"{key}.framing_shadow.json"
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            policy = _read_json(policy_path)
            fact = None
            if fact_dir is not None:
                fact_path = fact_dir / f"{key}.fact_sheet.json"
                if fact_path.is_file():
                    fact = _read_json(fact_path)
            record = evaluate(policy, fact)
            _write_json(out_path, record)
        records.append(record)
        span = record.get("anatomical_span") if isinstance(record.get("anatomical_span"), dict) else {}
        scale = record.get("standard_shot_scale") if isinstance(record.get("standard_shot_scale"), dict) else {}
        gate = record.get("pose_gate_shadow") if isinstance(record.get("pose_gate_shadow"), dict) else {}
        print(
            f"{key}: span={span.get('upper_anchor')}->{span.get('lower_anchor')} "
            f"scale={scale.get('label') or '-'} broad={gate.get('broad_pose_supported')} mode={gate.get('proposed_mode')}"
        )

    scale_counts = Counter(
        str((r.get("standard_shot_scale") or {}).get("label") or "withheld")
        for r in records
    )
    mode_counts = Counter(
        str((r.get("pose_gate_shadow") or {}).get("proposed_mode") or "unknown")
        for r in records
    )
    changed = [str(r.get("image_key")) for r in records if (r.get("pose_gate_shadow") or {}).get("changed_from_legacy")]
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "policy_dir": str(policy_dir),
        "fact_dir": str(fact_dir) if fact_dir else None,
        "output_dir": str(output_dir),
        "record_count": len(records),
        "standard_shot_scale_counts": dict(sorted(scale_counts.items())),
        "proposed_mode_counts": dict(sorted(mode_counts.items())),
        "pose_gate_changed_keys": changed,
        "records": records,
    }
    _write_json(output_dir / "framing_semantics_shadow.index.json", index)
    print(f"Index: {output_dir / 'framing_semantics_shadow.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
