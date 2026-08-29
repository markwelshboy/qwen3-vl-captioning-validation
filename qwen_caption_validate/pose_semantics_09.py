from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import pose_semantics as base
from . import pose_semantics_05 as v05
from . import pose_semantics_08 as v08
from .runner import model_slug, resolve_model_id


_RECLINE_SCENE_RE = re.compile(
    r"\b(?:bed|bedding|bedspread|mattress|pillow|blanket|sheet|duvet|couch|sofa|"
    r"fabric|textile|soft surface|recliner)\b",
    re.I,
)
_BODY_SUPPORT_ACTOR_RE = re.compile(
    r"\b(?:head|neck|shoulders?|upper torso|torso|back|upper back|body|hips?|pelvis)\b",
    re.I,
)
_REST_SUPPORT_RE = re.compile(
    r"\b(?:support(?:s|ed|ing)?|rest(?:s|ed|ing)?|lying|lies|lay|against|on)\b",
    re.I,
)
_STANDING_LOAD_RE = re.compile(
    r"\b(?:stand(?:s|ing|ing upright)?|standing|bearing weight|weight[- ]?bearing|"
    r"foot flat on (?:the )?floor|feet flat on (?:the )?floor|standing on (?:the )?(?:foot|feet|floor|ground))\b",
    re.I,
)
_LEG_PART_RE = re.compile(r"\b(?:(left|right)[ _-]+)?(?:leg|upper legs?|thigh|foot|feet)\b", re.I)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _analysis_root(value: dict[str, Any]) -> dict[str, Any]:
    return v05._analysis_root(value)


def _probe_root(value: dict[str, Any] | None) -> dict[str, Any]:
    return v08._probe_root(value)


def _reclining_support_evidence(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Independent Analyze support compatible with lying/reclining.

    This is deliberately separate from the seated support ontology. A torso/head
    resting on bedding, a pillow, bedspread, couch, or other explicitly soft
    scene support can corroborate reclining when Analyze and gestalt already
    agree on the whole pose.
    """
    evidence: list[dict[str, Any]] = []
    subject = analysis.get("target_subject") or {}
    summary = str(analysis.get("image_summary") or "")
    summary_has_recline_scene = bool(_RECLINE_SCENE_RE.search(summary))

    for item in subject.get("interactions") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("evidence_status") or "observed").lower() not in {"observed", ""}:
            continue
        conf = v05._safe_float(item.get("confidence"), 0.0)
        if conf < 0.70:
            continue
        actor = str(item.get("actor_part") or "").replace("_", " ")
        target = str(item.get("target") or "")
        relation = str(item.get("type") or "").lower()
        notes = str(item.get("notes") or "")
        if not _BODY_SUPPORT_ACTOR_RE.search(actor):
            continue
        if relation not in {"support", "contact"}:
            continue
        target_is_scene = bool(_RECLINE_SCENE_RE.search(target))
        rest_like = relation == "support" or bool(_REST_SUPPORT_RE.search(notes))
        if rest_like and (target_is_scene or summary_has_recline_scene):
            evidence.append({
                "source": "analysis.target_subject.interactions",
                "actor_part": actor,
                "target": target,
                "relation": relation,
                "notes": notes,
                "confidence": conf,
            })

    for item in subject.get("visible_body_parts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("ownership") not in {None, "target"}:
            continue
        conf = v05._safe_float(item.get("confidence"), 0.0)
        if conf < 0.70:
            continue
        actor = str(item.get("part") or "").replace("_", " ")
        if not _BODY_SUPPORT_ACTOR_RE.search(actor):
            continue
        text = " ".join(str(item.get(field) or "") for field in ("support", "contact", "geometry"))
        if _REST_SUPPORT_RE.search(text) and (_RECLINE_SCENE_RE.search(text) or summary_has_recline_scene):
            evidence.append({
                "source": "analysis.target_subject.visible_body_parts",
                "actor_part": actor,
                "target": text,
                "confidence": conf,
            })

    for item in analysis.get("non_target_entities") or []:
        if not isinstance(item, dict):
            continue
        conf = v05._safe_float(item.get("confidence"), 0.0)
        if conf < 0.70:
            continue
        description = str(item.get("description") or "")
        relation = " ".join(str(item.get(field) or "") for field in ("contact", "support"))
        if not _RECLINE_SCENE_RE.search(description):
            continue
        if not _REST_SUPPORT_RE.search(relation):
            continue
        if not re.search(r"\b(?:subject|target|torso|head|back|body|lying|resting|supporting)\b", relation, re.I):
            continue
        evidence.append({
            "source": "analysis.non_target_entities",
            "actor_part": "target body",
            "target": description,
            "relation": relation,
            "confidence": conf,
        })

    return evidence


def _side_from_part(value: Any) -> str | None:
    text = str(value or "").lower().replace("_", " ")
    match = re.search(r"\b(left|right)\b", text)
    return match.group(1) if match else None


def _standing_load_evidence(analysis: dict[str, Any], dwpose: dict[str, Any]) -> list[dict[str, Any]]:
    """Direct visible-body semantics that establish a weight-bearing stance.

    Unlike a generic upright torso, these records explicitly say that a visible
    leg/foot is standing or bearing weight. DWPose is used only to confirm that
    the corresponding lower-body chain is actually present in the crop.
    """
    evidence: list[dict[str, Any]] = []
    subject = analysis.get("target_subject") or {}
    connectivity = v08._dwpose_target(dwpose).get("connectivity") or {}

    for item in subject.get("visible_body_parts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("ownership") not in {None, "target"}:
            continue
        conf = v05._safe_float(item.get("confidence"), 0.0)
        if conf < 0.70:
            continue

        part = str(item.get("part") or "").lower().replace("_", " ")
        text = " ".join(str(item.get(field) or "") for field in ("geometry", "support", "contact"))
        if not _LEG_PART_RE.search(part) or not _STANDING_LOAD_RE.search(text):
            continue

        side = _side_from_part(part)
        if side in {"left", "right"}:
            visible_count = int((connectivity.get(f"{side}_leg") or {}).get("visible_count") or 0)
            geometry_visible = visible_count >= 2
        else:
            geometry_visible = v08._bilateral_hip_knee_visible(dwpose)

        if not geometry_visible:
            continue
        evidence.append({
            "source": "analysis.target_subject.visible_body_parts",
            "part": part,
            "side": side,
            "text": text,
            "confidence": conf,
            "dwpose_lower_body_visible": True,
        })

    return evidence


def _baseline_hypothesis(result: dict[str, Any], label: str) -> dict[str, Any]:
    for item in (result.get("posture") or {}).get("hypotheses") or []:
        if isinstance(item, dict) and item.get("label") == label:
            return item
    return {}


def _qualify(
    result: dict[str, Any],
    label: str,
    score: float,
    route: str,
    support: list[str],
    *,
    limitations: list[str] | None = None,
) -> None:
    old = result.get("posture") or {}
    result["posture"] = {
        "status": "qualified",
        "label": label,
        "primitive_id": f"posture_{label}_v09",
        "support_score": round(max(0.0, min(1.0, score)), 3),
        "confidence_band": "strong" if score >= 0.80 else "moderate",
        "caption_preferred": True,
        "support": support,
        "limitations": limitations or [],
        "subsumes": ["component geometry/support evidence used only to establish the whole-pose primitive"],
        "hypotheses": old.get("hypotheses") or [],
        "authority": f"v09_{route}",
    }
    result.setdefault("preferred_pose", {})["posture"] = label
    result["posture_candidate"] = None


def _apply_v09_posture_refinement(
    result: dict[str, Any],
    dwpose: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None,
) -> None:
    analysis = _analysis_root(analysis_payload)
    gestalt = _probe_root(gestalt_payload)
    probe_posture = v08._normalize_probe_posture(gestalt.get("posture"))
    probe_conf = v05._safe_float(gestalt.get("posture_confidence"), 0.0)
    probe_basis = str(gestalt.get("posture_basis") or "unknown").lower()
    summary = str(analysis.get("image_summary") or "")

    current_posture = result.get("posture") or {}
    already_qualified = current_posture.get("status") == "qualified"
    candidate = result.get("posture_candidate") if isinstance(result.get("posture_candidate"), dict) else None

    recline_support = _reclining_support_evidence(analysis)
    standing_load = _standing_load_evidence(analysis, dwpose)
    seated_hyp = _baseline_hypothesis(result, "seated")
    seated_score = v05._safe_float(seated_hyp.get("support_score"), 0.0)

    route = "v08_preserved" if already_qualified else "withheld"
    valid = already_qualified
    reasons: list[str] = []
    vetoed_probe_posture: str | None = None

    if not already_qualified:
        if (
            probe_posture == "reclining"
            and probe_conf >= 0.85
            and v08._RECLINE_RE.search(summary)
            and recline_support
        ):
            route = "reclining_analyze_gestalt_plus_bedlike_support"
            valid = True
            reasons.append("Analyze and gestalt agree on reclining/lying, with independent observed head/torso support by a bed-like or soft support")
            _qualify(result, "reclining", probe_conf, route, reasons)

        elif (
            probe_posture == "standing"
            and probe_conf >= 0.85
            and standing_load
        ):
            route = "standing_gestalt_plus_direct_weight_bearing_support"
            valid = True
            reasons.append("Geometric standing gestalt is corroborated by Analyze visible-body records that directly describe a visible lower-body segment as standing or bearing weight")
            _qualify(result, "standing", probe_conf, route, reasons)

        elif (
            probe_posture == "squatting"
            and standing_load
            and not v08._SQUAT_RE.search(summary)
        ):
            route = "standing_weight_bearing_vetoes_squat_gestalt"
            valid = True
            vetoed_probe_posture = "squatting"
            reasons.append("Direct Analyze/DWPose weight-bearing stance evidence contradicts the uncorroborated squatting gestalt")
            _qualify(result, "standing", max(0.85, probe_conf - 0.05), route, reasons)

        elif (
            probe_posture == "seated"
            and probe_conf >= 0.82
            and v05._SEATED_RE.search(summary)
            and seated_score >= 0.40
        ):
            route = "seated_analyze_gestalt_plus_lower_body_geometry"
            valid = True
            reasons.append("Analyze and gestalt agree on seated posture and the independent bottom-up seated hypothesis reaches at least weak geometric support")
            _qualify(
                result,
                "seated",
                probe_conf,
                route,
                reasons,
                limitations=["lower-body geometry is compatible but not independently sufficient; whole-pose authority comes from cross-channel agreement"],
            )

    if valid and not already_qualified:
        result["posture_candidate"] = None
        if vetoed_probe_posture:
            result["vetoed_posture_candidate"] = {
                "label": vetoed_probe_posture,
                "status": "vetoed_candidate",
                "model_confidence": round(probe_conf, 3),
                "support_score": 0.0,
                "confidence_band": "withheld",
                "caption_preferred": False,
                "review_recommended": False,
                "authority": "independent_weight_bearing_evidence_veto",
                "support": reasons,
            }
    elif candidate is not None:
        result["posture_candidate"] = candidate

    probe = result.get("pose_gestalt_probe") or {}
    if vetoed_probe_posture:
        probe["caption_preferred"] = False
        probe["promotion_reason"] = "vetoed by independent direct weight-bearing stance evidence"
    elif valid and not already_qualified:
        probe["caption_preferred"] = True
        probe["promotion_reason"] = route
    result["pose_gestalt_probe"] = probe

    previous = result.get("pose_gestalt_corroboration")
    if previous is not None:
        result["pose_gestalt_corroboration_v08"] = previous
    result["pose_gestalt_corroboration"] = {
        "valid": bool(valid),
        "route": route,
        "probe_posture": probe_posture,
        "probe_confidence": round(probe_conf, 3),
        "probe_basis": probe_basis,
        "qualified_posture": (result.get("preferred_pose") or {}).get("posture"),
        "vetoed_probe_posture": vetoed_probe_posture,
        "reclining_support_count": len(recline_support),
        "reclining_support": recline_support,
        "direct_standing_load_count": len(standing_load),
        "direct_standing_load": standing_load,
        "baseline_seated_support_score": round(seated_score, 3),
        "policy": (
            "v0.9 extends v0.8 with bed-like reclining support, direct visible weight-bearing standing evidence, "
            "a standing-load veto for false squat gestalt, and Analyze+gestalt+weak lower-body geometry seated verification. "
            "The gestalt prompt remains unchanged and remains hypothesis-only."
        ),
    }


def build_pose_semantics(
    dwpose: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    gestalt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = v08.build_pose_semantics(dwpose, fused_payload, analysis_payload, gestalt_payload)
    _apply_v09_posture_refinement(result, dwpose, analysis_payload, gestalt_payload)

    result["human_summary"] = base._human_summary(
        result.get("posture") or {},
        result.get("torso_orientation") or {},
        result.get("gestures") or [],
        result.get("head_and_gaze") or [],
        result.get("framing") or {},
    )
    result["schema_version"] = "pose-semantics-0.9"
    result["status"] = "experimental_report_only"
    result["refinements"] = list(result.get("refinements") or []) + [
        "Reclining/lying support recognizes independently observed bed, bedding, pillow, bedspread, couch, sofa, fabric/textile, and other explicitly soft support rather than reusing the seated-only seat ontology.",
        "Standing verification can use direct visible lower-body weight-bearing semantics from Analyze when DWPose independently confirms that the corresponding lower-body segment exists in the crop.",
        "Direct weight-bearing stance evidence can veto an uncorroborated squatting gestalt, preventing a lifted knee from being promoted as a squat.",
        "Seated can be verified by Analyze+gestalt agreement plus an independently weak-but-compatible bottom-up seated geometry hypothesis, without requiring a named chair/table.",
    ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-pose-semantics-09",
        description="Pose semantics v0.9: reclining support, weight-bearing standing verification, and squat veto.",
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
    output_dir = (args.output_dir or (run_dir / "pose-semantics-v0.9" / slug)).expanduser().resolve()

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
        result.update({
            "image_key": key,
            "source_paths": {
                "fusion": str(fusion_path),
                "dwpose": str(dw_path),
                "analysis": str(analysis_path),
                "pose_gestalt": str(gestalt_path),
            },
        })
        _write(out_json, result)
        out_txt.write_text(str(result.get("human_summary") or "") + "\n", encoding="utf-8")
        records.append({
            "image_key": key,
            "status": "written",
            "posture": (result.get("preferred_pose") or {}).get("posture"),
            "posture_candidate": (result.get("posture_candidate") or {}).get("label") if isinstance(result.get("posture_candidate"), dict) else None,
            "verification_route": (result.get("pose_gestalt_corroboration") or {}).get("route"),
            "human_summary": result.get("human_summary"),
        })

    index = {
        "schema_version": "pose-semantics-0.9-run",
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
    print(f"Pose semantics v0.9: {output_dir}")
    for record in records:
        suffix = f" [candidate={record.get('posture_candidate')}]" if record.get("posture_candidate") else ""
        print(f"{record['image_key']}: {record.get('human_summary') or record.get('status')}{suffix}")
    return 0 if records else 2


if __name__ == "__main__":
    raise SystemExit(main())
