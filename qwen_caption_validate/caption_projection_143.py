from __future__ import annotations

import copy
import re
from typing import Any

from .caption_projection_142 import build_caption_projection as _build_142
from .caption_projection_142 import lint_caption as _lint_142

_STANDING_RE = re.compile(r"\b(?:stands?|standing|stood)\b", re.IGNORECASE)
_KNEE_KINEMATIC_RE = re.compile(
    r"(?:,?\s*)\b(?:knee\s+(?:is\s+)?(?:slightly\s+)?(?:bent|straight)|"
    r"(?:slightly\s+)?bent\s+knee|straight\s+knee|leg\s+(?:is\s+)?straight|straight\s+leg)\b",
    re.IGNORECASE,
)
_FRAMING_TERMS = {
    "close_up": re.compile(r"\bclose[- ]?up\b", re.I),
    "medium_close_up": re.compile(r"\bmedium\s+close[- ]?up\b|\bupper\s+chest\b|\bhead\s+(?:and|&)\s+shoulders\b", re.I),
    "medium": re.compile(r"\bmedium\s+(?:shot|framing)\b|\bwaist[- ]?up\b", re.I),
    "three_quarter": re.compile(r"\bthree[- ]quarter\b|\bmedium[- ]full\b|\bmid[- ]?thigh\b|\bupper\s+thighs?\b|\bknee[- ]?up\b", re.I),
    "near_full_length": re.compile(r"\bnear[- ]full[- ]length\b|\balmost\s+full[- ]length\b|\bmid[- ]?calf\b|\bfeet\s+(?:partially\s+)?cropped\b", re.I),
    "full_length": re.compile(r"\bfull[- ]length\b|\bfull[- ]body\b|\bhead\s+to\s+feet\b", re.I),
}
_GENERIC_FRAMING_RE = re.compile(
    r"\b(?:framed|framing|shot|portrait|close[- ]?up|full[- ]length|full[- ]body|"
    r"three[- ]quarter|medium[- ]full|waist[- ]?up|mid[- ]?thigh|mid[- ]?calf)\b",
    re.I,
)
_HAND_RE = re.compile(r"\bhand\b", re.I)
_CHIN_RE = re.compile(r"\bchin\b", re.I)
_HAND_OR_FIST_RE = re.compile(r"\b(?:hand|fist)\b", re.I)


def _fusion_root(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("fusion") if isinstance(payload.get("fusion"), dict) else payload
    return value if isinstance(value, dict) else {}


def _analysis_root(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the semantic Analyze body for both raw and wrapped Analyze JSON.

    Workspace .analysis.json files wrap the model result under ``analysis``.
    Some unit fixtures and older callers pass the result body directly. Projection
    must accept both layouts; otherwise semantic corroboration silently disappears
    when replaying real cached runs.
    """
    value = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else payload
    return value if isinstance(value, dict) else {}


def _pose(evidence: dict[str, Any]) -> dict[str, Any]:
    value = evidence.get("pose_orientation")
    return value if isinstance(value, dict) else evidence


def _projection_root(audit: dict[str, Any]) -> dict[str, Any]:
    value = audit.get("projection") if isinstance(audit.get("projection"), dict) else audit
    return value if isinstance(value, dict) else audit


def _framing_label_and_description(
    framing: dict[str, Any], fusion: dict[str, Any]
) -> tuple[str, str, str]:
    source_scale = str(framing.get("shot_scale") or "").lower()
    extent = str(framing.get("subject_extent") or "").strip()
    text = extent.lower()
    deterministic = fusion.get("deterministic_geometry") or {}
    pose_extent = str(deterministic.get("pose_extent_hint") or "").lower()

    # Explicit crop language outranks DWPose's coarse extent hint. DWPose calls
    # any visible ankle ``full_length`` even when the actual feet are cropped.
    if any(
        token in text
        for token in (
            "mid-calf",
            "mid calf",
            "feet partially cropped",
            "feet cropped",
            "ankles cropped",
        )
    ):
        return (
            "near_full_length",
            "near-full-length framing showing most of the body with the feet or lower legs cropped",
            extent,
        )
    if pose_extent == "full_length":
        return "full_length", "full-length framing showing essentially the whole body", extent
    if pose_extent == "three_quarter_or_long" or any(
        token in text
        for token in (
            "mid-thigh",
            "mid thighs",
            "mid-thighs",
            "upper thigh",
            "upper thighs",
            "to knee",
            "knees",
        )
    ):
        return (
            "three_quarter",
            "three-quarter or medium-full framing from around the thighs/knees to the head",
            extent,
        )
    if any(token in text for token in ("waist", "hips to", "hip to")):
        return "medium", "medium or waist-up framing", extent
    if any(
        token in text
        for token in (
            "upper chest",
            "head and shoulders",
            "head & shoulders",
            "shoulders and head",
        )
    ):
        return (
            "medium_close_up",
            "medium close-up framing centered on the head and upper chest/shoulders",
            extent,
        )
    if source_scale == "close_up" or "head to upper torso" in text:
        return "close_up", "close-up portrait framing centered on the head and upper torso", extent
    if source_scale in {"medium_close_up", "medium", "full_length"}:
        descriptions = {
            "medium_close_up": "medium close-up framing",
            "medium": "medium framing",
            "full_length": "full-length framing",
        }
        return source_scale, descriptions[source_scale], extent
    return source_scale or "unspecified", extent or "supported subject framing", extent


def _normalize_framing(
    evidence: dict[str, Any],
    fused_payload: dict[str, Any],
    projection: dict[str, Any],
) -> dict[str, Any] | None:
    framing_camera = evidence.setdefault("framing_camera", {})
    framing = framing_camera.setdefault("framing", {})
    if not isinstance(framing, dict):
        return None
    source_scale = framing.get("shot_scale")
    label, description, source_extent = _framing_label_and_description(
        framing, _fusion_root(fused_payload)
    )
    framing["source_shot_scale"] = source_scale
    framing["normalized_shot_scale"] = label
    framing["shot_scale"] = label
    framing["normalized_extent_description"] = description
    if source_extent:
        framing["source_subject_extent"] = source_extent
    projection.setdefault("allowed", []).append(
        {
            "path": "caption-evidence-1.3.framing_camera.framing",
            "reason": "subject_extent_and_deterministic_pose_extent_normalize_photographic_framing",
            "source_shot_scale": source_scale,
            "normalized_shot_scale": label,
            "normalized_extent_description": description,
        }
    )
    return {
        "id": "framing_subject_extent",
        "priority": "required",
        "normalized_shot_scale": label,
        "description": description,
        "source_subject_extent": source_extent,
        "instruction": (
            "State the supported crop/framing once in natural language. Prefer the "
            "normalized subject extent over a conflicting generic shot label."
        ),
    }


def _qualified_side(item: dict[str, Any]) -> str | None:
    state = item.get("fusion_v2") or {}
    side = state.get("qualified_anatomical_side") or item.get("anatomical_side")
    if state.get("laterality_selection_usable") and side in {"left", "right"}:
        return str(side)
    return None


def _sanitize_cropped_leg_kinematics(
    evidence: dict[str, Any],
    fused_payload: dict[str, Any],
    projection: dict[str, Any],
) -> None:
    """Do not verbalize knee angle or distal ground support without a full leg chain."""
    fusion = _fusion_root(fused_payload)
    connectivity = (fusion.get("deterministic_geometry") or {}).get("connectivity") or {}
    pose = _pose(evidence)
    for index, item in enumerate(pose.get("visible_subject_parts") or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("part") or "").lower().replace("_", " ")
        if not re.search(r"\b(?:leg|thigh|knee)\b", label):
            continue
        side = str(item.get("anatomical_side") or "").lower()
        if side not in {"left", "right"}:
            continue
        chain = connectivity.get(f"{side}_leg") or {}
        if bool(chain.get("complete")):
            continue

        geometry = item.get("geometry")
        if isinstance(geometry, str) and geometry:
            reduced = _KNEE_KINEMATIC_RE.sub("", geometry)
            reduced = re.sub(r"\s*,\s*,+", ", ", reduced)
            reduced = re.sub(r"^\s*[,;]\s*|\s*[,;]\s*$", "", reduced)
            reduced = re.sub(r"\s{2,}", " ", reduced).strip()
            if reduced != geometry:
                item["geometry"] = reduced or None
                projection.setdefault("blocked", []).append(
                    {
                        "path": f"caption-evidence-1.3.pose_orientation.visible_subject_parts[{index}].geometry",
                        "reason": "knee_angle_withheld_without_complete_hip_knee_ankle_chain",
                        "source_geometry": geometry,
                        "retained_geometry": item.get("geometry"),
                        "visible_leg_landmarks": list(chain.get("visible") or []),
                    }
                )

        support = item.get("support")
        if isinstance(support, str) and re.search(
            r"\b(?:sand|ground|floor|feet?|foot)\b", support, re.I
        ):
            item["support"] = None
            projection.setdefault("blocked", []).append(
                {
                    "path": f"caption-evidence-1.3.pose_orientation.visible_subject_parts[{index}].support",
                    "reason": "distal_ground_support_withheld_without_visible_ankle_foot_chain",
                    "source_support": support,
                    "visible_leg_landmarks": list(chain.get("visible") or []),
                }
            )


def _qualify_cropped_standing(
    evidence: dict[str, Any],
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    projection: dict[str, Any],
) -> dict[str, Any] | None:
    pose = _pose(evidence)
    posture = pose.setdefault(
        "whole_body_posture",
        {"allowed": [], "authority": "direct_visible_support_only", "evidence": []},
    )
    allowed = [str(value) for value in (posture.get("allowed") or [])]
    if "standing" in allowed:
        return None

    analysis_body = _analysis_root(analysis)
    image_summary = str(analysis_body.get("image_summary") or "")
    if not _STANDING_RE.search(image_summary):
        return None

    fusion = _fusion_root(fused_payload)
    standing_sides: set[str] = set()
    for item in fusion.get("qualified_body_parts") or []:
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        if not state.get("selection_usable"):
            continue
        label = str(item.get("part") or "").lower().replace("_", " ")
        if not re.search(r"\b(?:leg|thigh|knee)\b", label):
            continue
        text = " ".join(
            str(item.get(field) or "") for field in ("geometry", "support")
        )
        if not _STANDING_RE.search(text):
            continue
        side = _qualified_side(item)
        if side:
            standing_sides.add(side)
    if standing_sides != {"left", "right"}:
        return None

    connectivity = (fusion.get("deterministic_geometry") or {}).get("connectivity") or {}
    bilateral_hip_knee = all(
        int((connectivity.get(f"{side}_leg") or {}).get("visible_count") or 0) >= 2
        for side in ("left", "right")
    )
    if not bilateral_hip_knee:
        return None

    posture["allowed"] = sorted(set(allowed + ["standing"]))
    posture["authority"] = "semantic_standing_plus_bilateral_observed_hip_knee_chains"
    posture["evidence"] = list(posture.get("evidence") or []) + [
        "Analyze summary explicitly reports standing",
        "both governed leg records independently report standing",
        "DWPose observes bilateral hip-to-knee chains; feet may be cropped",
    ]
    projection.setdefault("allowed", []).append(
        {
            "path": "caption-evidence-1.3.pose_orientation.whole_body_posture.standing",
            "reason": "cropped_standing_qualified_by_semantic_and_bilateral_visible_leg_agreement",
        }
    )
    return {
        "id": "whole_body_posture_standing",
        "priority": "required",
        "description": "subject is standing",
        "instruction": (
            "State that the subject is standing. Cropped feet do not invalidate this "
            "qualified posture, but do not infer knee bend or exact foot/ground contact "
            "from cropped lower legs."
        ),
    }


def _salient_interaction_claims(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    pose = _pose(evidence)
    claims: list[dict[str, Any]] = []
    for index, item in enumerate(pose.get("qualified_interactions") or []):
        if not isinstance(item, dict):
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.85:
            continue
        interaction_type = str(item.get("type") or "").lower()
        actor = str(item.get("actor_part") or "body part").replace("_", " ")
        target = str(item.get("target") or "").replace("_", " ").strip()
        target_lower = target.lower()
        actor_lower = actor.lower()
        body_gesture = _HAND_RE.search(actor_lower) and target_lower in {
            "hip",
            "head",
            "chin",
            "face",
        }
        held_object = interaction_type in {
            "hold",
            "holding",
            "grip",
            "grasp",
            "carry",
            "carrying",
        }
        if not body_gesture and not held_object:
            continue
        side = str(item.get("actor_anatomical_side") or "unknown").lower()
        actor_phrase = (
            actor
            if re.search(r"\b(?:left|right)\b", actor, re.I)
            else (f"{side} {actor}" if side in {"left", "right"} else actor)
        )
        if interaction_type == "contact" and target_lower == "hip":
            description = f"{actor_phrase} rests on the hip"
        elif held_object:
            verb = {
                "holding": "holds",
                "hold": "holds",
                "carrying": "carries",
                "carry": "carries",
                "grip": "grips",
                "grasp": "grasps",
            }.get(interaction_type, interaction_type)
            description = f"{actor_phrase} {verb} {target}".strip()
        else:
            description = f"{actor_phrase} {interaction_type} {target}".strip()
        claims.append(
            {
                "id": f"salient_interaction_{index + 1}",
                "priority": "required",
                "description": description,
                "actor_part": actor_phrase,
                "target": target,
                "interaction_type": interaction_type,
                "instruction": (
                    "Express this high-confidence interaction once as a natural "
                    "pose/action phrase; do not replace it with generic arm geometry."
                ),
            }
        )
    return claims


def _chin_gesture_claim(
    evidence: dict[str, Any], projection: dict[str, Any]
) -> dict[str, Any] | None:
    pose = _pose(evidence)
    for index, item in enumerate(pose.get("visible_subject_parts") or []):
        if not isinstance(item, dict) or not _HAND_RE.search(str(item.get("part") or "")):
            continue
        geometry = str(item.get("geometry") or "")
        contact = str(item.get("contact") or "")
        support = str(item.get("support") or "")
        if not (
            _CHIN_RE.search(geometry + " " + contact)
            and re.search(r"\b(?:support|rest)\w*\b", support + " " + geometry, re.I)
        ):
            continue
        if not re.search(r"\b(?:curl\w*|closed|clench\w*|fist)\b", geometry, re.I):
            continue
        side = str(item.get("anatomical_side") or "unknown").lower()
        qualified = bool(item.get("laterality_qualified")) and side in {"left", "right"}
        hand_phrase = f"{side} hand" if qualified else "hand"
        if re.search(r"\b(?:clench\w*|fist)\b", geometry, re.I):
            relation = f"chin resting on the {side + ' ' if qualified else ''}fist"
        else:
            relation = f"chin resting on the {hand_phrase}"
        source = {
            "geometry": item.get("geometry"),
            "contact": item.get("contact"),
            "support": item.get("support"),
        }
        item["geometry"] = relation
        item["contact"] = "chin resting on hand"
        item["support"] = "hand supporting chin/head"
        pose.setdefault("gesture_semantics", []).append(
            {
                "type": "chin_rest_on_hand",
                "anatomical_side": side if qualified else "unknown",
                "relation": relation,
                "authority": "qualified_hand_chin_contact_plus_support_geometry",
            }
        )
        projection.setdefault("allowed", []).append(
            {
                "path": f"caption-evidence-1.3.pose_orientation.visible_subject_parts[{index}]",
                "reason": "recognizable_chin_on_hand_gesture_semantically_compresses_finger_level_support_chain",
                "source": source,
                "relation": relation,
            }
        )
        return {
            "id": "chin_rest_on_hand_gesture",
            "priority": "required",
            "description": relation,
            "instruction": (
                "Describe the recognizable gesture naturally as the chin resting on "
                "the hand; say fist only if fist/clenched is explicitly supported. "
                "Do not enumerate fingers under the chin."
            ),
        }
    return None


def _ensure_generic_scene_gestalt(evidence: dict[str, Any]) -> dict[str, Any] | None:
    if evidence.get("required_scene_claims"):
        return None
    scene = ((evidence.get("environment_lighting") or {}).get("scene") or {})
    environment = str(scene.get("environment_type") or "").lower()
    try:
        confidence = float(scene.get("environment_confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if environment not in {"indoor", "outdoor"} or confidence < 0.85:
        return None
    return {
        "id": "scene_gestalt_generic_1",
        "description": f"{environment} setting",
        "keywords": [environment, "indoors" if environment == "indoor" else "outdoors"],
        "minimum_keyword_matches": 1,
        "attribution": "scene_or_background_not_trigger_identity",
        "semantic_compression_allowed": True,
    }


def build_caption_projection(
    fused_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    caption_policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, audit = _build_142(fused_payload, analysis, caption_policy=caption_policy)
    evidence["projection_revision"] = "1.4.3"
    projection = _projection_root(audit)
    projection["schema_version"] = "caption-projection-audit-1.4.3"

    claims = [
        copy.deepcopy(item)
        for item in (evidence.get("required_claims") or [])
        if isinstance(item, dict)
    ]
    existing = {str(item.get("id") or "") for item in claims}

    # Standing qualification must inspect Fusion before subordinate standing text
    # is removed from caption-facing leg fields. Kinematic sanitization then keeps
    # only the high-level posture, not unsupported knee/foot detail.
    standing_claim = _qualify_cropped_standing(
        evidence, fused_payload, analysis, projection
    )
    _sanitize_cropped_leg_kinematics(evidence, fused_payload, projection)

    for claim in (
        _normalize_framing(evidence, fused_payload, projection),
        standing_claim,
        _chin_gesture_claim(evidence, projection),
    ):
        if claim and claim["id"] not in existing:
            claims.append(claim)
            existing.add(claim["id"])

    existing_descriptions = {item.get("description") for item in claims}
    for claim in _salient_interaction_claims(evidence):
        if claim["id"] not in existing and claim.get("description") not in existing_descriptions:
            claims.append(claim)
            existing.add(claim["id"])
            existing_descriptions.add(claim.get("description"))

    generic_scene = _ensure_generic_scene_gestalt(evidence)
    if generic_scene:
        evidence.setdefault("required_scene_claims", []).append(generic_scene)
        projection.setdefault("allowed", []).append(
            {
                "path": "caption-evidence-1.3.environment_lighting.scene.environment_type",
                "reason": "high_confidence_generic_scene_gestalt_is_must_cover_when_no_more_specific_setting_exists",
                "claim_id": generic_scene["id"],
            }
        )

    evidence["required_claims"] = claims
    projection.setdefault("notes", []).append(
        "Projection 1.4.3 prioritizes semantic salience: normalize framing from actual "
        "subject extent, qualify cropped standing only from three-way semantic/leg-chain "
        "agreement, withhold cropped knee/ground kinematics, require high-confidence pose "
        "interactions, and compress hand/chin support into a recognizable gesture."
    )
    return evidence, audit


def _claim(evidence: dict[str, Any], claim_id: str) -> dict[str, Any] | None:
    for item in evidence.get("required_claims") or []:
        if isinstance(item, dict) and item.get("id") == claim_id:
            return item
    return None


def _framing_claim_present(caption: str, claim: dict[str, Any]) -> bool:
    label = str(claim.get("normalized_shot_scale") or "")
    pattern = _FRAMING_TERMS.get(label)
    if pattern and pattern.search(caption):
        return True
    return bool(_GENERIC_FRAMING_RE.search(caption)) and label == "unspecified"


def _interaction_claim_present(caption: str, claim: dict[str, Any]) -> bool:
    actor = str(claim.get("actor_part") or "").lower()
    target = str(claim.get("target") or "").lower()
    if not target:
        return False
    target_token = re.escape(target.split()[0])
    actor_tokens = [
        tok
        for tok in re.findall(r"[a-z]+", actor)
        if tok in {"left", "right", "hand", "arm", "wrist"}
    ]
    actor_ok = all(
        re.search(rf"\b{re.escape(tok)}\b", caption, re.I)
        for tok in actor_tokens
        if tok in {"left", "right", "hand"}
    )
    return actor_ok and bool(re.search(rf"\b{target_token}\b", caption, re.I))


def lint_caption(caption: str, evidence: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(_lint_142(caption, evidence))
    violations = list(result.get("violations") or [])
    warnings = list(result.get("warnings") or [])

    framing = _claim(evidence, "framing_subject_extent")
    if framing and not _framing_claim_present(caption, framing):
        warnings.append(
            {
                "type": "required_claim_not_detected",
                "claim_id": "framing_subject_extent",
                "description": framing.get("description"),
            }
        )

    if _claim(evidence, "whole_body_posture_standing") and not _STANDING_RE.search(caption):
        warnings.append(
            {
                "type": "required_claim_not_detected",
                "claim_id": "whole_body_posture_standing",
                "description": "subject is standing",
            }
        )

    for claim in evidence.get("required_claims") or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or "")
        if claim_id.startswith("salient_interaction_") and not _interaction_claim_present(
            caption, claim
        ):
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": claim_id,
                    "description": claim.get("description"),
                }
            )

    if _claim(evidence, "chin_rest_on_hand_gesture"):
        chin_windows = list(re.finditer(r"\bchin\b", caption, re.I))
        gesture_ok = any(
            _HAND_OR_FIST_RE.search(caption[max(0, m.start() - 55) : m.end() + 55])
            for m in chin_windows
        )
        if not gesture_ok:
            warnings.append(
                {
                    "type": "required_claim_not_detected",
                    "claim_id": "chin_rest_on_hand_gesture",
                    "description": "chin resting on hand",
                }
            )

    def dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            key = (
                str(item.get("type") or ""),
                str(item.get("claim_id") or ""),
                str(item.get("text") or item.get("description") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    result["schema_version"] = "caption-authority-lint-1.4.3"
    result["violations"] = dedupe(violations)
    result["warnings"] = dedupe(warnings)
    result["violation_count"] = len(result["violations"])
    result["warning_count"] = len(result["warnings"])
    result["passed"] = not result["violations"]
    return result
