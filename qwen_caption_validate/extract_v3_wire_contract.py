from __future__ import annotations

from typing import Any


CONFIDENCE_BANDS = {
    "h": 0.90,
    "m": 0.65,
    "l": 0.35,
    "u": 0.00,
}

LANDMARK_KEYS = {
    "hd": "head",
    "ls": "left_shoulder",
    "rs": "right_shoulder",
    "lh": "left_hip",
    "rh": "right_hip",
    "lk": "left_knee",
    "rk": "right_knee",
    "la": "left_ankle",
    "ra": "right_ankle",
}


def _confidence(value: Any) -> float:
    return CONFIDENCE_BANDS.get(str(value), 0.0)


def _entity_ref(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text == "t":
        return "target_subject"
    if text.startswith("e") and text[1:].isdigit():
        return f"entity_{int(text[1:]):02d}"
    return text


def _appearance_rows(rows: Any, start_index: int) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    index = start_index
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, list) or len(row) != 5:
            continue
        category, descriptors, frame_location, visibility, confidence = row
        out.append(
            {
                "id": f"appearance_{index:02d}",
                "category": str(category),
                "descriptors": list(descriptors or []),
                "frame_location": str(frame_location),
                "visibility": str(visibility),
                "confidence": _confidence(confidence),
            }
        )
        index += 1
    return out, index


def _landmark(row: Any) -> dict[str, Any]:
    visibility, confidence, evidence = row
    return {
        "visibility": visibility,
        "confidence": _confidence(confidence),
        "evidence": evidence,
    }


def _known_entity_ids(wire: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for row in wire.get("e") or []:
        if isinstance(row, list) and row:
            ids.add(str(row[0]))
    return ids


def _check_ref(value: Any, known: set[str], warnings: list[str], path: str) -> None:
    if value is None or value == "t":
        return
    text = str(value)
    if text not in known:
        warnings.append(f"{path}: reference {text!r} has no matching entity row")


def expand_extract_wire(wire: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand compact x3w1 transport JSON into canonical visual-extract-3.0.

    The transform is deliberately deterministic and semantic-free. It expands
    positional rows, stable short references and categorical confidence bands;
    it does not reinterpret image evidence or invent missing values.
    """
    if not isinstance(wire, dict) or wire.get("v") != "x3w1":
        raise ValueError("Expected Extract wire schema version x3w1")

    warnings: list[str] = []
    known_entities = _known_entity_ids(wire)

    f = wire["f"]
    s = wire["s"]
    sc = wire["sc"]
    co = wire["co"]
    h = wire["h"]

    clothing, next_appearance = _appearance_rows(s["cl"], 1)
    accessories, _ = _appearance_rows(s["ac"], next_appearance)

    visible_body_parts = []
    for row in s["bp"]:
        (
            part,
            side,
            ownership,
            visibility,
            visible_subparts,
            connectivity,
            geometry_cues,
            contact_cues,
            frame_location,
            confidence,
        ) = row
        visible_body_parts.append(
            {
                "part": part,
                "reported_anatomical_side": side,
                "ownership_candidate": ownership,
                "visibility": visibility,
                "visible_subparts": visible_subparts,
                "connectivity_to_target_chain": connectivity,
                "geometry_cues": geometry_cues,
                "contact_cues": contact_cues,
                "frame_location": frame_location,
                "confidence": _confidence(confidence),
            }
        )

    landmarks = {
        canonical_name: _landmark(s["lm"][wire_name])
        for wire_name, canonical_name in LANDMARK_KEYS.items()
    }

    interactions = []
    for index, row in enumerate(s["ix"]):
        kind, actor_part, ownership, target_ref, target_text, evidence, confidence, cues = row
        _check_ref(target_ref, known_entities, warnings, f"s.ix.{index}.target_ref")
        interactions.append(
            {
                "type": kind,
                "actor_part": actor_part,
                "actor_ownership_candidate": ownership,
                "target_ref": _entity_ref(target_ref),
                "target_text": target_text,
                "evidence_status": evidence,
                "confidence": _confidence(confidence),
                "cues": cues,
            }
        )

    entities = []
    entity_ref_map: dict[str, str] = {}
    for row in wire["e"]:
        wire_id, kind, cls, descriptors, visibility, frame_location, depth_band, confidence = row
        canonical_id = _entity_ref(wire_id)
        entity_ref_map[str(wire_id)] = str(canonical_id)
        entities.append(
            {
                "id": canonical_id,
                "type": kind,
                "class": cls,
                "descriptors": descriptors,
                "visibility": visibility,
                "frame_location": frame_location,
                "depth_band": depth_band,
                "confidence": _confidence(confidence),
            }
        )

    relations = []
    for index, row in enumerate(wire["r"]):
        subject_ref, predicate, object_ref, object_text, evidence, confidence, cues = row
        _check_ref(subject_ref, known_entities, warnings, f"r.{index}.subject_ref")
        _check_ref(object_ref, known_entities, warnings, f"r.{index}.object_ref")
        relations.append(
            {
                "subject_ref": _entity_ref(subject_ref),
                "predicate": predicate,
                "object_ref": _entity_ref(object_ref),
                "object_text": object_text,
                "evidence_status": evidence,
                "confidence": _confidence(confidence),
                "cues": cues,
            }
        )

    env = sc["env"]
    ill = sc["ill"]
    bg = sc["bg"]

    background_regions = [
        {
            "description": row[0],
            "relation_to_subject": row[1],
            "frame_location": row[2],
            "evidence_status": row[3],
            "confidence": _confidence(row[4]),
        }
        for row in sc["br"]
    ]

    nuisance_regions = [
        {
            "description": row[0],
            "frame_location": row[1],
            "frame_coverage": row[2],
            "texture_complexity": row[3],
            "structural_complexity": row[4],
            "specular_reflective": row[5],
            "entropy_focus_candidate": row[6],
        }
        for row in sc["nr"]
    ]

    p = h["p"]
    torso = h["to"]
    head = h["ho"]
    head_body = h["hb"]
    camera = h["cam"]
    capture = h["cap"]

    support_context = []
    for index, row in enumerate(h["sup"]):
        relation, target_ref, target_description, evidence, confidence, cues = row
        _check_ref(target_ref, known_entities, warnings, f"h.sup.{index}.target_ref")
        support_context.append(
            {
                "subject_relation": relation,
                "target_ref": _entity_ref(target_ref),
                "target_description": target_description,
                "evidence_status": evidence,
                "confidence": _confidence(confidence),
                "cues": cues,
            }
        )

    actions = [
        {
            "value": row[0],
            "confidence": _confidence(row[1]),
            "cues": row[2],
            "limitations": row[3],
        }
        for row in h["act"]
    ]

    canonical = {
        "schema_version": "visual-extract-3.0",
        "image_overview": wire["o"],
        "framing": {
            "shot_scale_candidate": f[0],
            "visible_extent": f[1],
            "subject_frame_coverage": f[2],
            "frame_observations": f[3],
        },
        "target_subject": {
            "entity_ref": "target_subject",
            "transient_appearance": {
                "clothing": clothing,
                "accessories": accessories,
                "hair_state": s["hs"],
                "expression_state": s["ex"],
            },
            "visible_body_parts": visible_body_parts,
            "geometry_landmark_visibility": landmarks,
            "orientation_cues": {
                "torso": s["or"][0],
                "head": s["or"][1],
                "image_plane_body_axis": s["or"][2],
            },
            "gaze": {
                "target_candidate": s["g"][0],
                "image_direction": s["g"][1],
                "confidence": _confidence(s["g"][2]),
                "cues": s["g"][3],
            },
            "interactions": interactions,
        },
        "entities": entities,
        "relations": relations,
        "scene": {
            "environment_candidate": env[0],
            "environment_confidence": _confidence(env[1]),
            "environment_cues": env[2],
            "environment_counterevidence": env[3],
            "illumination": {
                "type": ill[0],
                "directionality": ill[1],
                "contrast": ill[2],
                "observations": ill[3],
            },
            "background_structure": {
                "texture_complexity": bg[0],
                "structural_complexity": bg[1],
                "specular_reflective": bg[2],
                "repeated_geometry": bg[3],
                "strong_lines_or_angles": bg[4],
                "reflections_present": bg[5],
                "observations": bg[6],
            },
            "background_regions": background_regions,
            "nuisance_regions": nuisance_regions,
        },
        "composition_observations": {
            "subject_dominance": co[0],
            "foreground_relations": co[1],
            "visual_thrust_cues": co[2],
        },
        "hypotheses": {
            "posture": {
                "value": p[0], "confidence": _confidence(p[1]), "cues": p[2], "limitations": p[3]
            },
            "torso_orientation": {
                "orientation_band": torso[0], "body_faces_frame": torso[1], "confidence": _confidence(torso[2]),
                "cues": torso[3], "limitations": torso[4]
            },
            "head_orientation": {
                "yaw": head[0], "pitch": head[1], "roll": head[2], "confidence": _confidence(head[3]),
                "cues": head[4], "limitations": head[5]
            },
            "head_body_relation": {
                "value": head_body[0], "confidence": _confidence(head_body[1]),
                "cues": head_body[2], "limitations": head_body[3]
            },
            "camera": {
                "elevation": camera[0], "pitch": camera[1], "confidence": _confidence(camera[2]),
                "cues": camera[3], "counterevidence": camera[4]
            },
            "capture": {
                "mode": capture[0], "confidence": _confidence(capture[1]), "cues": capture[2]
            },
            "support_context": support_context,
            "actions": actions,
        },
        "uncertainties": wire["u"],
    }

    metadata = {
        "wire_schema_version": "x3w1",
        "confidence_band_mapping": dict(CONFIDENCE_BANDS),
        "entity_ref_mapping": entity_ref_map,
        "warnings": warnings,
    }
    return canonical, metadata
