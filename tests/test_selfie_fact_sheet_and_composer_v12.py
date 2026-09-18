from qwen_caption_validate import fact_sheet_specialist_normalizer_v18 as fs18
from qwen_caption_validate import fact_sheet_text_composer_v12 as c12


def _base_sheet():
    return {
        "schema_version": "caption-fact-sheet-0.2.16",
        "status": "ok",
        "image_key": "imageblind-01_00031",
        "facts": {
            "body": {
                "configuration": [
                    {
                        "composer_text": "right hand holding a phone",
                        "promotion_status": "accepted_specialist_lateralized_candidate",
                    }
                ]
            }
        },
        "audit": {"invariants": {}},
    }


def _selfie_record(subtype, promoted=None):
    mirror = subtype == "mirror_selfie"
    return {
        "schema_version": "selfie-evidence-shadow-0.7",
        "status": "ok",
        "image_key": "imageblind-01_00031",
        "capture_style": {
            "family": "selfie",
            "subtype": subtype,
            "composer_text": "mirror selfie" if mirror else "selfie-style capture",
            "authority": "test_authority",
            "spatial_surface_mode": "depicted_frame" if mirror else "direct_camera_frame",
        },
        "mirror_caption_surface_policy": {
            "active": mirror,
            "spatial_surface_mode": "depicted_frame" if mirror else None,
            "directional_reference_system": "frame_relative_only" if mirror else None,
            "composer_anatomical_laterality": "withhold" if mirror else None,
        },
        "evidence": {
            "neutral_semantic": {"grade": "strong"},
            "mirror_selfie_semantic": {"grade": "strong" if mirror else "none"},
            "camera_viewpoint": {"grade": "none"},
            "foreground_arm": {"grade": "strong" if not mirror else "none"},
            "nearer_shoulder": {"anatomical_side_nearer": "right"},
        },
        "decision": {
            "status": "selfie_supported",
            "publishable_selfie": True,
            "capture_subtype": subtype,
            "promoted_fact_candidates": promoted
            or (["mirror selfie"] if mirror else [
                "selfie-style capture",
                "an outstretched arm extends into the lower-frame-left foreground",
                "the right shoulder is nearer the camera",
            ]),
        },
    }


def test_phase4b17_promotes_mirror_capture_without_mutating_internal_body_laterality():
    sheet = _base_sheet()
    out = fs18._apply_selfie_capture_authority(
        sheet,
        _selfie_record("mirror_selfie"),
    )

    capture = out["facts"]["capture"]
    assert out["schema_version"] == "caption-fact-sheet-0.2.17"
    assert capture["subtype"] == "mirror_selfie"
    assert capture["composer_text"] == "mirror selfie"
    assert capture["spatial_surface_mode"] == "depicted_frame"
    assert capture["composer_spatial_contract"]["anatomical_laterality"] == "withhold"

    # Fact-sheet internals retain anatomical authority. Suppression happens only
    # at the composer projection boundary.
    assert (
        out["facts"]["body"]["configuration"][0]["composer_text"]
        == "right hand holding a phone"
    )


def test_phase4b17_promotes_direct_capture_facts():
    out = fs18._apply_selfie_capture_authority(
        _base_sheet(),
        _selfie_record("direct_selfie"),
    )
    capture = out["facts"]["capture"]
    assert capture["subtype"] == "direct_selfie"
    assert "the right shoulder is nearer the camera" in capture["direct_capture_facts"]


def test_phase4b17_does_not_promote_withheld_selfie_candidate():
    selfie = _selfie_record("direct_selfie")
    selfie["decision"]["status"] = "candidate_not_publishable"
    selfie["decision"]["publishable_selfie"] = False

    out = fs18._apply_selfie_capture_authority(_base_sheet(), selfie)
    assert "capture" not in out["facts"]


def test_mirror_surface_neutralizes_anatomical_words_and_structured_sides_only():
    value = {
        "body": {
            "configuration": [
                "right hand holding a phone",
                "left knee raised high",
                "torso turned toward frame_left",
            ],
            "global_support_shape": {
                "overall_shape": "mostly_upright_over_support_leg",
                "support_side": "left",
                "elevated_side": "right",
            },
            "torso_orientation": {
                "turn_direction": "frame_right",
            },
        },
        "appearance": [
            "watch on left wrist",
            "dark shirt",
        ],
    }

    out = c12._mirror_surface_value(value)

    assert out["body"]["configuration"] == [
        "hand holding a phone",
        "knee raised high",
        "torso turned toward frame_left",
    ]
    assert "support_side" not in out["body"]["global_support_shape"]
    assert "elevated_side" not in out["body"]["global_support_shape"]
    assert out["body"]["torso_orientation"]["turn_direction"] == "frame_right"
    assert out["appearance"][0] == "watch on wrist"


def test_mirror_capture_projection_contains_only_depicted_frame_contract():
    capture = fs18._capture_fact_from_selfie(_selfie_record("mirror_selfie"))
    projected = c12._capture_projection(capture)

    assert projected["subtype"] == "mirror_selfie"
    assert projected["composer_text"] == "mirror selfie"
    assert projected["spatial_surface_mode"] == "depicted_frame"
    assert projected["directional_reference_system"] == "frame_relative_only"
    assert projected["anatomical_laterality_policy"] == "withhold"
    assert "composition_facts" not in projected


def test_direct_capture_projection_preserves_specialist_promoted_composition_facts():
    capture = fs18._capture_fact_from_selfie(_selfie_record("direct_selfie"))
    projected = c12._capture_projection(capture)

    assert projected["subtype"] == "direct_selfie"
    assert projected["composer_text"] == "selfie-style capture"
    assert (
        "an outstretched arm extends into the lower-frame-left foreground"
        in projected["composition_facts"]
    )
    assert "the right shoulder is nearer the camera" in projected["composition_facts"]


def test_mirror_caption_audit_rejects_anatomical_laterality():
    projection = {
        "subject": {},
        "authoritative_facts": {
            "capture": {
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
                "spatial_surface_mode": "depicted_frame",
                "directional_reference_system": "frame_relative_only",
                "anatomical_laterality_policy": "withhold",
            }
        },
        "omitted_review_conflict_domains": [],
    }

    audit = c12._caption_audit(
        "A mirror selfie with her left hand holding the phone.",
        projection,
    )

    assert "mirror_selfie_anatomical_laterality_forbidden" in audit["violations"]


def test_mirror_caption_audit_accepts_side_neutral_limb_wording():
    projection = {
        "subject": {},
        "authoritative_facts": {
            "capture": {
                "family": "selfie",
                "subtype": "mirror_selfie",
                "composer_text": "mirror selfie",
                "spatial_surface_mode": "depicted_frame",
                "directional_reference_system": "frame_relative_only",
                "anatomical_laterality_policy": "withhold",
            }
        },
        "omitted_review_conflict_domains": [],
    }

    audit = c12._caption_audit(
        "A mirror selfie with a hand holding the phone in front of the torso.",
        projection,
    )

    assert "mirror_selfie_capture_missing" not in audit["violations"]
    assert "mirror_selfie_anatomical_laterality_forbidden" not in audit["violations"]


def test_caption_audit_rejects_selfie_language_without_capture_authority():
    projection = {
        "subject": {},
        "authoritative_facts": {},
        "omitted_review_conflict_domains": [],
    }
    audit = c12._caption_audit("A casual selfie outdoors in a park.", projection)
    assert "selfie_language_without_capture_authority" in audit["violations"]
