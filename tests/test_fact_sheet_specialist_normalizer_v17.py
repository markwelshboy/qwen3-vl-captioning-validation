from qwen_caption_validate import fact_sheet_specialist_normalizer_v17 as phase4b16


def _sheet():
    return {
        "schema_version": "caption-fact-sheet-0.2.15",
        "status": "ok",
        "image_key": "imageblind-01_00085",
        "facts": {
            "framing": {
                "available": True,
                "extent": "waist_or_upper_body",
                "source": "caption_perception_policy",
            },
            "body": {},
        },
        "sources": {},
        "reserved_domains": {},
        "audit": {"violations": [], "warnings": [], "invariants": {}},
    }


def test_promotes_anatomical_span_and_demotes_legacy_extent():
    framing = {
        "schema_version": "framing-semantics-1.0",
        "status": "ok",
        "image_key": "imageblind-01_00085",
        "authority": "deterministic_observation",
        "anatomical_span": {
            "status": "available",
            "upper_anchor": "shoulders",
            "lower_anchor": "hips",
            "upper_partial": "head",
            "lower_partial": "knees",
            "noncontiguous_observations": [],
            "composer_text": "framed from around the shoulders through the hips",
        },
        "standard_shot_scale": {
            "status": "withheld",
            "label": None,
            "composer_text": None,
        },
        "composer_framing": {
            "source": "anatomical_span",
            "composer_text": "framed from around the shoulders through the hips",
            "opening_template": "[[trigger]] is framed from around the shoulders through the hips",
        },
        "face_scale_geometry": {"status": "unavailable"},
        "person_scale_geometry": {"status": "available"},
        "face_person_height_ratio": None,
        "legacy_extent_hint": "waist_or_upper_body",
        "broad_pose_supported": False,
    }

    out = phase4b16._apply_framing_authority(_sheet(), framing)
    fact = out["facts"]["framing"]

    assert out["schema_version"] == "caption-fact-sheet-0.2.16"
    assert fact["anatomical_span"]["upper_anchor"] == "shoulders"
    assert fact["anatomical_span"]["lower_anchor"] == "hips"
    assert fact["composer_framing"]["composer_text"] == "framed from around the shoulders through the hips"
    assert fact["legacy"]["extent_hint"] == "waist_or_upper_body"
    assert fact["legacy"]["caption_authoritative"] is False
    assert out["reserved_domains"]["framing"]["owner"] == "framing-semantics-v1.0"


def test_authorized_scale_is_caption_surface_without_span_repetition():
    sheet = _sheet()
    sheet["image_key"] = "imageblind-01_00061"
    framing = {
        "status": "ok",
        "image_key": "imageblind-01_00061",
        "anatomical_span": {
            "status": "available",
            "upper_anchor": "head",
            "lower_anchor": "shoulders",
            "composer_text": "framed from the head through the shoulders",
        },
        "standard_shot_scale": {
            "status": "candidate",
            "label": "medium_close_up",
            "composer_text": "medium close-up",
        },
        "composer_framing": {
            "source": "standard_shot_scale",
            "composer_text": "medium close-up",
            "opening_template": "[[trigger]] is shown in a medium close-up",
            "anatomical_span_retained_internally": True,
        },
        "face_scale_geometry": {"status": "available"},
        "person_scale_geometry": {"status": "available"},
        "face_person_height_ratio": 0.393,
        "legacy_extent_hint": "close_or_medium_close",
        "broad_pose_supported": False,
    }

    out = phase4b16._apply_framing_authority(sheet, framing)
    fact = out["facts"]["framing"]

    assert fact["composer_framing"]["composer_text"] == "medium close-up"
    assert fact["anatomical_span"]["composer_text"] == "framed from the head through the shoulders"
    assert fact["composer_framing"]["opening_template"] == "[[trigger]] is shown in a medium close-up"


def test_mismatched_framing_key_marks_sheet_needs_review():
    framing = {
        "status": "ok",
        "image_key": "different",
    }
    out = phase4b16._apply_framing_authority(_sheet(), framing)
    assert out["status"] == "needs_review"
    assert "production_framing_image_key_mismatch" in out["audit"]["violations"]
