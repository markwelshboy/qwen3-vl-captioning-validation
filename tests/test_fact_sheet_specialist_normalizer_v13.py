from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v13 as mod


def _points():
    return {
        "left_hip": (0.0, 0.0),
        "left_knee": (0.0, 1.0),
        "left_ankle": (1.0, 1.0),
        "right_hip": (10.0, 0.0),
        "right_knee": (10.0, 1.0),
        "right_ankle": (11.0, 1.0),
    }


def _projected(left=120.0, right=110.0, authority=0.95):
    return {
        "pose": "crouching",
        "best_candidate_pose": "crouching",
        "geometry": {
            "left_knee_angle_deg": left,
            "right_knee_angle_deg": right,
        },
        "region_support": {"knees": authority},
        "assertion_authority": {"selected_path": "observed_crouch_hip_knee_chain"},
    }


def _sheet(configuration):
    return {
        "schema_version": "caption-fact-sheet-0.2.11",
        "image_key": "imageblind-01_00066",
        "policy": {"mode": "pose_guided"},
        "facts": {"body": {"configuration": configuration}},
        "audit": {"invariants": {}},
    }


def test_defaults():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.12"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.12"
    assert mod.MAX_DEEP_KNEE_ANGLE_DEG == 135.0
    assert mod.MIN_KNEE_REGION_AUTHORITY == 0.80


def test_00066_like_singular_knee_bend_is_replaced():
    sheet = _sheet([
        {"text": "torso bent forward", "composer_text": "upper body bent forward from the hips"},
        {"text": "one knee slightly bent", "composer_text": "one knee slightly bent"},
    ])
    out = mod._apply_bilateral_knee_flexion_authority(
        sheet,
        projected=_projected(),
        dwpose_points=_points(),
    )
    body = out["facts"]["body"]
    assert [x["composer_text"] for x in body["configuration"]] == [
        "upper body bent forward from the hips",
        "both knees bent",
    ]
    knee = body["configuration"][1]
    assert knee["promotion_status"] == "accepted_specialist_bilateral_knee_flexion_candidate"
    assert knee["specialist_owner"] == "sam3d_v16_bilateral_knee_flexion_specialist"
    assert knee["bilateral_knee_flexion_binding"]["sam3d_left_knee_angle_deg"] == 120.0
    adj = body["bilateral_knee_flexion_adjudication"]
    assert adj["status"] == "adjudicated"
    assert adj["applied"] is True
    assert adj["action"] == "replaced_under_specified_singular_knee_bend_relation"


def test_existing_generic_bilateral_knee_bend_is_preserved_verbatim():
    sheet = _sheet([
        {"text": "knees bent", "composer_text": "knees bent"},
    ])
    out = mod._apply_bilateral_knee_flexion_authority(
        sheet,
        projected=_projected(left=70.0, right=55.0),
        dwpose_points=_points(),
    )
    body = out["facts"]["body"]
    assert body["configuration"] == [{"text": "knees bent", "composer_text": "knees bent"}]
    assert body["bilateral_knee_flexion_adjudication"]["status"] == "confirmed_existing"
    assert body["bilateral_knee_flexion_adjudication"]["applied"] is False


def test_richer_bilateral_knee_language_is_not_simplified():
    sheet = _sheet([
        {"text": "both knees bent and raised", "composer_text": "both knees bent and raised"},
    ])
    out = mod._apply_bilateral_knee_flexion_authority(
        sheet,
        projected=_projected(left=71.0, right=72.0, authority=0.99),
        dwpose_points=_points(),
    )
    body = out["facts"]["body"]
    assert body["configuration"][0]["composer_text"] == "both knees bent and raised"
    assert body["bilateral_knee_flexion_adjudication"]["status"] == "confirmed_existing"


def test_unilateral_sam3d_flexion_does_not_mutate_configuration():
    sheet = _sheet([
        {"text": "one knee raised", "composer_text": "one knee raised"},
    ])
    out = mod._apply_bilateral_knee_flexion_authority(
        sheet,
        projected=_projected(left=57.7, right=156.7, authority=0.95),
        dwpose_points=_points(),
    )
    body = out["facts"]["body"]
    assert body["configuration"] == [{"text": "one knee raised", "composer_text": "one knee raised"}]
    assert body["bilateral_knee_flexion_adjudication"]["status"] == "not_applicable"
    assert body["bilateral_knee_flexion_adjudication"]["composer_authoritative"] is False


def test_missing_one_dwpose_ankle_blocks_authority():
    points = _points()
    points["right_ankle"] = None
    sheet = _sheet([
        {"text": "knees bent and aligned with feet", "composer_text": "knees bent and aligned with feet"},
    ])
    out = mod._apply_bilateral_knee_flexion_authority(
        sheet,
        projected=_projected(left=75.0, right=70.0, authority=0.92),
        dwpose_points=points,
    )
    body = out["facts"]["body"]
    assert body["configuration"][0]["composer_text"] == "knees bent and aligned with feet"
    assert body["bilateral_knee_flexion_adjudication"]["status"] == "not_applicable"
    assert "both_full_dwpose_chains_not_observed" in body["bilateral_knee_flexion_adjudication"]["reason"]
