from qwen_caption_validate import fact_sheet_specialist_normalizer_v14 as mod


def _sheet():
    return {
        "image_key": "control",
        "policy": {"mode": "pose_guided"},
        "facts": {
            "body": {
                "configuration": [
                    {
                        "text": "one knee raised",
                        "composer_text": "one knee raised",
                        "normalized_text": "one knee raised",
                        "domain": "configuration",
                    },
                    {
                        "text": "both hands holding visible fabric",
                        "composer_text": "both hands holding visible fabric",
                        "normalized_text": "both hands holding visible fabric",
                        "domain": "configuration",
                    },
                ]
            }
        },
    }


def _projected():
    return {
        "pose": "standing",
        "geometry": {
            "asymmetric_lower_body": {
                "per_side": {
                    "left": {
                        "knee_flexion_deg": 58.0,
                        "thigh_axis_from_image_down_deg": 101.0,
                        "leg_extension_ratio": 0.49,
                    },
                    "right": {
                        "knee_flexion_deg": 157.0,
                        "thigh_axis_from_image_down_deg": 14.0,
                        "leg_extension_ratio": 0.98,
                    },
                }
            }
        },
    }


def _points():
    return {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (100.0, 0.0),
        "left_hip": (20.0, 50.0),
        "right_hip": (80.0, 50.0),
        "left_knee": (10.0, 45.0),
        "right_knee": (80.0, 200.0),
        "left_ankle": (20.0, 100.0),
        "right_ankle": (80.0, 350.0),
    }


def test_promotes_direct_left_raised_leg_without_support_claim():
    out = mod._apply_unilateral_raised_leg_authority(
        _sheet(), projected=_projected(), dwpose_points=_points()
    )
    body = out["facts"]["body"]
    texts = [x.get("composer_text") for x in body["configuration"]]

    assert "left knee raised high with thigh nearly horizontal" in texts
    assert "one knee raised" not in texts
    assert "both hands holding visible fabric" in texts
    assert not any("right foot" in (x or "") for x in texts)
    assert not any("support" in (x or "").lower() for x in texts)

    adj = body["unilateral_raised_leg_adjudication"]
    assert adj["status"] == "adjudicated"
    assert adj["candidate_side"] == "left"
    assert adj["support_leg_claim_created"] is False
    assert adj["planted_foot_claim_created"] is False


def test_non_candidate_does_not_mutate_configuration():
    projected = _projected()
    projected["geometry"]["asymmetric_lower_body"]["per_side"]["left"]["thigh_axis_from_image_down_deg"] = 40.0
    before = _sheet()
    out = mod._apply_unilateral_raised_leg_authority(
        before, projected=projected, dwpose_points=_points()
    )
    assert out["facts"]["body"]["configuration"] == before["facts"]["body"]["configuration"]
    assert out["facts"]["body"]["unilateral_raised_leg_adjudication"]["status"] == "not_applicable"
