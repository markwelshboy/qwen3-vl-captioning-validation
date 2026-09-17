from qwen_caption_validate import framing_semantics_v01 as framing


def _policy(states):
    observed = {
        "head": {"state": states.get("head", "absent")},
        "shoulders": {"state": states.get("shoulders", "absent")},
        "hips": {"state": states.get("hips", "absent")},
        "knees": {"state": states.get("knees", "absent")},
        "ankles": {"state": states.get("ankles", "absent")},
    }
    return {
        "image_key": "synthetic",
        "image_size": [1000, 1000],
        "visibility": {
            "anatomical_tiers": observed,
            "extent_hint": "legacy",
            "broad_pose_supported": False,
        },
    }


def _uniface(face_h):
    return {
        "status": "ok",
        "face": {
            "bbox_xyxy": [100, 100, 400, 100 + int(face_h * 1000)],
            "score": 0.99,
            "selection_strategy": "synthetic",
        },
    }


def _person(person_h):
    return {
        "status": "ok",
        "person_geometry": {
            "status": "available",
            "visible_height_fraction": person_h,
            "visible_width_fraction": 0.8,
            "visible_area_fraction": 0.8 * person_h,
            "authority": "observed_person_detector",
            "source": "easy_dwpose_yolox_target_bound",
        },
    }


def test_head_shoulders_final_scale_bands():
    policy = _policy({"head": "strong", "shoulders": "strong"})

    # ratio .35 => medium
    out = framing.evaluate(policy, _uniface(0.35), _person(1.0))
    assert out["standard_shot_scale"]["label"] == "medium"
    assert out["composer_framing"]["composer_text"] == "medium shot"

    # ratio .37 => abstain
    out = framing.evaluate(policy, _uniface(0.37), _person(1.0))
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert out["composer_framing"]["source"] == "anatomical_span"

    # ratio .50 => MCU
    out = framing.evaluate(policy, _uniface(0.50), _person(1.0))
    assert out["standard_shot_scale"]["label"] == "medium_close_up"

    # ratio .67 => abstain
    out = framing.evaluate(policy, _uniface(0.67), _person(1.0))
    assert out["standard_shot_scale"]["status"] == "withheld"

    # ratio .70 => CU
    out = framing.evaluate(policy, _uniface(0.70), _person(1.0))
    assert out["standard_shot_scale"]["label"] == "close_up"


def test_shoulder_to_hip_crop_uses_literal_anatomical_span():
    policy = _policy({
        "head": "partial",
        "shoulders": "strong",
        "hips": "strong",
        "knees": "partial",
    })
    out = framing.evaluate(policy, None, _person(0.9))

    span = out["anatomical_span"]
    assert span["upper_anchor"] == "shoulders"
    assert span["lower_anchor"] == "hips"
    assert span["upper_partial"] == "head"
    assert span["lower_partial"] == "knees"
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert out["composer_framing"]["composer_text"] == "framed from around the shoulders through the hips"


def test_shoulder_to_knee_crop_never_becomes_full_body():
    policy = _policy({
        "head": "partial",
        "shoulders": "strong",
        "hips": "strong",
        "knees": "strong",
        "ankles": "partial",
    })
    out = framing.evaluate(policy, None, _person(0.98))

    assert out["anatomical_span"]["upper_anchor"] == "shoulders"
    assert out["anatomical_span"]["lower_anchor"] == "knees"
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert "full-body" not in (out["composer_framing"]["composer_text"] or "")


def test_head_to_ankles_withholds_full_body_without_foot_evidence():
    policy = _policy({
        "head": "strong",
        "shoulders": "strong",
        "hips": "strong",
        "knees": "strong",
        "ankles": "strong",
    })
    out = framing.evaluate(policy, _uniface(0.1), _person(1.0))

    assert out["anatomical_span"]["lower_anchor"] == "ankles"
    assert out["standard_shot_scale"]["status"] == "withheld"
    assert "feet_full_body_language_not_authorized" in out["standard_shot_scale"]["basis"][0]
