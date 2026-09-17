from __future__ import annotations

from qwen_caption_validate import fact_sheet_specialist_normalizer_v05 as production
from qwen_caption_validate import torso_yaw_sign_audit_v01 as mod


def _sheet(root_yaw: float, upper_yaw: float, *, root_dir, upper_dir, composer_eligible=True):
    return {
        "image_key": "control",
        "facts": {
            "body": {
                "torso_geometry": {
                    "available": True,
                    "composer_eligible": composer_eligible,
                    "body_root_orientation": {
                        "yaw_deg": root_yaw,
                        "yaw_magnitude_deg": abs(root_yaw),
                        "orientation_band": "oblique" if abs(root_yaw) > 25 else "frontal",
                        "approx_yaw_deg": round(abs(root_yaw) / 5) * 5,
                        "turn_direction": root_dir,
                        "turn_direction_publishable": bool(root_dir and composer_eligible),
                    },
                    "upper_torso_orientation": {
                        "yaw_deg": upper_yaw,
                        "yaw_magnitude_deg": abs(upper_yaw),
                        "orientation_band": "slightly_angled" if abs(upper_yaw) > 15 else "frontal",
                        "approx_yaw_deg": round(abs(upper_yaw) / 5) * 5,
                        "turn_direction": upper_dir,
                        "turn_direction_publishable": bool(upper_dir and composer_eligible),
                    },
                    "caption_orientation": {
                        "mode": "articulated",
                        "preferred_orientation_band": "slightly_angled",
                        "preferred_approx_yaw_deg": round(abs(upper_yaw) / 5) * 5,
                        "preferred_turn_direction": upper_dir,
                        "turn_direction_publishable": bool(upper_dir and composer_eligible),
                        "relative_twist_yaw_deg": upper_yaw - root_yaw,
                        "relative_twist_magnitude_deg": abs(upper_yaw - root_yaw),
                    },
                }
            }
        },
    }


def test_audit_uses_exact_production_direction_helper():
    assert production._turn_direction(34.9) == "frame_left"
    assert production._turn_direction(-34.9) == "frame_right"
    assert production._turn_direction(13.0) is None
    assert production._turn_direction(15.0) is None
    assert production._turn_direction(15.1) == "frame_left"


def test_00066_like_articulated_positive_yaws_are_frame_left():
    record = mod.audit_sheet(
        _sheet(34.9, 21.4, root_dir="frame_left", upper_dir="frame_left")
    )
    assert record["status"] == "consistent"
    by_name = {x["segment"]: x for x in record["segments"]}
    assert by_name["body_root_orientation"]["production_mapped_direction"] == "frame_left"
    assert by_name["upper_torso_orientation"]["production_mapped_direction"] == "frame_left"
    assert record["caption_orientation"]["preferred_turn_direction_from_production_helper"] == "frame_left"


def test_00039_like_near_frontal_yaw_withholds_signed_direction():
    record = mod.audit_sheet(_sheet(13.0, 13.0, root_dir=None, upper_dir=None))
    assert record["status"] == "consistent"
    assert all(x["inside_near_frontal_deadband"] for x in record["segments"])
    assert all(x["production_mapped_direction"] is None for x in record["segments"])
    assert all(x["expected_turn_direction_publishable"] is False for x in record["segments"])


def test_negative_yaw_maps_to_frame_right():
    record = mod.audit_sheet(
        _sheet(-35.0, -20.0, root_dir="frame_right", upper_dir="frame_right")
    )
    assert record["status"] == "consistent"
    assert {x["production_mapped_direction"] for x in record["segments"]} == {"frame_right"}
