from __future__ import annotations

from qwen_caption_validate import bilateral_knee_flexion_shadow_v01 as mod


def _sheet(*configuration: str, route: str = "pose_guided"):
    return {
        "policy": {"mode": route},
        "facts": {
            "body": {
                "configuration": [
                    {"text": text, "composer_text": text}
                    for text in configuration
                ]
            }
        },
    }


def _projected(left: float, right: float, authority: float, pose: str):
    return {
        "pose": pose,
        "best_candidate_pose": pose,
        "geometry": {
            "left_knee_angle_deg": left,
            "right_knee_angle_deg": right,
        },
        "region_support": {"knees": authority},
        "assertion_authority": {"selected_path": "test_path"},
    }


def _points(*, left_full: bool = True, right_full: bool = True):
    points = {}
    for side, x, full in (("left", 0.0, left_full), ("right", 10.0, right_full)):
        points[f"{side}_hip"] = (x, 0.0)
        points[f"{side}_knee"] = (x + 1.0, 5.0)
        points[f"{side}_ankle"] = (x, 10.0) if full else None
    return points


def test_00066_like_bilateral_deep_flexion_is_shadow_candidate():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00066",
        sheet=_sheet("one knee slightly bent"),
        projected=_projected(123.496, 111.157, 0.9502, "crouching"),
        dwpose_points=_points(),
    )
    assert record["status"] == "candidate_would_change"
    assert record["would_change"] is True
    assert record["proposed_relation"] == "both knees bent"
    assert record["gates"] == {
        "both_full_dwpose_chains_observed": True,
        "both_sam3d_knee_angles_available": True,
        "both_sam3d_knees_deep": True,
        "knee_region_authority_sufficient": True,
    }


def test_00014_like_standing_does_not_trigger():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00014",
        sheet=_sheet(route="pose_allowed"),
        projected=_projected(146.0, 162.0, 0.931, "standing"),
        dwpose_points=_points(),
    )
    assert record["status"] == "not_candidate"
    assert record["would_change"] is False
    assert "one_or_both_sam3d_knees_above_deep_flexion_boundary" in record["reason"]


def test_00064_like_unilateral_flexion_does_not_trigger_even_if_one_knee_is_extreme():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00064",
        sheet=_sheet("one knee raised"),
        projected=_projected(57.7, 156.7, 0.411, "standing"),
        dwpose_points=_points(),
    )
    assert record["status"] == "not_candidate"
    assert record["gates"]["both_sam3d_knees_deep"] is False
    assert record["gates"]["knee_region_authority_sufficient"] is False


def test_00087_like_missing_dwpose_chain_blocks_publication_without_calling_it_counterevidence():
    record = mod.evaluate_shadow(
        image_key="imageblind-01_00087",
        sheet=_sheet(),
        projected=_projected(74.8, 70.2, 0.915, "squatting"),
        dwpose_points=_points(right_full=False),
    )
    assert record["status"] == "not_candidate"
    assert record["gates"]["both_sam3d_knees_deep"] is True
    assert record["gates"]["knee_region_authority_sufficient"] is True
    assert record["gates"]["both_full_dwpose_chains_observed"] is False
    assert "both_full_dwpose_chains_not_observed" in record["reason"]


def test_existing_bilateral_knee_relation_is_confirmed_without_change():
    record = mod.evaluate_shadow(
        image_key="control",
        sheet=_sheet("knees bent"),
        projected=_projected(100.0, 120.0, 0.90, "crouching"),
        dwpose_points=_points(),
    )
    assert record["status"] == "candidate_already_present"
    assert record["would_change"] is False
    assert record["proposed_relation"] == "both knees bent"


def test_non_pose_bearing_route_abstains():
    record = mod.evaluate_shadow(
        image_key="crop",
        sheet=_sheet(route="configuration"),
        projected=_projected(90.0, 90.0, 0.99, "crouching"),
        dwpose_points=_points(),
    )
    assert record["status"] == "route_abstain"
    assert record["proposed_relation"] is None
