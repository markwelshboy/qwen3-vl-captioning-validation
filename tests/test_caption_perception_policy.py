from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "qwen_caption_validate" / "caption_perception_policy.py"
spec = importlib.util.spec_from_file_location("caption_perception_policy", MODULE_PATH)
policy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(policy)

W = 1000
H = 1000


def pts(**values):
    result = {name: None for name in policy.BODY18}
    result.update(values)
    return result


def high_conf_full_body_sam3d():
    return {
        "available": True,
        "reported_confidence": 0.9999,
        "projected_selected_joint_count": 13,
        "projected_inside_frame_count": 13,
        "projected_outside_frame_count": 0,
        "projected_outside_frame_fraction": 0.0,
    }


def test_tight_crop_is_not_promoted_by_high_confidence_sam3d():
    # Phase-1 hard gate: visible head/shoulders only, while SAM3D claims a very
    # confident complete reconstruction. Hidden body must not become caption evidence.
    observed = pts(
        nose=(500, 220),
        neck=(500, 330),
        left_shoulder=(610, 370),
        right_shoulder=(390, 370),
    )
    result = policy.route_policy(observed, width=W, height=H, sam3d=high_conf_full_body_sam3d())
    assert result["visibility"]["broad_pose_supported"] is False
    assert result["policy"]["mode"] == "framing_only"
    assert result["pose_relevance"] == "negligible"
    assert result["sam3d"]["can_promote_observability"] is False


def test_upper_body_head_shoulder_relationship_routes_configuration():
    # Representative 00049-style feature fixture: no lower-body support, but a
    # visibly canted shoulder line and offset head/shoulder relationship.
    observed = pts(
        nose=(690, 235),
        neck=(610, 340),
        left_shoulder=(730, 420),
        right_shoulder=(390, 350),
        left_elbow=(810, 560),
    )
    result = policy.route_policy(observed, width=W, height=H, sam3d=high_conf_full_body_sam3d())
    assert result["visibility"]["broad_pose_supported"] is False
    assert result["geometry"]["configuration_score"] >= 2
    assert result["policy"]["mode"] == "configuration"
    assert result["pose_relevance"] == "low"


def test_ordinary_broad_pose_routes_pose_allowed():
    observed = pts(
        nose=(500, 120), neck=(500, 220),
        left_shoulder=(590, 250), right_shoulder=(410, 250),
        left_hip=(565, 500), right_hip=(435, 500),
        left_knee=(560, 710), right_knee=(440, 710),
        left_ankle=(555, 910), right_ankle=(445, 910),
    )
    result = policy.route_policy(observed, width=W, height=H, sam3d=high_conf_full_body_sam3d())
    assert result["visibility"]["broad_pose_supported"] is True
    assert result["geometry"]["pose_complexity_score"] < 2
    assert result["policy"]["mode"] == "pose_allowed"
    assert result["pose_relevance"] == "medium"


def test_nonroutine_visible_pose_with_sam3d_routes_pose_guided():
    # Representative tree-image-style fixture: observed lower body is present,
    # and the body is strongly canted/folded. SAM3D is allowed to guide wording
    # only after the observed crop has already cleared broad-pose eligibility.
    observed = pts(
        nose=(300, 200), neck=(360, 300),
        right_shoulder=(300, 330), left_shoulder=(500, 430),
        right_hip=(500, 500), left_hip=(620, 560),
        right_knee=(610, 650), right_ankle=(760, 650),
        left_knee=(690, 690), left_ankle=(720, 850),
    )
    result = policy.route_policy(observed, width=W, height=H, sam3d=high_conf_full_body_sam3d())
    assert result["visibility"]["broad_pose_supported"] is True
    assert result["geometry"]["pose_complexity_score"] >= 2
    assert result["policy"]["mode"] == "pose_guided"
    assert result["pose_relevance"] == "high"


def test_nonroutine_visible_pose_without_sam3d_does_not_invent_guidance():
    observed = pts(
        nose=(300, 200), neck=(360, 300),
        right_shoulder=(300, 330), left_shoulder=(500, 430),
        right_hip=(500, 500), left_hip=(620, 560),
        right_knee=(610, 650), right_ankle=(760, 650),
        left_knee=(690, 690), left_ankle=(720, 850),
    )
    result = policy.route_policy(observed, width=W, height=H, sam3d={"available": False})
    assert result["visibility"]["broad_pose_supported"] is True
    assert result["geometry"]["pose_complexity_score"] >= 2
    assert result["policy"]["mode"] == "pose_allowed"


def test_aircraft_face_style_tight_crop_routes_framing_only():
    observed = pts(
        nose=(515, 250),
        neck=(500, 410),
        left_shoulder=(720, 500),
        right_shoulder=(300, 500),
    )
    result = policy.route_policy(observed, width=W, height=H, sam3d={"available": True, "reported_confidence": 0.98})
    assert result["policy"]["mode"] == "framing_only"
    assert result["visibility"]["hips"] == "absent"
    assert result["visibility"]["knees"] == "absent"
    assert result["visibility"]["feet"] == "absent"
