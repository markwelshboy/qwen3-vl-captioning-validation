from __future__ import annotations

from qwen_caption_validate import caption_perception_policy_v02 as policy


def _points(**present):
    names = policy.base.BODY18
    out = {name: None for name in names}
    for name, value in present.items():
        out[name] = value
    return out


def test_single_visible_arm_relationship_routes_to_configuration_without_broad_pose():
    pts = _points(
        nose=(50, 30),
        left_eye=(45, 28),
        right_eye=(55, 28),
        left_shoulder=(40, 50),
        right_shoulder=(60, 50),
        right_elbow=(72, 65),
    )
    out = policy.route_policy(pts, width=100, height=100, sam3d={"available": False})
    assert out["visibility"]["broad_pose_supported"] is False
    assert out["local_configuration_gate"]["supported"] is True
    assert out["policy"]["mode"] == "configuration"


def test_close_crop_without_local_relationship_remains_framing_only():
    pts = _points(
        nose=(50, 30),
        left_eye=(45, 28),
        right_eye=(55, 28),
        left_shoulder=(40, 50),
        right_shoulder=(60, 50),
    )
    out = policy.route_policy(pts, width=100, height=100, sam3d={"available": False})
    assert out["visibility"]["broad_pose_supported"] is False
    assert out["local_configuration_gate"]["supported"] is False
    assert out["policy"]["mode"] == "framing_only"


def test_one_knee_cannot_authorize_broad_pose_but_local_arm_can_route_configuration():
    pts = _points(
        neck=(50, 20),
        left_shoulder=(40, 30),
        right_shoulder=(60, 30),
        left_elbow=(35, 45),
        left_wrist=(30, 55),
        left_hip=(45, 60),
        right_hip=(55, 60),
        right_knee=(56, 82),
    )
    out = policy.route_policy(pts, width=100, height=100, sam3d={"available": True, "projected_selected_joint_count": 9})
    assert out["visibility"]["hips"] == "strong"
    assert out["visibility"]["knees"] == "partial"
    assert out["visibility"]["broad_pose_supported"] is False
    assert out["policy"]["mode"] == "configuration"
    assert out["sam3d"]["can_promote_observability"] is False


def test_bilateral_hips_and_knees_authorize_broad_pose():
    pts = _points(
        left_shoulder=(40, 20),
        right_shoulder=(60, 20),
        left_hip=(45, 50),
        right_hip=(55, 50),
        left_knee=(43, 75),
        right_knee=(57, 75),
    )
    out = policy.route_policy(pts, width=100, height=100, sam3d={"available": False})
    assert out["visibility"]["broad_pose_supported"] is True
    assert out["policy"]["mode"] == "pose_allowed"


def test_single_ankle_does_not_make_full_length_extent():
    pts = _points(
        nose=(50, 10),
        left_eye=(45, 10),
        right_eye=(55, 10),
        left_shoulder=(40, 25),
        right_shoulder=(60, 25),
        left_hip=(45, 50),
        right_hip=(55, 50),
        left_knee=(45, 70),
        right_knee=(55, 70),
        left_ankle=(45, 95),
    )
    vis = policy._visibility(pts, 100, 100)
    assert vis["feet"] == "partial"
    assert vis["extent_hint"] == "three_quarter_or_long"
    assert vis["broad_pose_supported"] is True
