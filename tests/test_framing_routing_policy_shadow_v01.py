from __future__ import annotations

from qwen_caption_validate import framing_routing_policy_shadow_v01 as mod


def test_overlay_changes_only_shadow_route_authority_fields() -> None:
    policy = {
        "schema_version": "caption-perception-policy-0.1",
        "image_key": "imageblind-01_00061",
        "image": "/tmp/image.png",
        "pose_relevance": "negligible",
        "policy": {"mode": "framing_only"},
        "visibility": {
            "broad_pose_supported": False,
            "extent_hint": "close_or_medium_close",
            "observed_landmarks": ["left_shoulder", "right_shoulder", "left_wrist", "right_elbow"],
        },
        "geometry": {
            "configuration_score": 1,
            "configuration_cues": ["visible_arm_relationship"],
        },
        "sources": {"dwpose": "/tmp/a.dwpose.json", "sam3d_arrays": None},
        "reasons": ["legacy reason"],
    }
    shadow = {
        "schema_version": "framing-semantics-shadow-0.2",
        "pose_gate_shadow": {
            "broad_pose_supported": False,
            "local_configuration_supported": True,
            "proposed_mode": "configuration",
            "local_configuration_gate": {
                "supported": True,
                "direct_qualifying_cues": ["visible_arm_relationship"],
            },
        },
    }

    out = mod.apply_overlay(policy, shadow)
    assert out["schema_version"] == mod.SCHEMA_VERSION
    assert out["policy"]["mode"] == "configuration"
    assert out["pose_relevance"] == "low"
    assert out["visibility"]["broad_pose_supported"] is False
    assert out["visibility"]["extent_hint"] == "close_or_medium_close"
    assert out["image"] == policy["image"]
    assert out["sources"] == policy["sources"]
    assert out["routing_shadow"]["original"]["mode"] == "framing_only"
    assert out["routing_shadow"]["proposed"]["local_configuration_supported"] is True


def test_overlay_rejects_unknown_mode() -> None:
    policy = {"policy": {"mode": "framing_only"}, "visibility": {}}
    shadow = {"pose_gate_shadow": {"proposed_mode": "unknown"}}
    try:
        mod.apply_overlay(policy, shadow)
    except ValueError as exc:
        assert "invalid proposed shadow mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")
