from pathlib import Path

from qwen_caption_validate import routed_gestalt_rebind_v02 as rebind


def test_rebind_preserves_acquisition_but_updates_route_metadata(tmp_path: Path):
    source_path = tmp_path / "source.gestalt.json"
    policy_path = tmp_path / "policy.json"
    body_path = tmp_path / "body.json"

    source = {
        "schema_version": "routed-gestalt-0.1",
        "status": "ok",
        "image_key": "imageblind-01_00085",
        "policy_mode": "pose_guided",
        "acquisition": {
            "appearance": ["black top"],
            "objects": ["white cup"],
            "scene": ["indoor room"],
            "expression_action": [],
            "secondary_people": [],
            "uncertainties": [],
            "gestalt": "drinking from a cup",
        },
        "raw_response": "{\"objects\":[\"white cup\"]}",
        "prompt_sha256": "abc",
    }
    policy = {
        "image_key": "imageblind-01_00085",
        "policy": {"mode": "configuration"},
    }
    body_path.write_text(
        '{"status":"ok","policy_mode":"configuration","model_call":true}',
        encoding="utf-8",
    )

    out = rebind._rebind_record(
        source,
        policy,
        source_path=source_path,
        policy_path=policy_path,
        body_path=body_path,
    )

    assert out["schema_version"] == "routed-gestalt-0.2"
    assert out["policy_mode"] == "configuration"
    assert out["acquisition"] == source["acquisition"]
    assert out["raw_response"] == source["raw_response"]
    assert out["route_rebind"]["source_policy_mode"] == "pose_guided"
    assert out["route_rebind"]["target_policy_mode"] == "configuration"
    assert out["route_rebind"]["acquisition_reused_without_model_call"] is True
    assert out["body_acquisition"]["policy_mode"] == "configuration"
