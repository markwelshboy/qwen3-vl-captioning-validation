from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "qwen_caption_validate" / "fact_sheet_normalizer_v01.py"
spec = importlib.util.spec_from_file_location("fact_sheet_normalizer_v01", MODULE_PATH)
facts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(facts)


def _policy(mode: str) -> dict:
    return {"image_key": "sample", "pose_relevance": "low", "policy": {"mode": mode}}


def _body(mode: str, pose: str | None = None, relationships: list[str] | None = None, status: str = "ok") -> dict:
    return {
        "image_key": "sample",
        "policy_mode": mode,
        "status": status,
        "extraction": {
            "pose_candidate": {"text": pose} if pose else None,
            "body_relationships": [{"text": text} for text in (relationships or [])],
        },
        "parse": {"route_violations": []},
    }


def _gestalt(mode: str, *, gestalt: str, expression_action: list[str] | None = None, appearance: list[str] | None = None) -> dict:
    return {
        "image_key": "sample",
        "policy_mode": mode,
        "status": "ok",
        "acquisition": {
            "appearance": appearance or [],
            "expression_action": expression_action or [],
            "objects": [],
            "scene": [],
            "secondary_people": [],
            "gestalt": gestalt,
            "uncertainties": [],
        },
    }


def _build(policy: dict, body: dict, gestalt: dict) -> dict:
    p = Path("policy.json")
    b = Path("body.json")
    g = Path("gestalt.json")
    return facts._build_fact_sheet(policy, body, gestalt, policy_path=p, body_path=b, gestalt_path=g)


def test_framing_only_gestalt_cannot_create_pose_fact():
    sheet = _build(
        _policy("framing_only"),
        _body("framing_only", status="skipped_by_policy"),
        _gestalt("framing_only", gestalt="close-up of a person standing outdoors", expression_action=["looking toward camera"]),
    )
    assert sheet["facts"]["body"]["pose_candidate"] is None
    assert sheet["facts"]["body"]["configuration"] == []
    assert sheet["context_only"]["gestalt"]["text"] == "close-up of a person standing outdoors"
    assert sheet["context_only"]["gestalt"]["authority"] == "context_only"


def test_configuration_does_not_promote_lying_from_gestalt():
    sheet = _build(
        _policy("configuration"),
        _body("configuration", relationships=["torso oriented horizontally relative to frame"]),
        _gestalt("configuration", gestalt="a person lying on a patterned blanket", expression_action=["lying down"]),
    )
    assert sheet["facts"]["body"]["pose_candidate"] is None
    assert [x["text"] for x in sheet["facts"]["body"]["configuration"]] == ["torso oriented horizontally relative to frame"]
    assert sheet["context_only"]["expression_action"][0]["promotion_status"] == "held_for_domain_normalization"


def test_pose_route_preserves_pose_as_hypothesis_not_truth():
    sheet = _build(
        _policy("pose_guided"),
        _body("pose_guided", pose="seated", relationships=["knees bent"]),
        _gestalt("pose_guided", gestalt="a person sitting between trees"),
    )
    pose = sheet["facts"]["body"]["pose_candidate"]
    assert pose["text"] == "seated"
    assert pose["authority"] == "qwen_pose_hypothesis"
    assert pose["promotion_status"] == "candidate"


def test_visual_laterality_is_held_for_dwpose():
    sheet = _build(
        _policy("pose_allowed"),
        _body("pose_allowed", pose="standing"),
        _gestalt("pose_allowed", gestalt="portrait", appearance=["pink smartwatch on left wrist", "floral top"]),
    )
    appearance = sheet["facts"]["visual"]["appearance"]
    assert appearance[0]["promotion_status"] == "held_for_laterality_normalization"
    assert appearance[1]["promotion_status"] == "accepted_candidate"


def test_mismatched_body_mode_is_a_review_violation():
    sheet = _build(
        _policy("configuration"),
        _body("pose_allowed", pose="standing"),
        _gestalt("configuration", gestalt="portrait"),
    )
    assert sheet["status"] == "needs_review"
    assert "body_policy_mode_mismatch" in sheet["audit"]["violations"]
    assert "broad_pose_candidate_present_in_non_pose_route" in sheet["audit"]["violations"]


def test_fact_sheet_is_never_caption_ready_in_phase_4a():
    sheet = _build(
        _policy("pose_allowed"),
        _body("pose_allowed", pose="standing"),
        _gestalt("pose_allowed", gestalt="portrait"),
    )
    assert sheet["audit"]["caption_ready"] is False
    assert sheet["reserved_domains"]["gaze"]["status"] == "pending_phase_4b"
