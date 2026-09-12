from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "qwen_caption_validate" / "fact_sheet_text_composer_v01.py"
spec = importlib.util.spec_from_file_location("fact_sheet_text_composer_v01", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _sheet(
    *,
    mode: str = "configuration",
    pose: str | None = None,
    config: list[dict] | None = None,
    torso: dict | None = None,
    head: dict | None = None,
    gaze: dict | None = None,
    appearance: list[str] | None = None,
    expression_action: list[str] | None = None,
    gestalt: str = "a person indoors",
    conflicts: list[dict] | None = None,
    framing_extent: str = "waist_or_upper_body",
) -> dict:
    visual_appearance = [
        {"text": text, "composer_text": text, "promotion_status": "accepted_candidate"}
        for text in (appearance or [])
    ]
    body = {
        "pose_candidate": (
            {"text": pose, "promotion_status": "candidate"} if pose else None
        ),
        "configuration": config or [],
        "torso_geometry": torso or {"available": False, "composer_eligible": False},
    }
    return {
        "schema_version": "caption-fact-sheet-0.2.1",
        "image_key": "sample",
        "policy": {"mode": mode},
        "facts": {
            "framing": {"extent": framing_extent},
            "body": body,
            "head_pose": head or {"available": False},
            "gaze": gaze or {"available": False, "publishable": False},
            "visual": {
                "appearance": visual_appearance,
                "objects": [],
                "scene": [],
                "secondary_people": [],
            },
        },
        "context_only": {
            "expression_action": [
                {"text": text, "authority": "context_only"}
                for text in (expression_action or [])
            ],
            "gestalt": {"text": gestalt, "authority": "context_only"},
            "uncertainties": [],
        },
        "audit": {
            "phase": "4B.1",
            "review_conflicts": conflicts or [],
        },
    }


def test_configuration_cannot_recover_lying_from_context():
    sheet = _sheet(
        mode="configuration",
        expression_action=["lying down", "neutral facial expression", "looking toward camera"],
        gestalt="a person lying on a patterned blanket, looking toward camera",
    )
    projection, audit = mod._projection(sheet)
    facts = projection["authoritative_facts"]
    assert "body" not in facts
    assert facts["safe_expression_action"] == ["neutral facial expression"]
    assert "lying" not in projection["holistic_context_non_authoritative"].lower()
    assert "looking" not in projection["holistic_context_non_authoritative"].lower()
    assert audit["holistic_context_sanitized"] is True


def test_framing_only_diagnostic_torso_is_not_projected():
    sheet = _sheet(
        mode="framing_only",
        torso={
            "available": True,
            "composer_eligible": False,
            "torso_camera_orientation": "slightly_angled",
            "semantic_scope": "diagnostic_only_by_route",
        },
        framing_extent="close_or_medium_close",
        gestalt="close-up selfie of a person standing outdoors",
    )
    projection, audit = mod._projection(sheet)
    assert "body" not in projection["authoritative_facts"]
    assert projection["authoritative_facts"]["framing"]["extent"] == "close framing"
    assert audit["diagnostic_only_torso_removed"] is True
    assert "standing" not in projection["holistic_context_non_authoritative"].lower()


def test_composer_eligible_torso_and_pose_are_projected():
    sheet = _sheet(
        mode="pose_guided",
        pose="standing with one leg raised",
        config=[{"text": "one knee raised", "composer_text": "one knee raised"}],
        torso={
            "available": True,
            "composer_eligible": True,
            "torso_camera_orientation": "three_quarter",
        },
        framing_extent="full_length",
    )
    projection, _ = mod._projection(sheet)
    body = projection["authoritative_facts"]["body"]
    assert body["broad_pose"] == "standing with one leg raised"
    assert body["configuration"] == ["one knee raised"]
    assert body["torso_orientation"] == {"camera_orientation": "three_quarter"}


def test_direction_only_head_projection_contains_no_magnitude():
    sheet = _sheet(
        head={
            "available": True,
            "horizontal": {
                "publishable": True,
                "value": "frame_left",
                "authority": "corroborated_direction",
                "degrees": None,
                "candidate_degrees": 64.2,
            },
            "vertical": {
                "publishable": True,
                "value": "down",
                "authority": "corroborated_direction",
                "degrees": None,
                "candidate_degrees": -54.1,
            },
            "yaw_strength": None,
            "yaw_strength_candidate": "profile",
        },
    )
    projection, _ = mod._projection(sheet)
    assert projection["authoritative_facts"]["head"] == {
        "horizontal": "frame_left",
        "vertical": "down",
    }


def test_uncertain_gaze_relationship_is_omitted_but_direction_survives():
    sheet = _sheet(
        gaze={
            "available": True,
            "publishable": True,
            "horizontal": "frame_left",
            "vertical": "center",
            "camera_relationship": "uncertain",
        }
    )
    projection, _ = mod._projection(sheet)
    gaze = projection["authoritative_facts"]["gaze"]
    assert gaze == {"horizontal": "frame_left", "vertical": "center"}


def test_safe_expression_survives_while_body_and_gaze_context_are_filtered():
    sheet = _sheet(
        expression_action=[
            "soft smile",
            "looking downward",
            "hands clasped on lap",
            "holding phone for selfie",
        ]
    )
    projection, _ = mod._projection(sheet)
    assert projection["authoritative_facts"]["safe_expression_action"] == [
        "soft smile",
        "holding phone for selfie",
    ]


def test_review_conflict_domain_and_held_qwen_orientation_are_omitted():
    sheet = _sheet(
        config=[{
            "text": "torso angled slightly to the side",
            "composer_text": None,
            "promotion_status": "held_for_torso_orientation_review",
        }],
        torso={
            "available": True,
            "composer_eligible": False,
            "torso_camera_orientation": "frontal",
        },
        conflicts=[{"domain": "torso_camera_orientation", "resolution": "manual_review_or_omit"}],
    )
    projection, audit = mod._projection(sheet)
    assert "body" not in projection["authoritative_facts"]
    assert projection["omitted_review_conflict_domains"] == ["torso_camera_orientation"]
    assert audit["review_conflict_domains_omitted"] == ["torso_camera_orientation"]


def test_post_generation_audit_catches_reintroduced_pose_and_laterality():
    sheet = _sheet(mode="framing_only", framing_extent="close_or_medium_close")
    projection, _ = mod._projection(sheet)
    audit = mod._caption_audit(
        "A person is standing outdoors with the right hand raised.",
        projection,
    )
    assert "unauthorized_broad_pose:standing" in audit["violations"]
    assert "unauthorized_anatomical_laterality" in audit["violations"]


def test_trigger_token_is_projected_without_becoming_a_visual_fact():
    sheet = _sheet(gestalt="a woman outdoors")
    projection, _ = mod._projection(sheet, trigger_token="sH1VX", subject_class="woman")
    assert projection["subject"] == {
        "trigger_token": "sH1VX",
        "subject_class": "woman",
        "reference_hint": "woman",
    }
