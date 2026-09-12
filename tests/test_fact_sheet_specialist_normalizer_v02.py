from __future__ import annotations

from qwen_caption_validate import fact_sheet_specialist_normalizer_v02 as mod


def _sheet(
    *,
    mode: str = "configuration",
    torso_visibility: str = "strong",
    torso_available: bool = True,
    config_texts: list[str] | None = None,
    horizontal_authority: str = "corroborated",
    horizontal_value: str | None = "frame_left",
    horizontal_degrees: float | None = 42.0,
    vertical_authority: str = "corroborated",
    vertical_value: str | None = "down",
    vertical_degrees: float | None = -20.0,
    yaw_strength: str | None = "strong_turn",
) -> dict:
    configuration = [
        {
            "text": text,
            "composer_text": text,
            "promotion_status": "accepted_route_scoped_candidate",
        }
        for text in (config_texts or [])
    ]
    return {
        "schema_version": "caption-fact-sheet-0.2",
        "policy": {"mode": mode},
        "facts": {
            "framing": {"body_visibility": {"torso": torso_visibility}},
            "body": {
                "configuration": configuration,
                "torso_geometry": {
                    "available": torso_available,
                    "torso_camera_orientation": "frontal" if torso_available else None,
                    "torso_yaw_magnitude_deg": 4.0 if torso_available else None,
                },
            },
            "head_pose": {
                "available": True,
                "horizontal": {
                    "value": horizontal_value,
                    "candidate_value": horizontal_value,
                    "degrees": horizontal_degrees,
                    "authority": horizontal_authority,
                    "publishable": horizontal_value is not None,
                },
                "vertical": {
                    "value": vertical_value,
                    "candidate_value": vertical_value,
                    "degrees": vertical_degrees,
                    "authority": vertical_authority,
                    "publishable": vertical_value is not None,
                },
                "yaw_strength": yaw_strength,
            },
        },
        "reserved_domains": {"torso_geometry": {"owner": "sam3d", "status": "resolved"}},
        "audit": {"warnings": [], "violations": [], "invariants": {}, "phase": "4B"},
    }


def test_framing_only_keeps_torso_diagnostic_but_not_composer_eligible():
    out = mod._apply_phase4b1(_sheet(mode="framing_only", torso_visibility="strong"))
    torso = out["facts"]["body"]["torso_geometry"]
    assert torso["available"] is True
    assert torso["composer_eligible"] is False
    assert torso["semantic_scope"] == "diagnostic_only_by_route"
    assert out["reserved_domains"]["torso_geometry"]["status"] == "resolved_diagnostic_only"


def test_direction_only_head_yaw_keeps_class_but_withholds_magnitude_and_strength():
    out = mod._apply_phase4b1(_sheet(horizontal_authority="corroborated_direction", horizontal_degrees=64.2, yaw_strength="profile"))
    head = out["facts"]["head_pose"]
    axis = head["horizontal"]
    assert axis["value"] == "frame_left"
    assert axis["candidate_degrees"] == 64.2
    assert axis["degrees"] is None
    assert axis["magnitude_publishable"] is False
    assert head["yaw_strength"] is None
    assert head["yaw_strength_candidate"] == "profile"
    assert head["yaw_strength_publishable"] is False


def test_direction_only_pitch_keeps_direction_but_withholds_angle():
    out = mod._apply_phase4b1(_sheet(vertical_authority="corroborated_direction", vertical_degrees=-54.1))
    axis = out["facts"]["head_pose"]["vertical"]
    assert axis["value"] == "down"
    assert axis["candidate_degrees"] == -54.1
    assert axis["degrees"] is None
    assert axis["magnitude_publishable"] is False


def test_strong_torso_observation_allows_specialist_to_supersede_qwen_camera_yaw():
    out = mod._apply_phase4b1(_sheet(config_texts=["torso angled slightly toward camera"], torso_visibility="strong"))
    item = out["facts"]["body"]["configuration"][0]
    torso = out["facts"]["body"]["torso_geometry"]
    assert torso["composer_eligible"] is True
    assert item["promotion_status"] == "superseded_by_torso_geometry"
    assert item["composer_text"] is None
    assert item["domain_handoff"] == "torso_camera_orientation"
    assert out["audit"]["review_conflicts"] == []


def test_partial_torso_observation_holds_both_sides_for_review():
    out = mod._apply_phase4b1(_sheet(config_texts=["torso angled slightly to the side"], torso_visibility="partial"))
    item = out["facts"]["body"]["configuration"][0]
    torso = out["facts"]["body"]["torso_geometry"]
    assert torso["composer_eligible"] is False
    assert torso["semantic_scope"] == "diagnostic_only_insufficient_torso_observation"
    assert item["promotion_status"] == "held_for_torso_orientation_review"
    assert item["composer_text"] is None
    assert len(out["audit"]["review_conflicts"]) == 1
    assert out["audit"]["review_conflicts"][0]["resolution"] == "manual_review_or_omit"


def test_in_plane_torso_posture_is_not_misclassified_as_camera_yaw():
    texts = ["torso bent forward", "torso oriented horizontally relative to frame"]
    out = mod._apply_phase4b1(_sheet(config_texts=texts, torso_visibility="strong"))
    items = out["facts"]["body"]["configuration"]
    assert [item["composer_text"] for item in items] == texts
    assert all("domain_handoff" not in item for item in items)


def test_phase_and_schema_are_upgraded():
    out = mod._apply_phase4b1(_sheet())
    assert out["schema_version"] == "caption-fact-sheet-0.2.1"
    assert out["audit"]["phase"] == "4B.1"
    assert out["audit"]["invariants"]["crop_governor_caps_torso_composer_bandwidth"] is True
