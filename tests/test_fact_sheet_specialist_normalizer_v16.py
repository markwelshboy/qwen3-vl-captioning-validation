from __future__ import annotations

from qwen_caption_validate import fact_sheet_specialist_normalizer_v16 as phase4b15


def _sheet(configuration, *, left, right):
    return {
        "schema_version": "caption-fact-sheet-0.2.14",
        "status": "ok",
        "image_key": "synthetic",
        "facts": {
            "body": {
                "configuration": configuration,
                "anatomical_laterality": {
                    "available": True,
                    "authority": "dwpose_anatomical_joint_labels",
                    "sides": {
                        "left": {"visible_joints": list(left), "visible_joint_count": len(left)},
                        "right": {"visible_joints": list(right), "visible_joint_count": len(right)},
                    },
                },
            }
        },
        "audit": {"violations": [], "warnings": [], "invariants": {}},
    }


def _item(text):
    return {
        "text": text,
        "composer_text": text,
        "normalized_text": text,
        "domain": "configuration",
        "authority": "route_scoped_candidate",
        "promotion_status": "accepted_route_scoped_candidate",
    }


def test_head_support_collapses_duplicates_and_binds_only_hand_side():
    sheet = _sheet(
        [
            _item("fist under the chin with forearm beneath it"),
            _item("forearm held across the torso"),
            _item("hand supporting the chin/head"),
        ],
        left=("shoulder", "wrist"),
        right=("shoulder", "elbow"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    body = out["facts"]["body"]
    config = body["configuration"]

    assert len(config) == 2
    canonical = config[0]
    assert canonical["composer_text"] == (
        "chin resting on the left fist, with the forearm beneath/supporting the pose"
    )
    assert canonical["head_support_binding"]["anatomical_side"] == "left"
    assert canonical["head_support_binding"]["forearm_side_publishable"] is False

    assert config[1]["composer_text"] == "forearm held across the torso"

    adjudication = body["head_support_adjudication"]
    assert adjudication["status"] == "adjudicated"
    assert adjudication["applied"] is True
    assert adjudication["composer_authoritative"] is True
    assert adjudication["qwen_laterality_trusted"] is False
    assert adjudication["forearm_side_published"] is False
    assert "broad_pose" not in body


def test_non_head_support_configuration_is_left_unchanged():
    configuration = [
        _item("hand holding a white cup near the mouth"),
        _item("forearm resting on a surface with the hand partially visible"),
        _item("torso angled slightly forward relative to the surface"),
        _item("elbow bent and positioned near the torso"),
        _item("arm extended upward with the forearm visible beneath the cup"),
    ]
    sheet = _sheet(
        configuration,
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    body = out["facts"]["body"]

    assert body["configuration"] == configuration
    assert body["head_support_adjudication"]["status"] == "not_applicable"
    assert body["head_support_adjudication"]["applied"] is False


def test_generic_forearm_held_language_does_not_create_head_support():
    sheet = _sheet(
        [_item("forearm held across the torso")],
        left=("shoulder", "wrist"),
        right=("shoulder", "elbow"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    body = out["facts"]["body"]

    assert body["configuration"][0]["composer_text"] == "forearm held across the torso"
    assert body["head_support_adjudication"]["status"] == "not_applicable"


def test_bilateral_wrists_keep_head_support_side_neutral():
    sheet = _sheet(
        [
            _item("fist under the chin with forearm beneath it"),
            _item("hand supporting the chin/head"),
        ],
        left=("shoulder", "elbow", "wrist"),
        right=("shoulder", "elbow", "wrist"),
    )

    out = phase4b15._apply_head_support_authority(sheet)
    canonical = out["facts"]["body"]["configuration"][0]
    adjudication = out["facts"]["body"]["head_support_adjudication"]

    assert canonical["composer_text"] == (
        "chin resting on a fist, with the forearm beneath/supporting the pose"
    )
    assert "left" not in canonical["composer_text"]
    assert "right" not in canonical["composer_text"]
    assert adjudication["laterality_binding"]["status"] == "unresolved"
