from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import caption_policy_identity_v05 as mod


def test_phase4b10_identity_policy_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.10"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.EXPECTED_INPUT_SCHEMA == "caption-fact-sheet-0.2.10"


def test_identity_policy_preserves_authoritative_pose_support_gate_and_provenance():
    source = {
        "schema_version": "caption-fact-sheet-0.2.10",
        "status": "ok",
        "image_key": "imageblind-01_00066",
        "facts": {
            "body": {
                "pose_candidate": {
                    "text": "crouching",
                    "composer_text": "crouching",
                    "normalized_text": "crouching",
                    "promotion_status": "accepted_specialist_adjudicated_candidate",
                    "specialist_owner": "sam3d_v16_broad_pose_adjudicator",
                    "broad_pose_adjudication_binding": {
                        "source_pose_candidate": {
                            "text": "standing with torso bent forward",
                            "composer_text": "standing with torso bent forward",
                        }
                    },
                },
                "broad_pose_adjudication": {
                    "status": "adjudicated",
                    "composer_authoritative": True,
                    "canonical_pose_text": "crouching",
                },
                "support_contact_adjudication": {
                    "status": "withheld_unverified_ground_contact",
                    "composer_authoritative": True,
                    "applied": True,
                },
                "configuration": [
                    {"text": "torso bent forward", "composer_text": "upper body bent forward from the hips"}
                ],
            },
            "visual": {"appearance": []},
        },
        "audit": {"warnings": [], "invariants": {}},
    }

    out = mod.base.apply_character_identity_policy(source)
    body = out["facts"]["body"]
    pose = body["pose_candidate"]
    assert out["schema_version"] == "caption-fact-sheet-0.3"
    assert pose["text"] == "crouching"
    assert pose["composer_text"] == "crouching"
    assert pose["promotion_status"] == "accepted_specialist_adjudicated_candidate"
    assert pose["broad_pose_adjudication_binding"]["source_pose_candidate"]["text"] == "standing with torso bent forward"
    assert body["broad_pose_adjudication"]["canonical_pose_text"] == "crouching"
    assert body["support_contact_adjudication"]["status"] == "withheld_unverified_ground_contact"
