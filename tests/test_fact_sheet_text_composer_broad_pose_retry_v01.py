from __future__ import annotations

from qwen_caption_validate import fact_sheet_text_composer_broad_pose_retry_v01 as mod


def test_only_unauthorized_broad_pose_violations_trigger_retry():
    record = {
        "caption_audit": {
            "violations": [
                "unauthorized_broad_pose:lying",
                "gaze_language_without_publishable_gaze",
            ]
        }
    }
    assert mod._broad_pose_violations(record) == ["unauthorized_broad_pose:lying"]
    assert mod._forbidden_groups(mod._broad_pose_violations(record)) == ["lying"]


def test_retry_prompt_preserves_trigger_binding_and_forbids_broad_pose():
    record = {
        "caption": "sH1VX lies on a blanket.",
        "caption_audit": {"violations": ["unauthorized_broad_pose:lying"]},
        "evidence_projection": {
            "subject": {
                "trigger_token": "sH1VX",
                "grammar_profile": "feminine",
                "subject_pronoun": "she",
                "possessive_pronoun": "her",
            },
            "authoritative_facts": {
                "framing": {"extent": "upper-body or waist-up framing"},
                "body": {"configuration": ["torso oriented horizontally relative to frame"]},
            },
        },
    }
    prompt = mod._retry_prompt(record)
    assert "Forbidden broad-pose group(s) for this retry: lying" in prompt
    assert "Preserve the exact trigger as the first grammatical subject" in prompt
    assert '"trigger_token": "sH1VX"' in prompt
    assert "sH1VX lies on a blanket." in prompt
