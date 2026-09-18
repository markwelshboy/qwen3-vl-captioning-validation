from qwen_caption_validate import selfie_evidence_shadow_v04 as v04


def _arm(side="right", grade="strong"):
    return {
        "grade": grade,
        "selected_arm": side,
        "composer_text": "an outstretched arm extends into the lower-frame-left foreground",
        "reason": "test",
    }


def test_matching_clear_nearer_shoulder_preserves_arm_grade():
    shoulder = {
        "status": "clear",
        "publishable_candidate": True,
        "anatomical_side_nearer": "right",
    }
    out = v04._apply_nearer_shoulder_consistency(_arm("right", "strong"), shoulder)

    assert out["grade"] == "strong"
    assert out["nearer_shoulder_consistency"]["status"] == "consistent"
    assert out["composer_text"] is not None


def test_opposite_clear_nearer_shoulder_vetoes_arm_as_selfie_evidence():
    shoulder = {
        "status": "clear",
        "publishable_candidate": True,
        "anatomical_side_nearer": "right",
    }
    out = v04._apply_nearer_shoulder_consistency(_arm("left", "strong"), shoulder)

    assert out["raw_grade_before_shoulder_crosscheck"] == "strong"
    assert out["grade"] == "none"
    assert out["composer_text"] is None
    assert out["nearer_shoulder_consistency"]["status"] == "conflict"


def test_ambiguous_shoulder_does_not_veto_arm():
    shoulder = {
        "status": "ambiguous",
        "publishable_candidate": False,
        "anatomical_side_nearer": None,
    }
    out = v04._apply_nearer_shoulder_consistency(_arm("left", "moderate"), shoulder)

    assert out["grade"] == "moderate"
    assert out["nearer_shoulder_consistency"]["status"] == "not_available"
