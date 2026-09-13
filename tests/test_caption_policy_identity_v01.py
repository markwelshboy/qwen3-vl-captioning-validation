from __future__ import annotations

from qwen_caption_validate import caption_policy_identity_v01 as mod


def _sheet(appearance_texts: list[str]) -> dict:
    return {
        "schema_version": "caption-fact-sheet-0.2.2",
        "status": "ok",
        "image_key": "test",
        "facts": {
            "visual": {
                "appearance": [
                    {
                        "text": text,
                        "composer_text": text,
                        "promotion_status": "accepted_candidate",
                    }
                    for text in appearance_texts
                ]
            }
        },
        "audit": {"warnings": [], "invariants": {}},
    }


def _appearance(out: dict) -> list[dict]:
    return out["facts"]["visual"]["appearance"]


def test_blonde_shoulder_length_hair_is_protected():
    out = mod.apply_character_identity_policy(_sheet(["blonde shoulder-length hair"]))
    item = _appearance(out)[0]
    assert item["composer_text"] is None
    assert item["promotion_status"] == "protected_intrinsic_identity"
    assert item["caption_policy"]["protected_trait"] == "hair_identity"


def test_mixed_hair_phrase_keeps_transient_state_only():
    out = mod.apply_character_identity_policy(_sheet(["blonde shoulder-length hair pulled back in a messy ponytail with loose strands"]))
    item = _appearance(out)[0]
    assert item["composer_text"] == "hair pulled back in a messy ponytail with loose strands"
    assert item["promotion_status"] == "accepted_transient_appearance"


def test_curly_brown_hair_without_transient_state_is_protected():
    out = mod.apply_character_identity_policy(_sheet(["curly brown hair"]))
    item = _appearance(out)[0]
    assert item["composer_text"] is None
    assert item["promotion_status"] == "protected_intrinsic_identity"


def test_transient_hair_state_is_preserved():
    out = mod.apply_character_identity_policy(_sheet(["hair tied back with loose strands"]))
    item = _appearance(out)[0]
    assert item["composer_text"] == "hair tied back with loose strands"


def test_clothing_and_accessories_are_unchanged():
    out = mod.apply_character_identity_policy(_sheet(["dark green sleeveless top", "black sunglasses", "silver necklace"]))
    assert [x["composer_text"] for x in _appearance(out)] == [
        "dark green sleeveless top",
        "black sunglasses",
        "silver necklace",
    ]


def test_eye_color_skin_tone_age_and_face_structure_are_protected():
    out = mod.apply_character_identity_policy(_sheet([
        "blue eyes",
        "fair skin",
        "young woman",
        "high cheekbones",
    ]))
    assert all(item["composer_text"] is None for item in _appearance(out))
    assert all(item["promotion_status"] == "protected_intrinsic_identity" for item in _appearance(out))


def test_policy_preserves_original_observation_for_audit():
    original = "blonde shoulder-length hair"
    out = mod.apply_character_identity_policy(_sheet([original]))
    item = _appearance(out)[0]
    assert item["text"] == original
    assert out["caption_policy"]["profile"] == mod.PROFILE
    assert out["schema_version"] == "caption-fact-sheet-0.3"
    assert out["audit"]["phase"] == "4C"
