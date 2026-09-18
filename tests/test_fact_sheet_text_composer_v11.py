from qwen_caption_validate import fact_sheet_text_composer_v11 as composer


def _projection(framing):
    return {
        "subject": {
            "trigger_token": "sH1VX",
            "grammar_profile": "feminine",
            "subject_pronoun": "she",
            "possessive_pronoun": "her",
        },
        "authoritative_facts": {
            "framing": framing,
            "body": {},
        },
        "omitted_review_conflict_domains": [],
    }


def test_projection_uses_phase4b16_composer_framing_not_legacy_extent():
    sheet = {
        "facts": {
            "framing": {
                "extent": None,
                "legacy": {
                    "extent_hint": "close_or_medium_close",
                    "caption_authoritative": False,
                },
                "anatomical_span": {
                    "upper_anchor": "head",
                    "lower_anchor": "shoulders",
                    "composer_text": "framed from the head through the shoulders",
                },
                "standard_shot_scale": {
                    "status": "candidate",
                    "label": "medium_close_up",
                    "composer_text": "medium close-up",
                },
                "composer_framing": {
                    "source": "standard_shot_scale",
                    "composer_text": "medium close-up",
                },
            },
            "body": {},
            "visual": {},
        },
        "context_only": {},
        "audit": {"review_conflicts": []},
    }

    projection, audit = composer._projection(
        sheet,
        trigger_token="sH1VX",
        subject_class="woman",
    )

    framing = projection["authoritative_facts"]["framing"]
    assert framing == {
        "composer_text": "medium close-up",
        "surface_source": "standard_shot_scale",
        "shot_scale_label": "medium_close_up",
    }
    assert audit["legacy_extent_projection_disabled"] is True


def test_medium_close_up_near_opening_passes_framing_audit():
    projection = _projection({
        "composer_text": "medium close-up",
        "surface_source": "standard_shot_scale",
        "shot_scale_label": "medium_close_up",
    })
    caption = (
        "sH1VX is shown in a medium close-up, with her chin resting on her fist. "
        "She wears a dark top in an indoor setting."
    )

    audit = composer._caption_audit(caption, projection)

    assert "authoritative_framing_missing" not in audit["violations"]
    assert "authoritative_framing_not_near_opening" not in audit["violations"]
    assert not any(v.startswith("unauthorized_shot_scale:") for v in audit["violations"])
    assert audit["authoritative_framing_word_position"] is not None
    assert audit["authoritative_framing_word_position"] <= 14


def test_anatomical_span_rejects_invented_shot_scale():
    projection = _projection({
        "composer_text": "framed from around the shoulders through the hips",
        "surface_source": "anatomical_span",
        "anatomical_span": {"upper": "shoulders", "lower": "hips"},
    })
    caption = (
        "sH1VX is shown in a medium shot, framed from around the shoulders through the hips. "
        "She holds a cup near her mouth."
    )

    audit = composer._caption_audit(caption, projection)

    assert "shot_scale_language_without_authority" in audit["violations"]


def test_missing_or_late_authoritative_framing_is_audited():
    projection = _projection({
        "composer_text": "medium close-up",
        "surface_source": "standard_shot_scale",
        "shot_scale_label": "medium_close_up",
    })

    missing = composer._caption_audit(
        "sH1VX wears a dark top in a softly lit room with a neutral expression.",
        projection,
    )
    assert "authoritative_framing_missing" in missing["violations"]

    late = composer._caption_audit(
        "sH1VX wears a dark top with a patterned neckline in a softly lit indoor room, "
        "with several background details visible around her before she is shown in a medium close-up.",
        projection,
    )
    assert "authoritative_framing_not_near_opening" in late["violations"]
