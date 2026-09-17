from pathlib import Path

from qwen_caption_validate import fact_sheet_text_composer_v10 as mod


def _projection(*, depth=True):
    body = {
        "broad_pose": (
            mod.QUALIFIED_CROUCHED_STANCE if depth else mod.BASE_CROUCHED_STANCE
        ),
        "canonical_broad_pose": "crouching",
        "configuration": ["both knees bent"],
    }
    if depth:
        body["crouched_stance_depth"] = {
            "classification": "moderate_lowering",
            "semantic_relation": "hips_slightly_lowered",
            "composer_text": "hips slightly lowered",
        }
    return {
        "authoritative_facts": {"body": body},
        "subject": {},
        "omitted_review_conflict_domains": [],
    }


def test_phase59_defaults():
    assert mod.DEFAULT_INPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "text-composer-v0.10"
    assert mod.SCHEMA_VERSION == "fact-sheet-text-composer-0.10"


def test_surface_lexicalization_distinguishes_base_and_moderate_depth():
    assert mod._surface_crouched_pose(depth_authorized=False) == "holds a crouched stance"
    assert (
        mod._surface_crouched_pose(depth_authorized=True)
        == "holds a crouched stance with her hips slightly lowered"
    )


def test_source_depth_fact_requires_specialist_authority():
    sheet = {
        "facts": {
            "body": {
                "crouched_stance_depth": {
                    "classification": "moderate_lowering",
                    "semantic_relation": "hips_slightly_lowered",
                    "promotion_status": "accepted_specialist_crouched_stance_depth_candidate",
                },
                "crouched_stance_depth_adjudication": {
                    "composer_authoritative": True,
                },
            }
        }
    }
    assert mod._source_depth_fact(sheet) == {
        "classification": "moderate_lowering",
        "semantic_relation": "hips_slightly_lowered",
        "composer_text": "hips slightly lowered",
    }

    sheet["facts"]["body"]["crouched_stance_depth_adjudication"]["composer_authoritative"] = False
    assert mod._source_depth_fact(sheet) is None


def test_qualified_crouched_stance_caption_passes_new_depth_audit():
    caption = (
        "sH1VX holds a crouched stance with her hips slightly lowered in full-body framing, "
        "her upper body bent forward from the hips with both knees bent."
    )
    audit = mod._caption_audit(caption, _projection(depth=True))
    assert "crouching_not_lexicalized_as_crouched_stance" not in audit["violations"]
    assert "authorized_hips_slightly_lowered_modifier_missing" not in audit["violations"]
    assert "unsupported_deep_or_low_crouch_language" not in audit["violations"]
    assert "hips_lowered_language_without_depth_authority" not in audit["violations"]


def test_bare_crouches_is_rejected_for_canonical_crouching():
    audit = mod._caption_audit(
        "sH1VX crouches in full-body framing with both knees bent.",
        _projection(depth=False),
    )
    assert "crouching_not_lexicalized_as_crouched_stance" in audit["violations"]


def test_hips_lowered_is_rejected_without_typed_depth_authority():
    audit = mod._caption_audit(
        "sH1VX holds a crouched stance with her hips slightly lowered and both knees bent.",
        _projection(depth=False),
    )
    assert "hips_lowered_language_without_depth_authority" in audit["violations"]


def test_deep_or_low_crouch_language_is_never_authorized_by_moderate_fact():
    audit = mod._caption_audit(
        "sH1VX holds a deep crouched stance with her hips slightly lowered and both knees bent.",
        _projection(depth=True),
    )
    assert "unsupported_deep_or_low_crouch_language" in audit["violations"]


def test_retry_prompt_preserves_conservative_lexical_contract():
    prompt = mod._retry_prompt(
        "BASE",
        "sH1VX crouches deeply.",
        [
            "crouching_not_lexicalized_as_crouched_stance",
            "authorized_hips_slightly_lowered_modifier_missing",
            "unsupported_deep_or_low_crouch_language",
        ],
    )
    assert "holds a crouched stance" in prompt
    assert "hips slightly lowered" in prompt
    assert "Remove 'deep' and 'low'" in prompt
