from __future__ import annotations

from qwen_caption_validate import crouch_topology_lexical_probe_v02 as mod


def test_v02_changes_only_opening_predicate():
    source = (
        "sH1VX crouches in full-body framing, her upper body bent forward from the hips "
        "with both knees bent and both hands holding a dark patterned fabric."
    )
    tail, variants = mod._build_variants(source)

    assert len(variants) == 5
    assert tail.startswith(" in full-body framing")
    assert {v["variant_id"] for v in variants} == {
        "baseline_crouches",
        "deep_crouched_stance",
        "deep_standing_crouch",
        "low_deep_crouched_stance",
        "deep_crouched_stance_hips_lowered",
    }
    assert all(v["caption"].endswith(tail) for v in variants)
    assert len({v["invariant_tail_sha256"] for v in variants}) == 1
    assert all(v["only_opening_predicate_changed"] is True for v in variants)


def test_v02_marks_hips_lowered_as_experimental_not_authoritative():
    source = "sH1VX crouches in full-body framing."
    _, variants = mod._build_variants(source)
    by_id = {v["variant_id"]: v for v in variants}

    hips = by_id["deep_crouched_stance_hips_lowered"]
    assert hips["experimental_global_topology_wording"] is True
    assert hips["hips_lowered_authoritative"] is False
    assert hips["promotion_candidate"] is False
    assert "hips lowered" in hips["caption"]


def test_v02_keeps_standing_crouch_as_comparator_only():
    source = "sH1VX crouches in full-body framing."
    _, variants = mod._build_variants(source)
    by_id = {v["variant_id"]: v for v in variants}

    assert by_id["deep_standing_crouch"]["promotion_candidate"] is False
    assert by_id["deep_crouched_stance"]["promotion_candidate"] is True
    assert by_id["low_deep_crouched_stance"]["promotion_candidate"] is True
