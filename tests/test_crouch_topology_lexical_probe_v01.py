from qwen_caption_validate import crouch_topology_lexical_probe_v01 as mod


def test_build_variants_changes_only_opening_predicate():
    caption = (
        "sH1VX crouches in full-body framing, her upper body bent forward from the hips "
        "with both knees bent."
    )
    invariant_tail, variants = mod._build_variants(caption)

    assert invariant_tail == (
        " in full-body framing, her upper body bent forward from the hips with both knees bent."
    )
    assert [v["variant_id"] for v in variants] == [
        "baseline_crouches",
        "deep_crouched_stance",
        "deep_standing_crouch",
    ]
    assert variants[0]["caption"] == caption
    assert variants[1]["caption"].startswith("sH1VX holds a deep crouched stance in full-body framing")
    assert variants[2]["caption"].startswith("sH1VX holds a deep standing crouch in full-body framing")
    assert len({v["invariant_tail_sha256"] for v in variants}) == 1
    assert all(v["only_opening_predicate_changed"] for v in variants)


def test_build_variants_requires_crouches_opening():
    try:
        mod._build_variants("sH1VX sits in full-body framing.")
    except ValueError as exc:
        assert "must begin '<subject> crouches'" in str(exc)
    else:
        raise AssertionError("expected ValueError")
