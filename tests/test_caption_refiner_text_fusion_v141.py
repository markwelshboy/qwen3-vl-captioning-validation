from pathlib import Path

from qwen_caption_validate.caption_refiner_text_fusion_v141 import (
    interpret_structured_decision,
    validate_correction_text,
)


def test_no_correction_structured_decision_is_valid():
    result = interpret_structured_decision(
        {"decision": "NO_CORRECTION", "correction": None},
        '{"decision":"NO_CORRECTION","correction":null}',
    )
    assert result["text"] == "NO_CORRECTION"
    assert result["contract_ok"] is True
    assert result["contract_reason"] is None


def test_correction_structured_decision_is_valid():
    result = interpret_structured_decision(
        {
            "decision": "CORRECTION",
            "correction": "Her gaze is directed toward image-left and downward, clearly off-camera.",
        },
        "{}",
    )
    assert result["contract_ok"] is True
    assert result["text"].startswith("Her gaze")


def test_rejects_rationale_language_even_if_one_sentence():
    ok, reason = validate_correction_text(
        "The current caption states that her gaze is right, but the evidence confirms image-left."
    )
    assert ok is False
    assert reason == "meta_or_rationale_language"


def test_rejects_truncated_unterminated_fragment():
    ok, reason = validate_correction_text(
        "Her gaze is directed toward image-left and downward, with specialist data clearly"
    )
    assert ok is False
    assert reason == "correction_not_terminal_sentence"


def test_rejects_no_correction_with_nonempty_correction_field():
    result = interpret_structured_decision(
        {"decision": "NO_CORRECTION", "correction": "Her gaze is left."},
        "{}",
    )
    assert result["contract_ok"] is False
    assert result["contract_reason"] == "no_correction_with_nonempty_correction"


def test_structured_prompt_forbids_laterality_depth_inference():
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "caption_refiner_v141_text_fusion_structured.txt"
    ).read_text(encoding="utf-8")
    assert "CANNOT establish or refute forward/back depth" in prompt
    assert '"her left"' in prompt
    assert "NOT a frame direction" in prompt
