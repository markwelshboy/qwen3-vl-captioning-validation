from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v04 as mod


def _sheet(yaw: float = 14.602951275676432) -> dict:
    return {
        "schema_version": "caption-fact-sheet-0.2.2",
        "facts": {
            "gaze": {
                "available": True,
                "publishable": True,
                "authority": "high",
                "horizontal": "frame_left",
                "vertical": "center",
                "camera_relationship": "uncertain",
                "yaw_deg": yaw,
                "pitch_deg": -8.379989828539909,
                "forward_deviation_deg": 16.79110152969647,
                "observability": {"eligible": True, "authority": "high"},
            }
        },
        "audit": {"warnings": [], "invariants": {}},
    }


def test_phase4b3_uses_separate_output_namespace():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.3"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.3"


def test_phase4b3_preserves_raw_gaze_and_adds_caption_semantics():
    out = mod._apply_phase4b3(_sheet())
    gaze = out["facts"]["gaze"]
    assert gaze["horizontal"] == "frame_left"
    assert gaze["yaw_deg"] == 14.602951275676432
    assert gaze["observability"]["authority"] == "high"
    assert gaze["caption_semantics"]["horizontal"]["semantic_class"] == "near_center"
    assert gaze["caption_semantics"]["publishable"] is False
    assert out["schema_version"] == "caption-fact-sheet-0.2.3"
    assert out["audit"]["phase"] == "4B.3"
    assert out["audit"]["invariants"]["gaze_observability_is_separate_from_caption_salience"] is True


def test_phase4b3_clear_lateral_gaze_remains_caption_publishable():
    out = mod._apply_phase4b3(_sheet(31.0))
    semantics = out["facts"]["gaze"]["caption_semantics"]
    assert semantics["horizontal"]["composer_value"] == "frame_left"
    assert semantics["publishable"] is True
