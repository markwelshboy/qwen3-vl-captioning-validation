from __future__ import annotations

from pathlib import Path

from qwen_caption_validate import fact_sheet_specialist_normalizer_v03 as mod


def test_phase4b2_uses_separate_output_namespace():
    assert mod.SCHEMA_VERSION == "caption-fact-sheet-0.2.2"
    assert mod.DEFAULT_OUTPUT_SUBDIR == Path("semantic-v3") / "caption-fact-sheet-v0.2.2"


def test_phase4b2_marks_audit_and_preserves_invariants_without_torso():
    sheet = {
        "schema_version": "caption-fact-sheet-0.2.1",
        "facts": {"body": {"torso_geometry": {"available": False}}},
        "audit": {"warnings": [], "invariants": {}},
    }
    out = mod._apply_phase4b2(sheet)
    assert out["schema_version"] == "caption-fact-sheet-0.2.2"
    assert out["audit"]["phase"] == "4B.2"
    assert out["audit"]["invariants"]["root_orientation_and_upper_torso_orientation_are_separate"] is True
    assert out["audit"]["invariants"]["torso_caption_reference_is_physical_camera_center"] is True
