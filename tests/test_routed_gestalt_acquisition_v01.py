from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "qwen_caption_validate" / "routed_gestalt_acquisition_v01.py"
spec = importlib.util.spec_from_file_location("routed_gestalt_acquisition_v01", MODULE_PATH)
gestalt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gestalt)


def test_extracts_plain_json_object():
    parsed = gestalt._extract_json_object('{"scene":["park"],"gestalt":"outdoor portrait"}')
    assert parsed["scene"] == ["park"]


def test_extracts_json_from_markdown_fence():
    parsed = gestalt._extract_json_object('```json\n{"appearance":["dark jacket"]}\n```')
    assert parsed["appearance"] == ["dark jacket"]


def test_normalizer_whitelists_schema_and_reports_extra_keys():
    normalized, unexpected = gestalt._normalize_payload({
        "appearance": [" dark jacket ", "dark jacket", 123],
        "scene": ["outdoors"],
        "pose": "standing",
        "gestalt": "  casual outdoor image  ",
    })
    assert normalized["appearance"] == ["dark jacket"]
    assert normalized["scene"] == ["outdoors"]
    assert normalized["gestalt"] == "casual outdoor image"
    assert "pose" not in normalized
    assert unexpected == ["pose"]


def test_list_fields_are_capped():
    normalized, _ = gestalt._normalize_payload({
        "objects": ["one", "two", "three", "four", "five"]
    })
    assert normalized["objects"] == ["one", "two", "three", "four"]


def test_body_reference_is_metadata_only(tmp_path):
    path = tmp_path / "sample.routed_fragments.json"
    path.write_text(
        '{"status":"ok","policy_mode":"configuration","model_call":true,"raw_response":"secret body prose"}\n',
        encoding="utf-8",
    )
    ref = gestalt._body_reference(tmp_path, "sample")
    assert ref == {
        "path": str(path),
        "status": "ok",
        "policy_mode": "configuration",
        "model_call": True,
    }
    assert "secret body prose" not in str(ref)
