from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_caption_validate import build_fact_sheet_current as mod


def test_default_stages_are_current_production_chain():
    assert [stage.key for stage in mod.resolve_stages(None)] == [
        "16",
        "17",
        "18",
        "identity",
    ]


def test_requested_stages_are_deduplicated_and_execution_ordered():
    stages = mod.resolve_stages(["11", "6", "10", "7", "6"])
    assert [stage.key for stage in stages] == ["6", "7", "10", "11"]


def test_named_aliases_resolve_to_same_stages():
    stages = mod.resolve_stages(
        ["contact", "head-support", "framing", "capture", "final"]
    )
    assert [stage.key for stage in stages] == [
        "11",
        "16",
        "17",
        "18",
        "identity",
    ]


def test_retired_stage_8_is_rejected():
    with pytest.raises(ValueError, match="retired"):
        mod.resolve_stages(["8"])


def test_argv_context_restores_process_arguments():
    before = list(sys.argv)
    fake_module = SimpleNamespace(__file__="/tmp/fake.py", __name__="fake")
    with mod._argv_for(fake_module, ["run", "--overwrite"]):
        assert sys.argv == ["/tmp/fake.py", "run", "--overwrite"]
    assert sys.argv == before


def test_snapshot_and_verify_exact_json(tmp_path: Path):
    stage = mod.Stage(
        key="x",
        label="test",
        module=SimpleNamespace(),
        output_subdir=Path("semantic-v3") / "test",
        order=1,
    )
    directory = tmp_path / stage.output_subdir
    directory.mkdir(parents=True)

    path = directory / "imageblind-01_00001.fact_sheet.json"
    path.write_text(
        json.dumps({"image_key": "imageblind-01_00001", "value": 1}),
        encoding="utf-8",
    )

    baseline = mod._snapshot_records(
        stage,
        tmp_path,
        ["imageblind-01_00001"],
    )
    identical, differences = mod._verify_snapshot(
        stage,
        tmp_path,
        ["imageblind-01_00001"],
        baseline,
    )
    assert identical is True
    assert differences == []

    path.write_text(
        json.dumps({"image_key": "imageblind-01_00001", "value": 2}),
        encoding="utf-8",
    )
    identical, differences = mod._verify_snapshot(
        stage,
        tmp_path,
        ["imageblind-01_00001"],
        baseline,
    )
    assert identical is False
    assert differences == ["imageblind-01_00001.fact_sheet.json"]


def test_run_stage_fork_isolates_legacy_module_mutation(tmp_path: Path):
    if not hasattr(os, "fork"):
        pytest.skip("fork isolation is POSIX-only")

    fake_module = SimpleNamespace(__name__="fake_stage", mutated=False)

    def fake_main():
        fake_module.mutated = True
        return 0

    fake_module.main = fake_main
    stage = mod.Stage(
        key="x",
        label="fake",
        module=fake_module,
        output_subdir=Path("semantic-v3") / "fake",
        order=1,
    )

    rc, _elapsed = mod.run_stage(
        stage,
        tmp_path,
        only=[],
        overwrite=False,
    )

    assert rc == 0
    assert fake_module.mutated is False


def test_stage_args_keep_one_only_batch_in_one_worker(tmp_path: Path):
    args = mod._stage_args(
        tmp_path,
        only=["imageblind-01_00001", "imageblind-01_00002"],
        overwrite=True,
    )
    assert args == [
        str(tmp_path),
        "--only",
        "imageblind-01_00001",
        "imageblind-01_00002",
        "--overwrite",
    ]
