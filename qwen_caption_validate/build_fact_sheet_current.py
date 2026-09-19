from __future__ import annotations

"""Warm-process orchestrator for the deterministic caption fact-sheet stack.

The numbered normalizer modules remain the source of truth. This module does
not reimplement their semantics. It imports the full deterministic stack once,
then runs each requested legacy stage in an isolated forked worker.

The fork boundary is deliberate: several historical normalizer main() functions
temporarily monkeypatch lower modules to assemble their cumulative stage. A
fresh standalone Python process naturally isolates those mutations. Forking
from one already-imported parent preserves that isolation while paying the
expensive Python/NumPy/Pillow/module import cost only once.

--stages selects which stage artifacts are materialized. Each selected
numbered stage remains cumulative according to its existing module wiring.

The current production path is:

    16 -> 17 -> 18 -> identity

Stage 8 is intentionally omitted because the current pipeline bypasses that
retired knee-angle shadow experiment.
"""

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

from . import caption_policy_identity_v11 as identity_v11
from . import fact_sheet_specialist_normalizer_v01 as v01
from . import fact_sheet_specialist_normalizer_v02 as v02
from . import fact_sheet_specialist_normalizer_v03 as v03
from . import fact_sheet_specialist_normalizer_v04 as v04
from . import fact_sheet_specialist_normalizer_v05 as v05
from . import fact_sheet_specialist_normalizer_v06 as v06
from . import fact_sheet_specialist_normalizer_v07 as v07
from . import fact_sheet_specialist_normalizer_v09 as v09
from . import fact_sheet_specialist_normalizer_v10 as v10
from . import fact_sheet_specialist_normalizer_v11 as v11
from . import fact_sheet_specialist_normalizer_v12 as v12
from . import fact_sheet_specialist_normalizer_v13 as v13
from . import fact_sheet_specialist_normalizer_v14 as v14
from . import fact_sheet_specialist_normalizer_v15 as v15
from . import fact_sheet_specialist_normalizer_v16 as v16
from . import fact_sheet_specialist_normalizer_v17 as v17
from . import fact_sheet_specialist_normalizer_v18 as v18


def _fact_sheet_output_subdir(module: ModuleType) -> Path:
    """Return the historical output namespace for a fact-sheet stage.

    Most successor modules define DEFAULT_OUTPUT_SUBDIR directly.  A small
    number of early wrappers (notably v02) only replace the inherited writer
    schema, so derive their namespace from SCHEMA_VERSION.
    """
    value = getattr(module, "DEFAULT_OUTPUT_SUBDIR", None)
    if value is not None:
        return Path(value)

    schema = str(getattr(module, "SCHEMA_VERSION", ""))
    prefix = "caption-fact-sheet-"
    if not schema.startswith(prefix):
        raise AttributeError(
            f"{module.__name__} has neither DEFAULT_OUTPUT_SUBDIR nor a "
            f"{prefix!r} SCHEMA_VERSION"
        )
    return Path("semantic-v3") / f"caption-fact-sheet-v{schema[len(prefix):]}"


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    module: ModuleType
    output_subdir: Path
    order: int
    production_default: bool = False


STAGES: tuple[Stage, ...] = (
    Stage("1", "base-specialist-normalizer", v01, _fact_sheet_output_subdir(v01), 1),
    Stage("2", "torso-scope-head-authority", v02, _fact_sheet_output_subdir(v02), 2),
    Stage("3", "torso-orientation-enrichment", v03, _fact_sheet_output_subdir(v03), 3),
    Stage("4", "gaze-caption-semantics", v04, _fact_sheet_output_subdir(v04), 4),
    Stage("5", "relation-laterality", v05, _fact_sheet_output_subdir(v05), 5),
    Stage("6", "leg-relation-laterality", v06, _fact_sheet_output_subdir(v06), 6),
    Stage("7", "support-shape", v07, _fact_sheet_output_subdir(v07), 7),
    Stage("9", "sam3d-pose-torso-shadow", v09, _fact_sheet_output_subdir(v09), 9),
    Stage("10", "broad-pose-torso-authority", v10, _fact_sheet_output_subdir(v10), 10),
    Stage("11", "support-contact-truth", v11, _fact_sheet_output_subdir(v11), 11),
    Stage("12", "support-topology-truth", v12, _fact_sheet_output_subdir(v12), 12),
    Stage("13", "bilateral-knee-flexion", v13, _fact_sheet_output_subdir(v13), 13),
    Stage("14", "unilateral-raised-leg", v14, _fact_sheet_output_subdir(v14), 14),
    Stage("15", "crouch-depth", v15, _fact_sheet_output_subdir(v15), 15),
    Stage("16", "head-support-current-specialist-stack", v16, _fact_sheet_output_subdir(v16), 16, True),
    Stage("17", "framing-authority", v17, _fact_sheet_output_subdir(v17), 17, True),
    Stage("18", "capture-authority", v18, _fact_sheet_output_subdir(v18), 18, True),
    Stage(
        "identity",
        "identity-policy-final-fact-sheet",
        identity_v11,
        _fact_sheet_output_subdir(identity_v11),
        19,
        True,
    ),
)

_STAGE_BY_KEY = {stage.key: stage for stage in STAGES}
_STAGE_BY_LABEL = {stage.label: stage for stage in STAGES}

ALIASES = {
    "19": "identity",
    "relation": "5",
    "leg": "6",
    "support": "7",
    "pose-shadow": "9",
    "pose": "10",
    "contact": "11",
    "topology": "12",
    "knees": "13",
    "raised-leg": "14",
    "crouch-depth": "15",
    "head-support": "16",
    "framing": "17",
    "capture": "18",
    "final": "identity",
}

DEFAULT_STAGE_KEYS = tuple(stage.key for stage in STAGES if stage.production_default)


def _resolve_stage(value: str) -> Stage:
    token = str(value).strip().lower()
    token = ALIASES.get(token, token)
    if token == "8":
        raise ValueError(
            "stage 8 is retired and intentionally bypassed by the current pipeline; "
            "use stage 9 for the active SAM3D-v16 shadow adjudication"
        )
    stage = _STAGE_BY_KEY.get(token) or _STAGE_BY_LABEL.get(token)
    if stage is None:
        valid = ", ".join(stage.key for stage in STAGES)
        raise ValueError(f"unknown stage {value!r}; valid stages: {valid}, 19")
    return stage


def resolve_stages(values: Iterable[str] | None) -> list[Stage]:
    raw = list(values or DEFAULT_STAGE_KEYS)
    resolved: dict[str, Stage] = {}
    for value in raw:
        stage = _resolve_stage(value)
        resolved[stage.key] = stage
    return sorted(resolved.values(), key=lambda stage: stage.order)


@contextlib.contextmanager
def _argv_for(module: ModuleType, args: list[str]):
    old = sys.argv
    sys.argv = [getattr(module, "__file__", module.__name__), *args]
    try:
        yield
    finally:
        sys.argv = old


def _stage_args(
    stage: Stage,
    run_dir: Path,
    *,
    only: list[str],
    overwrite: bool,
) -> list[str]:
    # Always pin the output namespace explicitly.  This matters for early
    # wrapper stages such as v02, whose standalone module inherits the v01
    # writer defaults and otherwise has no stage-specific output constant.
    args = [
        str(run_dir),
        "--output-dir",
        str(run_dir / stage.output_subdir),
    ]
    if only:
        args.extend(["--only", *only])
    if overwrite:
        args.append("--overwrite")
    return args


def _record_paths(stage: Stage, run_dir: Path, only: list[str]) -> list[Path]:
    directory = run_dir / stage.output_subdir
    paths = sorted(directory.glob("*.fact_sheet.json")) if directory.is_dir() else []
    if not only:
        return paths
    requested = set(only)
    return [
        path
        for path in paths
        if path.name.removesuffix(".fact_sheet.json") in requested
    ]


def _snapshot_records(
    stage: Stage,
    run_dir: Path,
    only: list[str],
) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for path in _record_paths(stage, run_dir, only):
        snapshot[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return snapshot


def _verify_snapshot(
    stage: Stage,
    run_dir: Path,
    only: list[str],
    baseline: dict[str, object],
) -> tuple[bool, list[str]]:
    current = _snapshot_records(stage, run_dir, only)
    names = sorted(set(baseline) | set(current))
    differences = [
        name
        for name in names
        if baseline.get(name) != current.get(name)
    ]
    return not differences, differences


def _run_stage_main(stage: Stage, args: list[str]) -> int:
    with _argv_for(stage.module, args):
        return int(stage.module.main())


def run_stage(
    stage: Stage,
    run_dir: Path,
    *,
    only: list[str],
    overwrite: bool,
) -> tuple[int, float]:
    """Run one legacy stage with standalone-process semantics but warm imports.

    Every stage is forked from the same pristine, fully imported parent. The
    child may freely perform the historical monkeypatching used by that stage;
    those mutations disappear when the child exits and therefore cannot leak
    into another requested stage.
    """
    args = _stage_args(stage, run_dir, only=only, overwrite=overwrite)
    started = time.perf_counter()

    if not hasattr(os, "fork"):
        print(
            "WARNING: os.fork unavailable; running stage in-process. "
            "Multiple historical stages may not be isolation-equivalent.",
            file=sys.stderr,
        )
        rc = _run_stage_main(stage, args)
        return rc, time.perf_counter() - started

    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        try:
            rc = _run_stage_main(stage, args)
        except SystemExit as exc:
            rc = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
        except BaseException:
            traceback.print_exc()
            rc = 1
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(int(rc))

    _, status = os.waitpid(pid, 0)
    rc = os.waitstatus_to_exitcode(status)
    elapsed = time.perf_counter() - started
    return int(rc), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic caption fact-sheet stages from one warmed Python "
            "orchestrator. Stage workers are fork-isolated so legacy monkeypatch "
            "semantics match standalone runs. Default: 16 -> 17 -> 18 -> identity."
        )
    )
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument(
        "--stages",
        nargs="+",
        help=(
            "Artifacts to materialize. Numeric stages and aliases are accepted, "
            "for example: --stages 6 7 10 11 or --stages head-support framing capture final."
        ),
    )
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Snapshot existing selected record JSON before each stage, rerun it, "
            "and require exact JSON equality. Intended for consolidation validation."
        ),
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print stage numbers, names, and output namespaces, then exit.",
    )
    return parser.parse_args()


def _print_stage_table() -> None:
    print("stage\tname\toutput")
    for stage in STAGES:
        marker = " [default]" if stage.production_default else ""
        print(f"{stage.key}\t{stage.label}{marker}\t{stage.output_subdir}")
    print("19\tidentity-policy-final-fact-sheet [alias for identity]\t"
          f"{identity_v11.DEFAULT_OUTPUT_SUBDIR}")
    print("8\tRETIRED (bypassed by current pipeline)\t-")


def main() -> int:
    args = parse_args()
    if args.list_stages:
        _print_stage_table()
        return 0

    if args.run_dir is None:
        print("run_dir is required unless --list-stages is used", file=sys.stderr)
        return 2

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    try:
        stages = resolve_stages(args.stages)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    only = [str(value) for value in args.only]
    print(
        "build-fact-sheet: one warm import process + isolated stage forks | "
        + " -> ".join(f"{stage.key}:{stage.label}" for stage in stages)
    )
    if only:
        print(f"build-fact-sheet: selected records={len(only)}")

    total_started = time.perf_counter()
    timings: list[tuple[Stage, float]] = []

    for stage in stages:
        print()
        print(f"===== stage {stage.key}: {stage.label} =====")

        baseline: dict[str, object] = {}
        if args.verify_existing:
            baseline = _snapshot_records(stage, run_dir, only)
            if not baseline:
                print(
                    f"cannot verify stage {stage.key}: no existing selected "
                    f"fact-sheet records in {run_dir / stage.output_subdir}",
                    file=sys.stderr,
                )
                return 2
            print(f"verification baseline: {len(baseline)} record(s)")

        rc, elapsed = run_stage(
            stage,
            run_dir,
            only=only,
            overwrite=bool(args.overwrite or args.verify_existing),
        )
        timings.append((stage, elapsed))
        print(f"stage {stage.key} elapsed: {elapsed:.3f}s")
        if rc != 0:
            print(
                f"build-fact-sheet stopped: stage {stage.key} returned {rc}",
                file=sys.stderr,
            )
            return rc

        if args.verify_existing:
            identical, differences = _verify_snapshot(
                stage,
                run_dir,
                only,
                baseline,
            )
            if not identical:
                print(
                    f"stage {stage.key} deterministic equivalence FAILED: "
                    f"{len(differences)} differing record(s)",
                    file=sys.stderr,
                )
                for name in differences[:20]:
                    print(f"  {name}", file=sys.stderr)
                if len(differences) > 20:
                    print(
                        f"  ... and {len(differences) - 20} more",
                        file=sys.stderr,
                    )
                return 1
            print(
                f"stage {stage.key} deterministic equivalence: "
                f"PASS ({len(baseline)}/{len(baseline)} exact JSON)"
            )

    total = time.perf_counter() - total_started
    print()
    print("===== build-fact-sheet timings =====")
    for stage, elapsed in timings:
        print(f"{stage.key:>8}  {elapsed:8.3f}s  {stage.label}")
    print(f"{'total':>8}  {total:8.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
