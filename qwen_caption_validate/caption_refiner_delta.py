from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import caption_refiner_pose_vlm as base
from . import extract_v3
from . import pose_atlas_v3 as atlas
from .extract_v3_wire import _install_image_only_vllm
from .runner import generate, load_model, resolve_backend, resolve_model_id, unload_model


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "pose_card_caption_v12_delta_refiner_laterality.txt"
ARTIFACT_VERSION = "caption-refiner-delta-0.12"
RUN_VERSION = "caption-refiner-delta-0.12-run"

LATERALITY_JOINTS = (
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


def _resolve_dwpose_dir(run_dir: Path, supplied: Path | None) -> Path | None:
    if supplied is not None:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"DWPose directory not found: {value}")
        return value
    direct = run_dir / "dwpose"
    if direct.is_dir():
        return direct
    matches = sorted(p for p in run_dir.iterdir() if p.is_dir() and "dwpose" in p.name.lower())
    return matches[-1] if matches else None


def _point_is_usable(point: np.ndarray, *, reject_origin: bool = False) -> bool:
    value = np.asarray(point, dtype=np.float64).reshape(-1)
    if value.size < 2 or not np.isfinite(value[:2]).all():
        return False
    if reject_origin and abs(float(value[0])) < 1e-8 and abs(float(value[1])) < 1e-8:
        return False
    return True


def _frame_side(x: float, width: int) -> str:
    if width <= 0:
        return "unknown"
    fraction = float(x) / float(width)
    if fraction < 0.45:
        return "frame-left"
    if fraction > 0.55:
        return "frame-right"
    return "near frame-center"


def _inside_frame(point: np.ndarray, width: int, height: int) -> bool:
    if not _point_is_usable(point):
        return False
    x, y = float(point[0]), float(point[1])
    return 0.0 <= x <= float(width) and 0.0 <= y <= float(height)


def _sam3d_points(arrays: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    raw = np.asarray(arrays.get("pred_keypoints_2d", np.empty((0, 2))), dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return atlas._normalized_to_pixels(raw, width, height)


def _dwpose_points(record: dict[str, Any] | None, width: int, height: int) -> np.ndarray:
    if not record:
        return np.empty((0, 2), dtype=np.float64)
    try:
        return np.asarray(atlas._dwpose_target_points(record, width, height), dtype=np.float64)
    except Exception:
        return np.empty((0, 2), dtype=np.float64)


def _laterality_facts(
    arrays: dict[str, np.ndarray],
    width: int,
    height: int,
    dwpose_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return named laterality facts without asking the VLM to infer left/right.

    DWPose is preferred for 2-D limb laterality. SAM3D projected named joints are used
    as a fallback when the corresponding DWPose joint is unavailable. "inside_frame"
    means only that the projected joint lies within the photograph boundary; it is not
    an occlusion/visibility claim.
    """
    dw = _dwpose_points(dwpose_record, width, height)
    sam = _sam3d_points(arrays, width, height)
    facts: list[dict[str, Any]] = []

    for name in LATERALITY_JOINTS:
        point: np.ndarray | None = None
        source: str | None = None

        dw_index = atlas.IDX.get(name)
        if dw_index is not None and dw_index < len(dw) and _point_is_usable(dw[dw_index], reject_origin=True):
            point = dw[dw_index]
            source = "dwpose"
        else:
            sam_index = atlas.MHR70.get(name)
            if sam_index is not None and sam_index < len(sam) and _point_is_usable(sam[sam_index]):
                point = sam[sam_index]
                source = "sam3d"

        if point is None:
            continue

        inside = _inside_frame(point, width, height)
        anatomical_side = "subject-left" if name.startswith("left_") else "subject-right"
        joint = name.split("_", 1)[1]
        fact = {
            "joint": joint,
            "anatomical_side": anatomical_side,
            "inside_frame": inside,
            "source": source,
        }
        if inside:
            fact["frame_side"] = _frame_side(float(point[0]), width)
        else:
            fact["frame_side"] = "outside photograph"
        facts.append(fact)

    return facts


def _format_laterality_facts(facts: list[dict[str, Any]]) -> str:
    if not facts:
        return "- No reliable named laterality facts are available; do not make a left/right correction."
    lines: list[str] = []
    for fact in facts:
        subject_side = str(fact.get("anatomical_side") or "unknown")
        joint = str(fact.get("joint") or "joint")
        frame_side = str(fact.get("frame_side") or "unknown")
        if fact.get("inside_frame"):
            lines.append(f"- {subject_side} {joint}: {frame_side}, inside photograph")
        else:
            lines.append(f"- {subject_side} {joint}: outside photograph")
    return "\n".join(lines)


def _render_prompt(template: str, caption: str, laterality_text: str) -> str:
    return (
        template
        .replace("{{CURRENT_CAPTION}}", caption.strip())
        .replace("{{LATERALITY_FACTS}}", laterality_text.strip())
    )


def _generate_vllm_batch_prompts(
    loaded,
    image_paths: list[Path],
    prompts: list[str],
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from vllm import SamplingParams

    if len(image_paths) != len(prompts):
        raise ValueError("image_paths and prompts must have the same length")

    requests: list[dict[str, Any]] = []
    prepare_started = time.perf_counter()
    for image_path, prompt in zip(image_paths, prompts):
        requests.append(extract_v3.runner_module._prepare_vllm_multimodal(loaded, image_path, prompt))
    prepare_seconds = time.perf_counter() - prepare_started

    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    generation_started = time.perf_counter()
    outputs = loaded.model.generate(requests, sampling_params=sampling, use_tqdm=False)
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(image_paths):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(image_paths)} delta-refiner requests")

    items = [extract_v3._request_perf_fields(output, max_new_tokens) for output in outputs]
    return items, {
        "prepare_seconds": prepare_seconds,
        "generation_seconds": generation_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption-aware SAM3D geometry delta refiner with deterministic laterality facts."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--captions-dir", type=Path, required=True)
    parser.add_argument("--caption-field")
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--sam3d-dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    captions_dir = args.captions_dir.expanduser().resolve()
    images_dir = (args.images_dir.expanduser().resolve() if args.images_dir else run_dir / "images")
    sam3d_dir = base._resolve_sam3d_dir(run_dir, args.sam3d_dir)
    dwpose_dir = _resolve_dwpose_dir(run_dir, args.dwpose_dir)
    pose_dir = args.pose_dir.expanduser().resolve() if args.pose_dir else None
    output_dir = (args.output_dir or run_dir / "caption-refiner-delta-v0.12").expanduser().resolve()
    cards_dir = output_dir / "pose-cards"
    assets_dir = output_dir / "assets"

    for directory, label in ((run_dir, "Run"), (captions_dir, "Caption"), (images_dir, "Images")):
        if not directory.is_dir():
            print(f"{label} directory not found: {directory}", file=sys.stderr)
            return 2
    if args.batch_size < 1 or args.max_tokens < 1:
        print("--batch-size and --max-tokens must be >= 1", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.prompt.expanduser().resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")

    images = [
        p for p in atlas._discover_images(images_dir)
        if atlas._matches_only(atlas._image_key(p), p, args.only)
    ]
    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path, Path, str]] = []
    missing: list[dict[str, str]] = []

    for image_path in images:
        key = atlas._image_key(image_path)
        sam_npz = base._find_for_key(sam3d_dir, key, (".sam3d_arrays.npz",))
        caption_json = base._find_for_key(captions_dir, key, (".json",))
        if sam_npz is None:
            missing.append({"image_key": key, "reason": "missing_sam3d_arrays"})
            continue
        if caption_json is None:
            missing.append({"image_key": key, "reason": "missing_caption_json"})
            continue

        caption_payload = base._read_json(caption_json)
        caption, caption_field = base._caption_from_payload(caption_payload, args.caption_field)
        if not caption:
            missing.append({"image_key": key, "reason": "caption_field_not_found"})
            continue

        sam_obj = base._find_for_key(sam3d_dir, key, (".sam3d.obj", ".obj"))
        dwpose_json = base._find_for_key(dwpose_dir, key, (".json",)) if dwpose_dir else None
        dwpose_record = base._read_json(dwpose_json) if dwpose_json else None
        arrays = base._load_arrays(sam_npz)
        with Image.open(image_path) as source:
            width, height = source.size

        laterality = _laterality_facts(arrays, width, height, dwpose_record)
        laterality_text = _format_laterality_facts(laterality)
        rendered_prompt = _render_prompt(prompt_template, caption, laterality_text)

        pose_card = cards_dir / f"{key}.pose_card.webp"
        if args.overwrite or not pose_card.exists():
            card_meta = base.make_pose_card(image_path, sam_npz, sam_obj, pose_card)
        else:
            card_meta = {
                "source_image_size": [width, height],
                "sam3d_arrays": str(sam_npz),
                "sam3d_mesh": str(sam_obj) if sam_obj else None,
                "pose_card": str(pose_card),
            }

        image_asset = assets_dir / f"{key}{image_path.suffix.lower()}"
        if args.overwrite or not image_asset.exists():
            shutil.copy2(image_path, image_asset)
        card_asset = assets_dir / pose_card.name
        if args.overwrite or not card_asset.exists():
            shutil.copy2(pose_card, card_asset)

        out_json = output_dir / f"{key}.delta_refiner.json"
        existing = base._read_json(out_json) if out_json.exists() and not args.overwrite else {}
        record = {
            "schema_version": ARTIFACT_VERSION,
            "image_key": key,
            "image": str(image_path),
            "image_asset": f"assets/{image_asset.name}",
            "caption_source": str(caption_json),
            "caption_field": caption_field,
            "existing_caption": caption,
            "spatial_review_terms": base._spatial_terms(caption),
            "pose_card_asset": f"assets/{card_asset.name}",
            "pose_card": card_meta,
            "laterality_facts": laterality,
            "laterality_source_preference": "dwpose_then_sam3d_fallback",
            "existing_pose_language": base._pose_language_for_key(pose_dir, key),
            "pose_delta": existing.get("pose_delta") if existing else None,
        }
        records.append(record)
        if not args.dry_run and (args.overwrite or not record.get("pose_delta")):
            pending.append((record, pose_card, out_json, rendered_prompt))
        else:
            base._write_json(out_json, record)

    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)
    loaded = None
    model_load_seconds = 0.0
    if pending:
        if backend == "vllm":
            _install_image_only_vllm()
        print(f"Loading {model_id} for {len(pending)} caption-delta refinement(s) ...")
        loaded = load_model(
            model_id,
            backend=backend,
            dtype=args.dtype,
            quantization="none",
            cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
        )
        model_load_seconds = float(loaded.load_seconds)
        print(f"Loaded in {model_load_seconds:.2f}s")

    try:
        if loaded is not None and loaded.backend == "vllm":
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset:offset + args.batch_size]
                items, perf = _generate_vllm_batch_prompts(
                    loaded,
                    [row[1] for row in batch],
                    [row[3] for row in batch],
                    max_new_tokens=args.max_tokens,
                )
                for (record, _, out_json, _), item in zip(batch, items):
                    text = str(item.get("text") or "").strip()
                    sentences = base._split_sentences(text) if text != "NO_CORRECTION" else []
                    record["pose_delta"] = {
                        "text": text,
                        "no_correction": text == "NO_CORRECTION",
                        "sentences": sentences,
                        "model": model_id,
                        "backend": loaded.backend,
                        "prompt_template": str(prompt_path),
                        "finish_reason": item.get("finish_reason"),
                        "prompt_tokens": item.get("prompt_tokens"),
                        "output_tokens": item.get("output_tokens"),
                        "batch_generation_seconds": perf.get("generation_seconds"),
                    }
                    base._write_json(out_json, record)
                    print(f"{record['image_key']}: {text}")
        elif loaded is not None:
            for record, pose_card, out_json, rendered_prompt in pending:
                text, inference_seconds = generate(
                    loaded, pose_card, rendered_prompt, max_new_tokens=args.max_tokens
                )
                text = text.strip()
                sentences = base._split_sentences(text) if text != "NO_CORRECTION" else []
                record["pose_delta"] = {
                    "text": text,
                    "no_correction": text == "NO_CORRECTION",
                    "sentences": sentences,
                    "model": model_id,
                    "backend": loaded.backend,
                    "prompt_template": str(prompt_path),
                    "inference_seconds": inference_seconds,
                }
                base._write_json(out_json, record)
                print(f"{record['image_key']}: {text}")
    finally:
        if loaded is not None:
            unload_model(loaded)

    indexed: list[dict[str, Any]] = []
    for record in records:
        path = output_dir / f"{record['image_key']}.delta_refiner.json"
        indexed.append(base._read_json(path) if path.exists() else record)

    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "captions_dir": str(captions_dir),
        "caption_field": args.caption_field,
        "images_dir": str(images_dir),
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir) if dwpose_dir else None,
        "pose_dir": str(pose_dir) if pose_dir else None,
        "prompt": str(prompt_path),
        "model": model_id,
        "backend": backend,
        "model_load_seconds": model_load_seconds,
        "dry_run": bool(args.dry_run),
        "record_count": len(indexed),
        "generated_pose_deltas": sum(1 for r in indexed if r.get("pose_delta")),
        "no_correction_count": sum(
            1 for r in indexed if (r.get("pose_delta") or {}).get("no_correction") is True
        ),
        "missing": missing,
        "records": indexed,
    }
    base._write_json(output_dir / "caption_refiner_delta.index.json", index)
    print(f"Caption delta-refiner bundle: {output_dir}")
    print(
        f"Records: {len(indexed)}; deltas: {index['generated_pose_deltas']}; "
        f"NO_CORRECTION: {index['no_correction_count']}; missing: {len(missing)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
