from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import extract_v3
from . import pose_atlas_v3 as atlas
from .extract_v3_wire import _install_image_only_vllm
from .runner import generate, load_model, model_slug, resolve_backend, resolve_model_id, unload_model
from .semantic_v3_rich_caption import _generate_vllm_batch

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "pose_card_caption_v01.txt"
ARTIFACT_VERSION = "caption-refiner-pose-vlm-0.1"
RUN_VERSION = "caption-refiner-pose-vlm-0.1-run"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PANEL_W = 500
PANEL_H = 560
TITLE_H = 42
CARD_BG = "#0b0d10"
PANEL_BG = "#11151b"
FRAME_FILL = "#202832"
FRAME_OUTLINE = "#f4f6f8"
BODY_LINE = "#8bd5ff"
BODY_POINT = "#ffffff"
MESH_POINT = "#d2d7df"
MUTED = "#9da7b3"


def _font(size: int = 16, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _get_dotted(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _caption_from_payload(payload: dict[str, Any], field: str | None) -> tuple[str, str | None]:
    if field:
        value = _get_dotted(payload, field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
        return "", None

    preferred = [
        "final_caption",
        "training_caption",
        "caption",
        "output.caption",
        "result.caption",
        "text",
    ]
    for candidate in preferred:
        value = _get_dotted(payload, candidate)
        if isinstance(value, str) and len(value.strip().split()) >= 5:
            return value.strip(), candidate

    found: list[tuple[int, str, str]] = []
    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, str) and "caption" in str(key).lower() and len(child.strip().split()) >= 5:
                    found.append((len(child), name, child.strip()))
                elif isinstance(child, (dict, list)):
                    walk(child, name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{prefix}.{index}" if prefix else str(index))
    walk(payload)
    if not found:
        return "", None
    found.sort(reverse=True)
    _, name, text = found[0]
    return text, name


def _find_for_key(directory: Path | None, key: str, suffixes: tuple[str, ...]) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    for suffix in suffixes:
        direct = directory / f"{key}{suffix}"
        if direct.is_file():
            return direct
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend(directory.rglob(f"{key}*{suffix}"))
    return sorted(set(candidates))[0] if candidates else None


def _resolve_sam3d_dir(run_dir: Path, supplied: Path | None) -> Path:
    if supplied:
        value = supplied.expanduser().resolve()
        if not value.is_dir():
            raise SystemExit(f"SAM3D directory not found: {value}")
        return value
    value = atlas._resolve_dir(run_dir, None, ["sam3d", "sam3d-probe"], "sam3d")
    if value is None:
        raise SystemExit("Could not locate cached SAM3D data; pass --sam3d-dir.")
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _selected_body_indices() -> list[int]:
    names = [
        "nose", "neck",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]
    return [atlas.MHR70[name] for name in names if name in atlas.MHR70]


def _finite_xy(point: np.ndarray) -> bool:
    return bool(len(point) >= 2 and np.isfinite(point[:2]).all())


def _panel(title: str) -> Image.Image:
    image = Image.new("RGB", (PANEL_W, PANEL_H), PANEL_BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, PANEL_W, TITLE_H), fill="#1a2028")
    draw.text((14, 11), title, fill="#f5f7fa", font=_font(17, True))
    return image


def _projection_panel(sam2d: np.ndarray, width: int, height: int) -> Image.Image:
    panel = _panel("Full-body projection + photograph frame")
    draw = ImageDraw.Draw(panel)

    indices = _selected_body_indices()
    points = [np.asarray(sam2d[i, :2], dtype=np.float64) for i in indices if i < len(sam2d) and _finite_xy(sam2d[i])]
    if not points:
        draw.text((18, 72), "No projected body points available", fill=MUTED, font=_font(16))
        return panel

    cloud = np.stack(points)
    min_x = min(float(cloud[:, 0].min()), 0.0)
    max_x = max(float(cloud[:, 0].max()), float(width))
    min_y = min(float(cloud[:, 1].min()), 0.0)
    max_y = max(float(cloud[:, 1].max()), float(height))
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    min_x -= span_x * 0.08
    max_x += span_x * 0.08
    min_y -= span_y * 0.08
    max_y += span_y * 0.08

    left, top, right, bottom = 28, TITLE_H + 24, PANEL_W - 28, PANEL_H - 30
    scale = min((right-left)/(max_x-min_x), (bottom-top)/(max_y-min_y))
    ox = left + ((right-left) - (max_x-min_x)*scale)/2.0
    oy = top + ((bottom-top) - (max_y-min_y)*scale)/2.0

    def xy(p: np.ndarray | tuple[float, float]) -> tuple[float, float]:
        x, y = float(p[0]), float(p[1])
        return ox + (x-min_x)*scale, oy + (y-min_y)*scale

    fx0, fy0 = xy((0.0, 0.0))
    fx1, fy1 = xy((float(width), float(height)))
    draw.rectangle((fx0, fy0, fx1, fy1), fill=FRAME_FILL, outline=FRAME_OUTLINE, width=3)
    draw.text((fx0 + 8, fy0 + 8), "PHOTOGRAPH FRAME", fill="#d9e0e8", font=_font(12, True))

    edges = [(atlas.MHR70[a], atlas.MHR70[b]) for a, b in atlas.MHR_KNOWN_EDGES if a in atlas.MHR70 and b in atlas.MHR70]
    for a, b in edges:
        if a >= len(sam2d) or b >= len(sam2d) or not _finite_xy(sam2d[a]) or not _finite_xy(sam2d[b]):
            continue
        draw.line((*xy(sam2d[a]), *xy(sam2d[b])), fill=BODY_LINE, width=4)
    for i in indices:
        if i >= len(sam2d) or not _finite_xy(sam2d[i]):
            continue
        x, y = xy(sam2d[i])
        r = 4
        draw.ellipse((x-r, y-r, x+r, y+r), fill=BODY_POINT, outline="#101215")

    draw.text((18, PANEL_H-22), "Body outside the rectangle is reconstructed beyond the photograph crop.", fill=MUTED, font=_font(11))
    return panel


def _mesh_panel(vertices_body: np.ndarray, axes: tuple[int, int], title: str) -> Image.Image:
    panel = _panel(title)
    draw = ImageDraw.Draw(panel)
    if vertices_body.size == 0:
        draw.text((18, 72), "No cached mesh available", fill=MUTED, font=_font(16))
        return panel
    finite = vertices_body[np.isfinite(vertices_body).all(axis=1)]
    if not len(finite):
        return panel
    step = max(1, len(finite) // 9000)
    cloud = finite[::step]
    x = cloud[:, axes[0]]
    y = cloud[:, axes[1]]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(1e-8, xmax-xmin), max(1e-8, ymax-ymin)
    left, top, right, bottom = 30, TITLE_H + 26, PANEL_W - 30, PANEL_H - 28
    scale = min((right-left)/dx, (bottom-top)/dy)
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    ox, oy = (left+right)/2.0, (top+bottom)/2.0
    for px, py in zip(x, y):
        sx = ox + (float(px)-cx)*scale
        sy = oy - (float(py)-cy)*scale
        draw.point((sx, sy), fill=MESH_POINT)
    return panel


def make_pose_card(image_path: Path, sam_npz: Path, sam_obj: Path | None, output: Path) -> dict[str, Any]:
    with Image.open(image_path) as source:
        width, height = source.size
    arrays = _load_arrays(sam_npz)
    sam2d = np.asarray(arrays.get("pred_keypoints_2d", np.empty((0, 2))), dtype=np.float64)
    vertices = atlas._load_obj_vertices(sam_obj)
    vertices_body = atlas._body_frame_vertices(vertices, arrays) if vertices.size else vertices

    panels = [
        _projection_panel(sam2d, width, height),
        _mesh_panel(vertices_body, (0, 1), "3D body — front/camera plane"),
        _mesh_panel(vertices_body, (2, 1), "3D body — side/depth"),
    ]
    card = Image.new("RGB", (PANEL_W * 3, PANEL_H), CARD_BG)
    for index, panel in enumerate(panels):
        card.paste(panel, (index * PANEL_W, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    card.save(output, format="WEBP", quality=92, method=6)
    return {
        "source_image_size": [width, height],
        "sam3d_arrays": str(sam_npz),
        "sam3d_mesh": str(sam_obj) if sam_obj else None,
        "pose_card": str(output),
    }


def _pose_language_for_key(pose_dir: Path | None, key: str) -> dict[str, Any] | None:
    path = _find_for_key(pose_dir, key, (".pose_language.json", ".json")) if pose_dir else None
    return _read_json(path) if path else None


def _spatial_terms(text: str) -> list[str]:
    pattern = re.compile(r"\b(?:frame\s+left|frame\s+right|image\s+left|image\s+right|left|right|behind|foreground|in front of)\b", re.I)
    seen: list[str] = []
    for match in pattern.finditer(text or ""):
        value = match.group(0)
        if value.lower() not in {item.lower() for item in seen}:
            seen.append(value)
    return seen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype caption refiner: existing JSON caption + SAM3D pose-card VLM reference.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--captions-dir", type=Path, required=True, help="Directory containing the existing JSON-derived caption artifacts.")
    parser.add_argument("--caption-field", help="Optional dotted JSON path for the existing caption, e.g. caption or result.caption.")
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--sam3d-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path, help="Optional existing pose-language-v0.1 directory for side-by-side diagnostics.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="32b-fp8")
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="vllm")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true", help="Build cards/index without loading the VLM.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    captions_dir = args.captions_dir.expanduser().resolve()
    if not captions_dir.is_dir():
        print(f"Caption JSON directory not found: {captions_dir}", file=sys.stderr)
        return 2
    images_dir = (args.images_dir.expanduser().resolve() if args.images_dir else run_dir / "images")
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 2
    sam3d_dir = _resolve_sam3d_dir(run_dir, args.sam3d_dir)
    pose_dir = args.pose_dir.expanduser().resolve() if args.pose_dir else None
    output_dir = (args.output_dir or run_dir / "caption-refiner-pose-vlm-v0.1").expanduser().resolve()
    cards_dir = output_dir / "pose-cards"
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = args.prompt.expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    images = [p for p in atlas._discover_images(images_dir) if atlas._matches_only(atlas._image_key(p), p, args.only)]
    records: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path, Path]] = []
    missing: list[dict[str, str]] = []

    for image_path in images:
        key = atlas._image_key(image_path)
        sam_npz = _find_for_key(sam3d_dir, key, (".sam3d_arrays.npz",))
        if sam_npz is None:
            missing.append({"image_key": key, "reason": "missing_sam3d_arrays"})
            continue
        sam_obj = _find_for_key(sam3d_dir, key, (".sam3d.obj", ".obj"))
        caption_json = _find_for_key(captions_dir, key, (".json",))
        if caption_json is None:
            missing.append({"image_key": key, "reason": "missing_caption_json"})
            continue
        caption_payload = _read_json(caption_json)
        caption, caption_field = _caption_from_payload(caption_payload, args.caption_field)
        if not caption:
            missing.append({"image_key": key, "reason": "caption_field_not_found"})
            continue

        pose_card = cards_dir / f"{key}.pose_card.webp"
        if args.overwrite or not pose_card.exists():
            card_meta = make_pose_card(image_path, sam_npz, sam_obj, pose_card)
        else:
            with Image.open(image_path) as source:
                card_meta = {
                    "source_image_size": list(source.size),
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

        out_json = output_dir / f"{key}.pose_refiner.json"
        existing = _read_json(out_json) if out_json.exists() and not args.overwrite else {}
        record = {
            "schema_version": ARTIFACT_VERSION,
            "image_key": key,
            "image": str(image_path),
            "image_asset": f"assets/{image_asset.name}",
            "caption_source": str(caption_json),
            "caption_field": caption_field,
            "existing_caption": caption,
            "spatial_review_terms": _spatial_terms(caption),
            "pose_card_asset": f"assets/{card_asset.name}",
            "pose_card": card_meta,
            "existing_pose_language": _pose_language_for_key(pose_dir, key),
            "pose_vlm": existing.get("pose_vlm") if existing else None,
        }
        records.append(record)
        if not args.dry_run and (args.overwrite or not record.get("pose_vlm")):
            pending.append((record, pose_card, out_json))
        else:
            _write_json(out_json, record)

    model_id = resolve_model_id(args.model)
    backend = resolve_backend(model_id, args.backend)
    loaded = None
    model_load_seconds = 0.0
    if pending:
        if backend == "vllm":
            _install_image_only_vllm()
        print(f"Loading {model_id} for {len(pending)} pose-card caption(s) ...")
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
            for offset in range(0, len(pending), max(1, args.batch_size)):
                batch = pending[offset:offset + max(1, args.batch_size)]
                items, perf = _generate_vllm_batch(loaded, [row[1] for row in batch], prompt, max_new_tokens=args.max_tokens)
                for (record, _, out_json), item in zip(batch, items):
                    text = str(item.get("text") or "").strip()
                    sentences = _split_sentences(text)
                    record["pose_vlm"] = {
                        "text": text,
                        "sentences": sentences,
                        "sentence_1": sentences[0] if len(sentences) > 0 else "",
                        "sentence_2": sentences[1] if len(sentences) > 1 else "",
                        "exactly_two_sentences": len(sentences) == 2,
                        "model": model_id,
                        "backend": loaded.backend,
                        "prompt": str(prompt_path),
                        "finish_reason": item.get("finish_reason"),
                        "prompt_tokens": item.get("prompt_tokens"),
                        "output_tokens": item.get("output_tokens"),
                        "batch_generation_seconds": perf.get("generation_seconds"),
                    }
                    _write_json(out_json, record)
                    print(f"{record['image_key']}: {len(sentences)} sentence(s) — {text}")
        elif loaded is not None:
            for record, pose_card, out_json in pending:
                text, inference_seconds = generate(loaded, pose_card, prompt, max_new_tokens=args.max_tokens)
                text = text.strip()
                sentences = _split_sentences(text)
                record["pose_vlm"] = {
                    "text": text,
                    "sentences": sentences,
                    "sentence_1": sentences[0] if len(sentences) > 0 else "",
                    "sentence_2": sentences[1] if len(sentences) > 1 else "",
                    "exactly_two_sentences": len(sentences) == 2,
                    "model": model_id,
                    "backend": loaded.backend,
                    "prompt": str(prompt_path),
                    "inference_seconds": inference_seconds,
                }
                _write_json(out_json, record)
                print(f"{record['image_key']}: {len(sentences)} sentence(s) — {text}")
    finally:
        if loaded is not None:
            unload_model(loaded)

    # Reload records so the index always contains newly generated pose text.
    indexed: list[dict[str, Any]] = []
    for record in records:
        path = output_dir / f"{record['image_key']}.pose_refiner.json"
        indexed.append(_read_json(path) if path.exists() else record)
    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "captions_dir": str(captions_dir),
        "caption_field": args.caption_field,
        "images_dir": str(images_dir),
        "sam3d_dir": str(sam3d_dir),
        "pose_dir": str(pose_dir) if pose_dir else None,
        "prompt": str(prompt_path),
        "model": model_id,
        "backend": backend,
        "model_load_seconds": model_load_seconds,
        "dry_run": bool(args.dry_run),
        "record_count": len(indexed),
        "generated_pose_refs": sum(1 for r in indexed if r.get("pose_vlm")),
        "missing": missing,
        "records": indexed,
    }
    _write_json(output_dir / "caption_refiner.index.json", index)
    print(f"Caption refiner bundle: {output_dir}")
    print(f"Records: {len(indexed)}; pose refs: {index['generated_pose_refs']}; missing: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
