from __future__ import annotations

from pathlib import Path

import cv2

from . import gaze_probe_ptgaze as impl
from .dwpose_compat import target_points_from_profile_record


def _discover_unique_images(path: Path) -> list[Path]:
    """Return one deterministic image path per stem.

    The validation workspace can contain repeated basenames in nested folders or
    multiple image formats. The gaze artifact key is the stem, so processing more
    than one path for the same stem would repeatedly overwrite the same JSON/PNG
    while also duplicating index records. Keep the first sorted path and report the
    ignored duplicates.
    """
    paths = impl._discover_images_original(path)
    chosen: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for image_path in paths:
        key = image_path.stem.lower()
        if key not in chosen:
            chosen[key] = image_path
            continue
        duplicates.setdefault(key, [chosen[key]]).append(image_path)

    for key, values in sorted(duplicates.items()):
        kept = chosen[key]
        ignored = [p for p in values if p != kept]
        print(
            f"WARNING: duplicate gaze image key {key!r}; keeping {kept} and ignoring "
            + ", ".join(str(p) for p in ignored)
        )

    return list(chosen.values())


def _process_one_with_failed_crop_artifact(image_path: Path, output_dir: Path, device: str, dwpose_dir: Path | None):
    record = impl._process_one_original(image_path, output_dir, device, dwpose_dir)
    if record.get("status") != "no_face":
        return record

    acquisition = record.get("face_acquisition") or {}
    retry_crop = acquisition.get("retry_crop") or {}
    bbox = retry_crop.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return record

    image = cv2.imread(image_path.as_posix())
    if image is None:
        return record
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return record
    x0 = max(0, min(x0, image.shape[1]))
    x1 = max(0, min(x1, image.shape[1]))
    y0 = max(0, min(y0, image.shape[0]))
    y1 = max(0, min(y1, image.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return record

    crop = image[y0:y1, x0:x1]
    path = output_dir / f"{image_path.stem}.gaze_retry_crop.png"
    if crop.size and cv2.imwrite(path.as_posix(), crop):
        acquisition["retry_crop_asset"] = path.as_posix()
        record["face_acquisition"] = acquisition
        impl._write_json(output_dir / f"{image_path.stem}.gaze.json", record)
        print(f"{image_path.stem}: saved failed retry crop -> {path}")
    return record


def main() -> int:
    # The gaze probe predates the historical DWPose adapter and resolves these
    # helpers from module globals at runtime. Reuse the shared compatibility
    # reader so both direct BODY18 caches and newer candidate-mapping caches work.
    impl._dwpose_target_points = target_points_from_profile_record

    # The base probe keys artifacts by image stem. Deduplicate the recursive
    # discovery result before --only filtering so duplicate basenames cannot
    # repeatedly overwrite the same output record.
    if not hasattr(impl, "_discover_images_original"):
        impl._discover_images_original = impl._discover_images
    impl._discover_images = _discover_unique_images

    # Preserve the failed DWPose-assisted crop for visual debugging when
    # MediaPipe still cannot acquire a face from it.
    if not hasattr(impl, "_process_one_original"):
        impl._process_one_original = impl._process_one
    impl._process_one = _process_one_with_failed_crop_artifact
    return impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
