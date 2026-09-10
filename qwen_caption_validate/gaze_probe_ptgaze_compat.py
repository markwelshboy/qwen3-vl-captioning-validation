from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import gaze_probe_ptgaze as impl
from .dwpose_compat import target_points_from_profile_record


_FACE_NAMES = ("nose", "right_eye", "left_eye", "right_ear", "left_ear")
_EXPECTED_HEAD_CENTER: np.ndarray | None = None


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


def _tight_dwpose_head_crop(record: dict, width: int, height: int):
    """Prefer a face-landmark-sized crop when DWPose has facial landmarks.

    The base probe originally let shoulder width dominate crop size. That is useful
    for locating an approximate head region, but can leave a small/downward-looking
    face occupying too little of the retry image for MediaPipe. When at least two
    facial landmarks are available, center on their median and size from their
    extent instead. Fall back to the original head/shoulder heuristic otherwise.
    """
    fallback = impl._dwpose_head_crop_original(record, width, height)
    points = target_points_from_profile_record(record, width, height)
    if len(points) < 18:
        return fallback

    values: list[tuple[str, np.ndarray]] = []
    for name in _FACE_NAMES:
        index = impl.BODY18_IDX[name]
        if index >= len(points):
            continue
        point = np.asarray(points[index, :2], dtype=np.float64)
        if (
            point.size >= 2
            and np.isfinite(point).all()
            and float(point[0]) >= 0.0
            and float(point[1]) >= 0.0
        ):
            values.append((name, point))

    if len(values) < 2:
        return fallback

    cloud = np.stack([point for _, point in values])
    center = np.median(cloud, axis=0)
    extent_x = float(np.ptp(cloud[:, 0]))
    extent_y = float(np.ptp(cloud[:, 1]))
    face_extent = max(extent_x, extent_y)
    if not np.isfinite(face_extent) or face_extent <= 1.0:
        return fallback

    # Make the face substantially larger than the former shoulder-driven crop,
    # while retaining enough context for MediaPipe's face mesh detector.
    side = int(round(max(96.0, face_extent * 2.25)))
    side = min(side, width, height, int(round(min(width, height) * 0.42)))
    if side < 48:
        return fallback

    x0 = int(round(float(center[0]) - side / 2.0))
    y0 = int(round(float(center[1]) - side / 2.0))
    x0 = max(0, min(x0, width - side))
    y0 = max(0, min(y0, height - side))
    x1 = x0 + side
    y1 = y0 + side

    return {
        "bbox_xyxy": [x0, y0, x1, y1],
        "center_xy": [float(center[0]), float(center[1])],
        "side_px": side,
        "evidence_landmarks": [name for name, _ in values],
        "strategy": "tight_face_landmarks",
        "face_landmark_extent_px": face_extent,
        "fallback_crop": fallback,
    }


def _dwpose_expected_head_center(image_path: Path, dwpose_dir: Path | None) -> np.ndarray | None:
    """Return the expected full-image head center from cached DWPose evidence."""
    if dwpose_dir is None:
        return None
    dwpose_path = impl._find_for_key(dwpose_dir, image_path.stem)
    if dwpose_path is None:
        return None
    image = cv2.imread(image_path.as_posix())
    if image is None:
        return None
    record = impl._read_json(dwpose_path)
    crop = _tight_dwpose_head_crop(record, image.shape[1], image.shape[0])
    if not crop:
        return None
    center = np.asarray(crop.get("center_xy", []), dtype=np.float64).reshape(-1)
    if center.size < 2 or not np.isfinite(center[:2]).all():
        return None
    return center[:2].copy()


def _bbox_score(bbox: np.ndarray) -> float:
    """Prefer the detected face nearest DWPose's head center when available.

    The base probe historically selected the largest MediaPipe face box. A rare
    false-positive can therefore beat the real face (for example a torso region).
    DWPose already tells us where the subject's head is, so use proximity when we
    have that evidence; otherwise preserve the original area-based behaviour.
    """
    if _EXPECTED_HEAD_CENTER is None:
        return impl._bbox_area_original(bbox)

    box = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if box.size < 4 or not np.isfinite(box[:4]).all():
        return float("-inf")
    center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)
    distance_sq = float(np.sum((center - _EXPECTED_HEAD_CENTER) ** 2))
    # Max() chooses the largest score. Area is only a tiny deterministic tie-break.
    return -distance_sq + impl._bbox_area_original(bbox) * 1e-6


def _process_one_with_failed_crop_artifact(image_path: Path, output_dir: Path, device: str, dwpose_dir: Path | None):
    global _EXPECTED_HEAD_CENTER

    expected = _dwpose_expected_head_center(image_path, dwpose_dir)
    _EXPECTED_HEAD_CENTER = expected
    try:
        record = impl._process_one_original(image_path, output_dir, device, dwpose_dir)
    finally:
        _EXPECTED_HEAD_CENTER = None

    acquisition = record.get("face_acquisition") or {}
    if expected is not None and int(acquisition.get("full_image_face_count") or 0) > 1:
        acquisition["selection_strategy"] = "nearest_dwpose_head_center"
        acquisition["expected_head_center_xy"] = [float(expected[0]), float(expected[1])]
        record["face_acquisition"] = acquisition
        impl._write_json(output_dir / f"{image_path.stem}.gaze.json", record)

    if record.get("status") != "no_face":
        return record

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

    # Prefer a face-focused DWPose retry crop when facial landmarks exist. Keep
    # the original crop heuristic available for fallback and audit metadata.
    if not hasattr(impl, "_dwpose_head_crop_original"):
        impl._dwpose_head_crop_original = impl._dwpose_head_crop
    impl._dwpose_head_crop = _tight_dwpose_head_crop

    # Preserve the base area scorer so multi-face selection can use DWPose head
    # proximity without changing behaviour when DWPose evidence is unavailable.
    if not hasattr(impl, "_bbox_area_original"):
        impl._bbox_area_original = impl._bbox_area
    impl._bbox_area = _bbox_score

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
