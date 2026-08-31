from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import pose_atlas_v3_03 as atlas
from . import sam3d_hand_geometry as hand
from . import sam3d_relational_pose_profile_02 as v02


MHR = atlas.MHR_BODY


def _read_json(path: Path | None) -> dict[str, Any]:
    return v02._read_json(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    v02._write_json(path, value)


def _round(value: float | None, digits: int = 3) -> float | None:
    return v02._round(value, digits)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return v02._distance(a, b)


def _mean_hand_point(keypoints: np.ndarray, side: str) -> np.ndarray:
    indices = hand.mhr_hand_order(side)[1:]  # finger joints/tips, excluding wrist
    valid = [
        idx for idx in indices
        if idx < len(keypoints) and np.all(np.isfinite(keypoints[idx, :3]))
    ]
    if not valid:
        return np.full(3, np.nan, dtype=np.float64)
    return np.mean(keypoints[valid, :3], axis=0)


def _discovery_primitives(
    profile: dict[str, Any],
    keypoints: np.ndarray,
    sam2d: np.ndarray,
) -> dict[str, Any]:
    """Expose low-level geometry for discovery without assigning new semantics.

    These values are intentionally mechanical. The census may group recurring
    combinations for human review, but Fusion/Caption remains responsible for
    semantic actions/contact such as waving, holding, or resting on a surface.
    """
    if len(sam2d) <= max(MHR["left_shoulder"], MHR["right_shoulder"]):
        shoulder_span_px = 0.0
    else:
        shoulder_span_px = _distance(
            sam2d[MHR["left_shoulder"], :2],
            sam2d[MHR["right_shoulder"], :2],
        )

    shoulder_width_3d = _distance(
        keypoints[MHR["left_shoulder"]],
        keypoints[MHR["right_shoulder"]],
    )

    shoulder_mid_2d = (
        sam2d[MHR["left_shoulder"], :2] + sam2d[MHR["right_shoulder"], :2]
    ) / 2.0 if shoulder_span_px > 1e-9 else np.full(2, np.nan)

    hip_mid_2d = (
        sam2d[MHR["left_hip"], :2] + sam2d[MHR["right_hip"], :2]
    ) / 2.0 if len(sam2d) > max(MHR["left_hip"], MHR["right_hip"]) else np.full(2, np.nan)

    torso_mid_2d = (
        (shoulder_mid_2d + hip_mid_2d) / 2.0
        if np.all(np.isfinite(shoulder_mid_2d)) and np.all(np.isfinite(hip_mid_2d))
        else np.full(2, np.nan)
    )

    per_side: dict[str, Any] = {}
    for side in ("left", "right"):
        arm = (profile.get("arm_geometry") or {}).get(side) or {}
        h = (profile.get("hand_geometry") or {}).get(side) or {}
        dw = h.get("dwpose_hand") or {}

        shoulder_idx = MHR[f"{side}_shoulder"]
        elbow_idx = MHR[f"{side}_elbow"]
        wrist_idx = MHR[f"{side}_wrist"]

        wrist_above = None
        wrist_outward = None
        wrist_from_torso_x = None
        if (
            shoulder_span_px > 1e-9
            and max(shoulder_idx, elbow_idx, wrist_idx) < len(sam2d)
            and atlas._finite_xy(sam2d[shoulder_idx])
            and atlas._finite_xy(sam2d[wrist_idx])
        ):
            shoulder = sam2d[shoulder_idx, :2]
            wrist = sam2d[wrist_idx, :2]
            wrist_above = float((shoulder[1] - wrist[1]) / shoulder_span_px)

            # Sign is image-space only; do not treat it as anatomical direction.
            wrist_outward = float((wrist[0] - shoulder[0]) / shoulder_span_px)
            if np.all(np.isfinite(torso_mid_2d)):
                wrist_from_torso_x = float((wrist[0] - torso_mid_2d[0]) / shoulder_span_px)

        per_side[side] = {
            "elbow_flexion_deg": arm.get("elbow_flexion_deg"),
            "elbow_flexion_band": arm.get("elbow_flexion_band"),
            "arm_anchor_crop_support": arm.get("anchor_crop_support"),
            "hand_near_hip": bool((arm.get("geometry_flags") or {}).get("hand_near_hip")),
            "hand_near_face": bool((arm.get("geometry_flags") or {}).get("hand_near_face")),
            "hand_near_knee": bool((arm.get("geometry_flags") or {}).get("hand_near_knee")),
            "hand_shape": h.get("preferred_shape_label"),
            "hand_shape_source": h.get("preferred_shape_source"),
            "observed_hand_crop_support": dw.get("crop_support"),
            "observed_hand_crop_support_percent": dw.get("crop_support_percent"),
            "wrist_above_shoulder_shoulder_widths": _round(wrist_above),
            "wrist_x_from_shoulder_shoulder_widths": _round(wrist_outward),
            "wrist_x_from_torso_center_shoulder_widths": _round(wrist_from_torso_x),
        }

    left_hand = _mean_hand_point(keypoints, "left")
    right_hand = _mean_hand_point(keypoints, "right")
    bilateral: dict[str, Any] = {
        "hand_centroid_distance_shoulder_widths": None,
        "wrist_distance_shoulder_widths": None,
        "wrist_screen_distance_shoulder_widths": None,
    }
    if shoulder_width_3d > 1e-9:
        if np.all(np.isfinite(left_hand)) and np.all(np.isfinite(right_hand)):
            bilateral["hand_centroid_distance_shoulder_widths"] = _round(
                _distance(left_hand, right_hand) / shoulder_width_3d
            )
        lw, rw = MHR["left_wrist"], MHR["right_wrist"]
        if max(lw, rw) < len(keypoints):
            bilateral["wrist_distance_shoulder_widths"] = _round(
                _distance(keypoints[lw], keypoints[rw]) / shoulder_width_3d
            )

    if shoulder_span_px > 1e-9:
        lw, rw = MHR["left_wrist"], MHR["right_wrist"]
        if (
            max(lw, rw) < len(sam2d)
            and atlas._finite_xy(sam2d[lw])
            and atlas._finite_xy(sam2d[rw])
        ):
            bilateral["wrist_screen_distance_shoulder_widths"] = _round(
                _distance(sam2d[lw, :2], sam2d[rw, :2]) / shoulder_span_px
            )

    return {
        "authority": "mechanical_geometry_for_pose_library_discovery_only",
        "per_side": per_side,
        "bilateral": bilateral,
    }


def build_profile(
    arrays: dict[str, np.ndarray],
    dwpose: dict[str, Any] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    profile = v02.build_profile(arrays, dwpose, width, height)
    profile["schema_version"] = "sam3d-relational-pose-profile-0.3"

    relations = profile.get("relations") or {}

    # Do not turn a still-image geometry pattern into an action label. Keep
    # open/closed hand shape and raw wrist height as primitives for Fusion.
    relations.pop("raised_open_hand", None)
    relations.pop("waving_candidate", None)

    # A composite assertion cannot be more crop-supported than the weakest
    # required observed component. "Head supported by fist" therefore caps
    # the broad arm/head support by the observed fist/hand support.
    fist = relations.get("head_supported_by_fist") or {}
    if fist.get("geometry_match"):
        side = fist.get("side")
        head = relations.get("head_supported_by_hand") or {}
        hand_geometry = (profile.get("hand_geometry") or {}).get(str(side)) or {}
        hand_observed = hand_geometry.get("dwpose_hand") or {}
        head_support = float(head.get("crop_support") or 0.0)
        hand_support = float(hand_observed.get("crop_support") or 0.0)
        composite_support = min(head_support, hand_support)
        fist["crop_support"] = _round(composite_support, 4)
        fist["crop_support_percent"] = int(round(100.0 * composite_support))
        fist["support_components"] = {
            "head_hand_relation_crop_support": _round(head_support, 4),
            "observed_fist_crop_support": _round(hand_support, 4),
            "aggregation": "minimum_required_component",
        }
    relations["head_supported_by_fist"] = fist
    profile["relations"] = relations

    keypoints = np.asarray(
        arrays.get("pred_keypoints_3d", np.empty((0, 3))),
        dtype=np.float64,
    )
    if keypoints.ndim > 2:
        keypoints = keypoints.reshape((-1, keypoints.shape[-1]))
    keypoints = keypoints[:, :3]

    sam2d = atlas._sam2d_points(arrays)
    profile["discovery_primitives"] = _discovery_primitives(profile, keypoints, sam2d)

    policy = profile.get("policy") or {}
    policy.pop("waving_requires_vlm_confirmation", None)
    policy.update(
        {
            "action_semantics_are_not_emitted_by_geometry_profile": True,
            "open_closed_hand_and_wrist_height_are_fusion_primitives": True,
            "composite_relation_support_capped_by_weakest_required_component": True,
            "discovery_primitives_are_not_named_pose_claims": True,
        }
    )
    profile["policy"] = policy
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qwen-sam3d-relational-pose-profile-03",
        description=(
            "Build report-only SAM3D/DWPose pose profiles with hand geometry, "
            "crop support, and low-level discovery primitives."
        ),
    )
    parser.add_argument("sam3d_dir", type=Path)
    parser.add_argument("--dwpose-dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _hand_console_label(profile: dict[str, Any], side: str) -> str:
    h = (profile.get("hand_geometry") or {}).get(side) or {}
    source = h.get("preferred_shape_source")
    shape = h.get("preferred_shape_label") or "unavailable"
    dw = h.get("dwpose_hand") or {}
    if source == "dwpose_observed":
        return f"{side}:{shape}[observed,{dw.get('crop_support_percent') or 0}%]"
    return f"{side}:{shape}[sam3d,recon-only]"


def main() -> int:
    args = parse_args()
    sam3d_dir = args.sam3d_dir.expanduser().resolve()
    if not sam3d_dir.is_dir():
        raise SystemExit(f"SAM3D directory not found: {sam3d_dir}")

    dwpose_dir = (
        args.dwpose_dir.expanduser().resolve()
        if args.dwpose_dir
        else sam3d_dir.parent / "dwpose"
    )
    images_dir = (
        args.images_dir.expanduser().resolve()
        if args.images_dir
        else sam3d_dir.parent / "images"
    )
    output = (
        args.output or (sam3d_dir / "relational-pose-profile-v0.3")
    ).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in sam3d_dir.rglob("*.sam3d_arrays.npz") if p.is_file()
    )
    if args.include:
        wanted = [str(item).lower() for item in args.include]
        paths = [
            p for p in paths
            if any(token in p.stem.lower() for token in wanted)
        ]
    if not paths:
        raise SystemExit("No matching SAM3D arrays found")

    rows: list[dict[str, Any]] = []
    for path in paths:
        key = path.name.removesuffix(".sam3d_arrays.npz")
        with np.load(path) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}

        dwpose_path = dwpose_dir / f"{key}.dwpose.json"
        dwpose = _read_json(dwpose_path)

        width = int(dwpose.get("image_width") or 0)
        height = int(dwpose.get("image_height") or 0)
        if width <= 0 or height <= 0:
            image_matches = []
            if images_dir.is_dir():
                image_matches = [
                    p for p in images_dir.rglob(f"{key}.*") if p.is_file()
                ]
            if not image_matches:
                raise SystemExit(f"Cannot determine image size for {key}")
            from PIL import Image

            with Image.open(image_matches[0]) as im:
                width, height = im.size

        profile = build_profile(arrays, dwpose or None, width, height)
        record = {
            "image_key": key,
            "sam3d_arrays": str(path),
            "dwpose": str(dwpose_path) if dwpose_path.is_file() else None,
            "image_width": width,
            "image_height": height,
            "profile": profile,
        }
        out_path = output / f"{key}.sam3d_relational_pose.json"
        _write_json(out_path, record)
        rows.append(record)

        projected = profile["sam3d_projected_pose"]
        relations = profile["relations"]
        flags = []
        for name in (
            "hands_on_hips",
            "head_supported_by_hand",
            "head_supported_by_fist",
        ):
            value = relations.get(name) or {}
            if value.get("geometry_match"):
                label = name
                if value.get("side"):
                    label += f":{value['side']}"
                label += f"@{value.get('crop_support_percent') or 0}%"
                flags.append(label)

        hand_labels = ",".join(
            _hand_console_label(profile, side) for side in ("left", "right")
        )
        print(
            f"{key}: projected={projected['pose']} "
            f"recon={projected['reconstruction_match_percent']}% "
            f"crop={projected['crop_support_percent']}% "
            f"coverage={projected['crop_coverage_percent']}% "
            f"regions={','.join(projected['crop_supported_regions']) or '-'} "
            f"hands={hand_labels} relations={','.join(flags) or '-'}"
        )

    index = {
        "schema_version": "sam3d-relational-pose-profile-run-0.3",
        "sam3d_dir": str(sam3d_dir),
        "dwpose_dir": str(dwpose_dir),
        "record_count": len(rows),
        "records": rows,
    }
    index_path = output / "sam3d_relational_pose.index.json"
    _write_json(index_path, index)
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
