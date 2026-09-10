from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qwen_caption_validate.face_target_binding import select_face_for_target


@dataclass
class _Face:
    bbox: np.ndarray
    confidence: float


def test_target_body_scale_can_reject_small_background_face_nearer_head() -> None:
    target = {
        "body_scale_px": 200.0,
        "shoulder_span_px": 300.0,
        "head_center_xy": [200.0, 200.0],
        "neck_xy": [220.0, 320.0],
        "body_bbox_xyxy": [50.0, 100.0, 450.0, 800.0],
    }
    background = _Face(np.array([160.0, 160.0, 220.0, 220.0]), 0.999)
    foreground = _Face(np.array([160.0, 130.0, 340.0, 310.0]), 0.995)

    selected = select_face_for_target([background, foreground], target)
    assert selected is not None
    face, strategy, diagnostics = selected

    assert face is foreground
    assert strategy == "dwpose_body_membership_head_scale"
    assert len(diagnostics) == 2
    assert diagnostics[0]["tiny_face_penalty"] > 0
    assert diagnostics[1]["tiny_face_penalty"] == 0


def test_without_target_geometry_falls_back_to_detector_confidence() -> None:
    a = _Face(np.array([0.0, 0.0, 50.0, 50.0]), 0.8)
    b = _Face(np.array([100.0, 100.0, 160.0, 160.0]), 0.95)

    selected = select_face_for_target([a, b], None)
    assert selected is not None
    face, strategy, diagnostics = selected

    assert face is b
    assert strategy == "highest_retinaface_score_no_dwpose_body"
    assert len(diagnostics) == 2
