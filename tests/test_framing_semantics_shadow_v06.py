from qwen_caption_validate import framing_semantics_shadow_v06 as v06


def test_person_geometry_binds_box_by_keypoint_coverage():
    boxes = [
        [0.0, 0.0, 0.3, 0.3],
        [0.0, 0.1, 1.0, 1.0],
    ]
    kp = [0.2, 0.2, 0.8, 0.9]
    person = v06._person_geometry(boxes, kp)
    assert person["status"] == "available"
    assert person["target_box_index"] == 1
    assert person["target_keypoint_coverage_fraction"] > 0.99
    assert person["crop_edges"]["left"] is True
    assert person["crop_edges"]["bottom"] is True


def test_head_shoulders_medium_from_relative_face_person_geometry():
    span = {"upper_anchor": "head", "lower_anchor": "shoulders"}
    face = {"status": "available", "height_fraction": 0.236}
    person = {"status": "available", "visible_height_fraction": 0.764}
    scale = v06._scale(span, face, person)
    assert scale["label"] == "medium"
    assert "face_person_height_ratio=0.309" in scale["basis"]


def test_head_shoulders_medium_close_from_relative_face_person_geometry():
    span = {"upper_anchor": "head", "lower_anchor": "shoulders"}
    face = {"status": "available", "height_fraction": 0.343}
    person = {"status": "available", "visible_height_fraction": 0.875}
    scale = v06._scale(span, face, person)
    assert scale["label"] == "medium_close_up"


def test_head_shoulders_ratio_abstention_band():
    span = {"upper_anchor": "head", "lower_anchor": "shoulders"}
    face = {"status": "available", "height_fraction": 0.296}
    person = {"status": "available", "visible_height_fraction": 0.800}
    scale = v06._scale(span, face, person)
    assert scale["status"] == "withheld"
    assert scale["label"] is None
    assert "abstention_band" in scale["basis"][0]


def test_head_only_reviewed_range_is_close_not_extreme_close():
    span = {"upper_anchor": "head", "lower_anchor": "head"}
    face = {"status": "available", "height_fraction": 0.698}
    person = {"status": "available", "visible_height_fraction": 0.985}
    scale = v06._scale(span, face, person)
    assert scale["label"] == "close_up"
    assert scale["composer_text"] == "close-up"


def test_head_only_below_close_calibration_is_withheld():
    span = {"upper_anchor": "head", "lower_anchor": "head"}
    face = {"status": "available", "height_fraction": 0.20}
    person = {"status": "available", "visible_height_fraction": 0.80}
    scale = v06._scale(span, face, person)
    assert scale["status"] == "withheld"
    assert scale["label"] is None


def test_non_close_family_span_does_not_get_overridden():
    span = {"upper_anchor": "head", "lower_anchor": "hips"}
    face = {"status": "available", "height_fraction": 0.30}
    person = {"status": "available", "visible_height_fraction": 0.80}
    assert v06._scale(span, face, person) is None
