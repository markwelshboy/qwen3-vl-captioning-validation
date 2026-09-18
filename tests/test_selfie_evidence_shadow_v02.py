import numpy as np

from qwen_caption_validate import mesh_arm_occupancy_shadow_v01 as mesh
from qwen_caption_validate import selfie_evidence_shadow_v02 as selfie


def _dwpose(visible):
    return {
        "derived": {
            "target": {
                "visible_body_landmarks": list(visible),
            }
        }
    }


def test_focal_length_recovery_matches_cached_projection():
    width, height = 1000, 800
    focal = 1200.0
    cam_t = np.array([0.0, 0.0, 3.0])
    k3 = np.array([
        [-0.2, -0.1, 0.2],
        [0.3, 0.15, 0.4],
        [0.1, -0.25, 0.3],
    ])
    cam = k3 + cam_t
    k2 = np.column_stack([
        cam[:, 0] * focal / cam[:, 2] + width / 2,
        cam[:, 1] * focal / cam[:, 2] + height / 2,
    ])

    recovered, meta = mesh._recover_focal_length(k3, k2, cam_t, width, height)

    assert meta["status"] == "available"
    assert recovered is not None
    assert abs(recovered - focal) < 1e-6


def test_arm_observation_support_grades_complete_adjacent_and_single_joint():
    strong = mesh._arm_observation_support(
        _dwpose(["left_shoulder", "left_elbow", "left_wrist"]),
        "left",
    )
    moderate = mesh._arm_observation_support(
        _dwpose(["left_shoulder", "left_elbow"]),
        "left",
    )
    weak = mesh._arm_observation_support(
        _dwpose(["left_shoulder"]),
        "left",
    )

    assert strong["grade"] == "strong"
    assert moderate["grade"] == "moderate"
    assert weak["grade"] == "weak"


def test_large_mesh_area_needs_observation_support_for_strong_grade():
    residual = {"median_fraction_of_image_diagonal": 0.001}

    assert mesh._evidence_grade(
        0.12,
        {"grade": "moderate"},
        residual,
    ) == "strong"

    assert mesh._evidence_grade(
        0.12,
        {"grade": "weak"},
        residual,
    ) == "weak"

    assert mesh._evidence_grade(
        0.12,
        {"grade": "none"},
        residual,
    ) == "insufficient"


def test_neutral_gestalt_selfie_language_is_strong_when_unprompted():
    record = {
        "acquisition": {
            "expression_action": ["neutral expression"],
            "gestalt": "close-up selfie of a man indoors",
            "uncertainties": [],
        }
    }
    evidence = selfie._neutral_selfie_semantic(record)
    assert evidence["grade"] == "strong"
    assert evidence["matched_text"]


def test_selfie_decision_requires_two_independent_primary_families():
    portrait = {"eligible": True}
    shoulder = {
        "publishable_candidate": True,
        "composer_text": "the left shoulder is nearer the camera",
    }

    one_only = selfie._decision(
        {"grade": "strong"},
        {"grade": "none"},
        {"grade": "none"},
        shoulder,
        portrait,
    )
    assert one_only["publishable_selfie"] is False
    assert one_only["status"] == "candidate_not_publishable"

    semantic_plus_camera = selfie._decision(
        {"grade": "strong"},
        {"grade": "strong", "composer_text": "from a slightly elevated camera viewpoint"},
        {"grade": "none"},
        shoulder,
        portrait,
    )
    assert semantic_plus_camera["publishable_selfie"] is True
    assert semantic_plus_camera["status"] == "selfie_supported"

    camera_plus_arm = selfie._decision(
        {"grade": "none"},
        {"grade": "strong", "composer_text": "from a slightly elevated camera viewpoint"},
        {
            "grade": "strong",
            "composer_text": "an outstretched arm fills much of the lower frame-left foreground",
        },
        shoulder,
        portrait,
    )
    assert camera_plus_arm["publishable_selfie"] is True


def test_shoulder_is_supportive_but_never_primary():
    portrait = {"eligible": True}
    shoulder = {
        "publishable_candidate": True,
        "composer_text": "the left shoulder is nearer the camera",
    }
    result = selfie._decision(
        {"grade": "none"},
        {"grade": "none"},
        {"grade": "none"},
        shoulder,
        portrait,
    )
    assert result["publishable_selfie"] is False
    assert result["qualifying_primary_families"] == []
