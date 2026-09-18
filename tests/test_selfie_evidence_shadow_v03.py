import numpy as np

from qwen_caption_validate import mesh_arm_occupancy_shadow_v02 as arm_v02
from qwen_caption_validate import selfie_evidence_shadow_v03 as selfie_v03


def _raw_dwpose_with_hands(left_scores=0, right_scores=0):
    hands = np.zeros((2, 21, 2), dtype=float)
    scores = np.zeros((2, 21), dtype=float)

    for i in range(left_scores):
        hands[0, i] = [0.20 + i * 0.001, 0.80]
        scores[0, i] = 0.9

    for i in range(right_scores):
        hands[1, i] = [0.80 - i * 0.001, 0.80]
        scores[1, i] = 0.9

    return {
        "derived": {"target_person_index": 0},
        "raw_pose": {
            "body_scores": [[1.0] * 18],
            "hands": hands.tolist(),
            "hands_scores": scores.tolist(),
        },
    }


def test_hand_support_uses_easy_dwpose_left_then_right_block_order():
    dwpose = _raw_dwpose_with_hands(left_scores=10, right_scores=5)

    left = arm_v02._hand_support(dwpose, "left", 1000, 800)
    right = arm_v02._hand_support(dwpose, "right", 1000, 800)

    assert left["grade"] == "strong"
    assert left["confident_point_count"] == 10
    assert right["grade"] == "moderate"
    assert right["confident_point_count"] == 5


def test_complete_observed_chain_can_be_strong_without_missing_wrist():
    points = {
        "right_shoulder": (500.0, 260.0),
        "right_elbow": (320.0, 520.0),
        "right_wrist": (85.0, 755.0),
    }

    result = arm_v02._distal_arm_path_proxy(
        "right",
        points=points,
        hand_support={"grade": "none"},
        sam3d_projected_keypoints=None,
        width=1000,
        height=800,
    )

    assert result["wrist_observed"] is True
    assert result["distal_observed_joint"] == "wrist"
    assert result["distal_frame_region"] == "lower_frame_left"
    assert result["grade"] == "strong"


def test_missing_wrist_with_visible_same_side_hand_is_penalized():
    points = {
        "left_shoulder": (520.0, 260.0),
        "left_elbow": (250.0, 690.0),
        "left_wrist": None,
    }
    sam = np.full((70, 2), np.nan, dtype=float)
    sam[62] = [-80.0, 900.0]

    no_hand = arm_v02._distal_arm_path_proxy(
        "left",
        points=points,
        hand_support={"grade": "none"},
        sam3d_projected_keypoints=sam,
        width=1000,
        height=800,
    )
    visible_hand = arm_v02._distal_arm_path_proxy(
        "left",
        points=points,
        hand_support={"grade": "strong"},
        sam3d_projected_keypoints=sam,
        width=1000,
        height=800,
    )

    assert no_hand["grade"] in {"strong", "moderate"}
    assert visible_hand["hand_presence_conflicts_with_missing_wrist_crop_exit"] is True
    assert visible_hand["score"] <= no_hand["score"] - 2


def test_combined_arm_grade_requires_mesh_and_path_agreement_for_strong():
    assert arm_v02._combined_arm_grade("strong", "strong") == "strong"
    assert arm_v02._combined_arm_grade("moderate", "strong") == "strong"
    assert arm_v02._combined_arm_grade("strong", "weak") == "moderate"
    assert arm_v02._combined_arm_grade("moderate", "none") == "weak"
    assert arm_v02._combined_arm_grade("insufficient", "none") == "insufficient"


def test_v03_foreground_arm_uses_combined_grade_and_distal_region():
    mesh_record = {
        "status": "ok",
        "arms": {
            "left": {
                "combined_foreground_arm_grade": "weak",
                "visible_mesh_area_fraction": 0.03,
                "evidence_grade": "weak",
                "distal_arm_path_proxy": {
                    "grade": "weak",
                    "distal_frame_region": "lower_frame_right",
                },
            },
            "right": {
                "combined_foreground_arm_grade": "strong",
                "visible_mesh_area_fraction": 0.10,
                "occupancy_band": "large",
                "evidence_grade": "strong",
                "distal_arm_path_proxy": {
                    "grade": "strong",
                    "distal_frame_region": "lower_frame_left",
                    "complete_chain_straightness": 0.92,
                },
                "dwpose_hand_support": {"grade": "none"},
                "dwpose_observation_support": {"grade": "strong"},
            },
        },
    }

    evidence = selfie_v03._foreground_arm_evidence(mesh_record)

    assert evidence["grade"] == "strong"
    assert evidence["selected_arm"] == "right"
    assert evidence["frame_region"] == "lower_frame_left"
    assert "outstretched arm" in evidence["composer_text"]


def test_camera_plus_arm_can_publish_selfie_even_without_neutral_qwen_label():
    neutral = {"grade": "none"}
    camera = {
        "grade": "strong",
        "composer_text": "from a slightly elevated, downward-angled camera viewpoint",
    }
    arm = {
        "grade": "strong",
        "composer_text": "an outstretched arm extends into the lower-frame-left foreground",
    }
    shoulder = {
        "publishable_candidate": True,
        "composer_text": "the right shoulder is nearer the camera",
    }
    portrait = {"eligible": True}

    decision = selfie_v03._decision(neutral, camera, arm, shoulder, portrait)

    assert decision["publishable_selfie"] is True
    assert decision["status"] == "selfie_supported"
    assert decision["promoted_fact_candidates"][0] == "selfie-style capture"
    assert decision["selfie_semantic_authority"] == "multi_family_fused_evidence_shadow"
