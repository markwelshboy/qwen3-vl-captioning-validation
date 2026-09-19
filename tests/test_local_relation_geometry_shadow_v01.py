from __future__ import annotations

from qwen_caption_validate import local_relation_geometry_shadow_v01 as mod


def _sheet(config):
    return {
        "image_key": "imageblind-01_00002",
        "facts": {
            "framing": {
                "broad_pose_supported": False,
                "composer_framing": {
                    "composer_text": "medium shot",
                },
            },
            "body": {
                "configuration": config,
            },
        },
    }


def test_relation_candidates_detect_head_proximity_without_promoting_it():
    candidates = mod._relation_candidates(
        _sheet(
            [
                {
                    "text": "hand near the face",
                    "composer_text": "hand near the face",
                    "promotion_status": "accepted_route_scoped_candidate",
                },
                {
                    "text": "forearm held beneath the chin/hand arrangement",
                    "composer_text": "forearm held beneath the chin/hand arrangement",
                },
            ]
        )
    )

    assert [x["relation_type"] for x in candidates] == [
        "hand_or_forearm_near_face_or_chin",
        "hand_or_forearm_near_face_or_chin",
    ]
    assert candidates[0]["surface_strength"] == "proximity"


def test_relation_candidates_preserve_source_text_from_laterality_binding():
    candidates = mod._relation_candidates(
        _sheet(
            [
                {
                    "text": "right forearm resting along the torso with hand near the hip",
                    "composer_text": "left hand resting on hip",
                    "laterality_binding": {
                        "source_text": "right forearm resting along the torso with hand near the hip",
                    },
                }
            ]
        )
    )

    assert len(candidates) == 1
    assert candidates[0]["relation_type"] == "hand_or_forearm_near_hip_or_waist"
    assert candidates[0]["source_text"].endswith("hand near the hip")
    assert candidates[0]["surface_strength"] == "proximity"


def test_relation_candidates_distinguish_explicit_contact_from_near():
    candidates = mod._relation_candidates(
        _sheet(
            [
                {
                    "text": "right hand on hip",
                    "composer_text": "right hand resting on hip",
                },
            ]
        )
    )

    assert candidates[0]["surface_strength"] == "contact"


def test_geometry_separates_near_face_from_far_wrist():
    points = {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (100.0, 0.0),
        "left_hip": (10.0, 150.0),
        "right_hip": (90.0, 150.0),
        "neck": (50.0, -20.0),
        "nose": (50.0, -60.0),
        "left_eye": (42.0, -65.0),
        "right_eye": (58.0, -65.0),
        "left_ear": (30.0, -60.0),
        "right_ear": (70.0, -60.0),
        "left_elbow": (20.0, -10.0),
        "right_elbow": (120.0, 80.0),
        "left_wrist": (45.0, -55.0),
        "right_wrist": (140.0, 140.0),
    }

    geometry = mod._geometry(points)

    assert geometry["nearest_wrist_to_face_side"] == "left"
    assert geometry["nearest_wrist_to_face_norm_body"] < 0.2
    assert geometry["sides"]["right"]["wrist_to_face_min_norm_body"] > 1.0


def test_geometry_reports_forearm_segment_distance_to_face():
    points = {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (100.0, 0.0),
        "left_hip": (10.0, 150.0),
        "right_hip": (90.0, 150.0),
        "neck": (50.0, -20.0),
        "nose": (50.0, -60.0),
        "left_eye": (42.0, -65.0),
        "right_eye": (58.0, -65.0),
        "left_ear": (30.0, -60.0),
        "right_ear": (70.0, -60.0),
        # Wrist is not especially close to the face, but the visible forearm
        # crosses immediately beneath it.
        "left_elbow": (20.0, -45.0),
        "left_wrist": (95.0, -45.0),
        "right_elbow": (120.0, 80.0),
        "right_wrist": (140.0, 140.0),
    }

    geometry = mod._geometry(points)

    left = geometry["sides"]["left"]
    assert left["forearm_segment_to_face_min_norm_body"] is not None
    assert left["forearm_segment_to_face_min_norm_body"] < left["wrist_to_face_min_norm_body"]
    assert geometry["nearest_upper_limb_to_face_side"] == "left"
    assert geometry["nearest_upper_limb_to_face_norm_body"] == left["upper_limb_to_face_min_norm_body"]


def test_point_to_segment_distance_clamps_to_visible_forearm():
    assert mod._point_to_segment_distance(
        (5.0, 2.0),
        (0.0, 0.0),
        (10.0, 0.0),
    ) == 2.0
    assert mod._point_to_segment_distance(
        (15.0, 0.0),
        (0.0, 0.0),
        (10.0, 0.0),
    ) == 5.0


def test_geometry_reports_hip_offsets_and_existing_production_binding():
    points = {
        "left_shoulder": (0.0, 0.0),
        "right_shoulder": (100.0, 0.0),
        "left_hip": (10.0, 150.0),
        "right_hip": (90.0, 150.0),
        # Make the left arm an unambiguous hand-on-hip geometry control:
        # wrist very near the left hip, bent elbow, and a clearly worse
        # opposite-side candidate. The production binder intentionally
        # withholds when the two side scores are too close.
        "left_elbow": (-30.0, 80.0),
        "right_elbow": (80.0, 80.0),
        "left_wrist": (12.0, 148.0),
        "right_wrist": (200.0, 150.0),
        "neck": (50.0, -20.0),
        "nose": (50.0, -60.0),
    }

    geometry = mod._geometry(points)

    assert geometry["sides"]["left"]["wrist_to_same_hip_norm_body"] < 0.05
    assert geometry["sides"]["left"]["wrist_minus_hip_dx_norm_body"] < 0.05
    assert geometry["production_hand_on_hip_binding"]["anatomical_side"] == "left"


def test_family_lookup_exposes_crop_rank_and_relative_field_of_view():
    registry = {
        "families": [
            {
                "family_id": "source-image-00002",
                "kind": "crop_family",
                "members": ["wide", "tight"],
                "wide_to_tight": ["wide", "tight"],
                "relative_fov_area": {
                    "wide": 1.25,
                    "tight": 1.0,
                },
            }
        ]
    }

    lookup = mod._family_lookup(registry)

    assert lookup["wide"]["crop_rank"] == 0
    assert lookup["tight"]["crop_rank"] == 1
    assert lookup["wide"]["relative_fov_area"] == 1.25
