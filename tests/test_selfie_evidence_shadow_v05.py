from qwen_caption_validate import selfie_evidence_shadow_v05 as v05


def _arm(grade, region):
    return {
        "combined_foreground_arm_grade": grade,
        "frame_region": region,
        "visible_mesh_area_fraction": 0.08,
        "occupancy_band": "large",
        "evidence_grade": grade,
        "distal_arm_path_proxy": {
            "grade": grade,
            "distal_frame_region": region,
            "complete_chain_straightness": 0.9,
        },
        "dwpose_hand_support": {"grade": "none"},
        "dwpose_observation_support": {"grade": "strong"},
    }


def test_opposite_side_arm_remains_composition_fact_but_not_selfie_arm():
    mesh = {
        "status": "ok",
        "arms": {
            "left": _arm("strong", "lower_frame_right"),
            "right": _arm("insufficient", "lower_frame_left"),
        },
    }
    shoulder = {
        "status": "clear",
        "publishable_candidate": True,
        "anatomical_side_nearer": "right",
    }

    out = v05._foreground_arm_partition(mesh, shoulder)

    assert out["selfie_arm_evidence"]["grade"] == "none"
    assert out["selfie_arm_evidence"]["selected_arm"] is None
    assert len(out["composition_arm_candidates"]) == 1
    assert out["composition_arm_candidates"][0]["anatomical_side_internal"] == "left"
    assert out["composition_arm_candidates"][0]["frame_region"] == "lower_frame_right"


def test_same_side_nearer_shoulder_arm_can_count_as_selfie_evidence():
    mesh = {
        "status": "ok",
        "arms": {
            "left": _arm("weak", "lower_frame_right"),
            "right": _arm("strong", "lower_frame_left"),
        },
    }
    shoulder = {
        "status": "clear",
        "publishable_candidate": True,
        "anatomical_side_nearer": "right",
    }

    out = v05._foreground_arm_partition(mesh, shoulder)

    selfie_arm = out["selfie_arm_evidence"]
    assert selfie_arm["grade"] == "strong"
    assert selfie_arm["selected_arm"] == "right"
    assert selfie_arm["frame_region"] == "lower_frame_left"


def test_no_clear_shoulder_keeps_composition_fact_but_withholds_selfie_arm():
    mesh = {
        "status": "ok",
        "arms": {
            "left": _arm("strong", "lower_frame_right"),
            "right": _arm("insufficient", "lower_frame_left"),
        },
    }
    shoulder = {
        "status": "ambiguous",
        "publishable_candidate": False,
        "anatomical_side_nearer": None,
    }

    out = v05._foreground_arm_partition(mesh, shoulder)

    assert out["selfie_arm_evidence"]["grade"] == "none"
    assert len(out["composition_arm_candidates"]) == 1
