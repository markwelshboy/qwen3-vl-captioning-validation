from __future__ import annotations

from qwen_caption_validate import local_configuration_semantics_shadow_v01 as mod


def _fragment(*relationships: str) -> dict:
    return {
        "image_key": "imageblind-01_00061",
        "policy_mode": "configuration",
        "extraction": {
            "pose_candidate": None,
            "body_relationships": [
                {"text": text, "authority": "visible_candidate"}
                for text in relationships
            ],
        },
    }


def _policy() -> dict:
    return {
        "image_key": "imageblind-01_00061",
        "policy": {"mode": "configuration"},
    }


def test_head_support_relations_collapse_to_one_semantic() -> None:
    fragment = _fragment(
        "fist under the chin with forearm beneath it",
        "forearm held across the torso",
        "hand supporting the chin/head",
    )
    points = {
        "left_shoulder": (10.0, 20.0),
        "left_elbow": None,
        "left_wrist": (20.0, 15.0),
        "right_shoulder": (30.0, 20.0),
        "right_elbow": (32.0, 30.0),
        "right_wrist": None,
    }
    out = mod.evaluate(fragment, _policy(), points=points)
    hs = out["head_support"]
    assert hs["status"] == "candidate"
    assert hs["uses_fist_semantics"] is True
    assert hs["canonical_side_neutral"] == "chin resting on a fist, with the forearm beneath/supporting the pose"
    assert hs["laterality_binding"]["status"] == "bound"
    assert hs["laterality_binding"]["anatomical_side"] == "left"
    assert hs["laterality_binding"]["forearm_side_publishable"] is False
    assert hs["composer_text"] == "chin resting on the left fist, with the forearm beneath/supporting the pose"


def test_forearm_side_requires_same_side_elbow_and_wrist() -> None:
    fragment = _fragment("chin resting on a fist with the forearm beneath it")
    points = {
        "left_shoulder": (10.0, 20.0),
        "left_elbow": (15.0, 25.0),
        "left_wrist": (20.0, 15.0),
        "right_shoulder": (30.0, 20.0),
        "right_elbow": None,
        "right_wrist": None,
    }
    out = mod.evaluate(fragment, _policy(), points=points)
    binding = out["head_support"]["laterality_binding"]
    assert binding["anatomical_side"] == "left"
    assert binding["forearm_side_publishable"] is True
    assert out["head_support"]["composer_text"] == "chin resting on the left fist, with the left forearm beneath/supporting the pose"


def test_qwen_laterality_is_stripped_but_retained_for_audit() -> None:
    fragment = _fragment(
        "right hand holding a white cup near the mouth",
        "left forearm resting on a surface with the hand partially visible",
        "left elbow bent and positioned near the edge of the surface",
    )
    out = mod.evaluate(fragment, _policy(), points={})
    assert out["side_neutral_relationships"] == [
        "hand holding a white cup near the mouth",
        "forearm resting on a surface with the hand partially visible",
        "elbow bent and positioned near the edge of the surface",
    ]
    assert len(out["laterality_leakage"]) == 3
    assert out["head_support"]["status"] == "not_present"


def test_two_visible_wrists_do_not_guess_head_support_side() -> None:
    fragment = _fragment("fist under the chin")
    points = {
        "left_shoulder": (10.0, 20.0),
        "left_elbow": (15.0, 25.0),
        "left_wrist": (20.0, 15.0),
        "right_shoulder": (30.0, 20.0),
        "right_elbow": (25.0, 25.0),
        "right_wrist": (22.0, 16.0),
    }
    out = mod.evaluate(fragment, _policy(), points=points)
    binding = out["head_support"]["laterality_binding"]
    assert binding["status"] == "unresolved"
    assert binding["anatomical_side"] is None


def test_no_head_support_relation_stays_not_present() -> None:
    fragment = _fragment("hand holding a cup near the mouth")
    out = mod.evaluate(fragment, _policy(), points={})
    assert out["head_support"]["status"] == "not_present"
    assert out["head_support"]["composer_text"] is None
