from __future__ import annotations

import unittest

from qwen_caption_validate.caption_projection_134 import lint_caption


def _evidence(*, parts=None, required=None, allowed_postures=None) -> dict:
    return {
        "caption_policy": {"trigger_token": "BLIND7"},
        "pose_orientation": {
            "visible_subject_parts": parts or [],
            "qualified_interactions": [],
            "qualified_laterality": [],
            "qualified_hand_sides": [],
            "semantic_orientation": {},
            "whole_body_posture": {"allowed": allowed_postures or []},
        },
        "hard_constraints": {"visibility": {"visible": [], "partial": [], "not_visible": [], "unknown": []}},
        "required_claims": required or [],
        "required_scene_claims": [],
    }


class ComposeGovernance134Tests(unittest.TestCase):
    def test_scene_object_lies_does_not_count_as_subject_lying(self) -> None:
        evidence = _evidence(allowed_postures=["standing"])
        caption = "BLIND7 stands upright near a wall. A black backpack lies on the floor."
        result = lint_caption(caption, evidence)
        self.assertFalse(any(v.get("type") == "unsupported_whole_body_posture" and v.get("posture") == "lying" for v in result["violations"]))

    def test_negated_square_on_phrase_does_not_contradict_signed_torso(self) -> None:
        evidence = _evidence(required=[{
            "id": "signed_torso_depth_direction",
            "priority": "required",
            "nearer_anatomical_side": "left",
            "description": "the torso is angled in depth rather than square-on to the camera",
        }])
        caption = "BLIND7 has the torso angled in depth rather than square-on to the camera."
        result = lint_caption(caption, evidence)
        self.assertFalse(any(v.get("type") == "contradicts_signed_torso_depth" for v in result["violations"]))

    def test_qualified_side_does_not_license_new_shoulder_depth(self) -> None:
        evidence = _evidence(parts=[{
            "part": "shoulders",
            "anatomical_side": "left",
            "laterality_qualified": True,
            "geometry": None,
        }])
        caption = "BLIND7 faces the camera with the left shoulder closer to the camera."
        result = lint_caption(caption, evidence)
        self.assertTrue(any(v.get("type") == "unqualified_signed_shoulder_depth_relation" for v in result["violations"]))

    def test_explicit_shoulder_depth_geometry_authorizes_relation(self) -> None:
        evidence = _evidence(parts=[{
            "part": "left shoulder",
            "anatomical_side": "left",
            "laterality_qualified": True,
            "geometry": "closer to camera than the opposite shoulder",
        }])
        caption = "BLIND7 has the left shoulder closer to the camera than the right."
        result = lint_caption(caption, evidence)
        self.assertFalse(any(v.get("type") == "unqualified_signed_shoulder_depth_relation" for v in result["violations"]))

    def test_unlicensed_non_square_torso_is_rejected(self) -> None:
        evidence = _evidence(parts=[{
            "part": "shoulders",
            "anatomical_side": "left",
            "laterality_qualified": True,
            "geometry": None,
        }])
        caption = "BLIND7 has the torso angled in depth rather than square-on."
        result = lint_caption(caption, evidence)
        self.assertTrue(any(v.get("type") == "unqualified_torso_depth_relation" for v in result["violations"]))

    def test_support_relation_must_be_explicit(self) -> None:
        evidence = _evidence(required=[{
            "id": "support_relation_1",
            "priority": "required",
            "description": "left arm: hand supporting chin",
            "support_text": "hand supporting chin",
            "anatomical_side": "left",
        }])
        weak = lint_caption("BLIND7 rests the left hand under the chin.", evidence)
        self.assertTrue(any(w.get("claim_id") == "support_relation_1" for w in weak["warnings"]))
        strong = lint_caption("BLIND7 supports the chin with the left hand.", evidence)
        self.assertFalse(any(w.get("claim_id") == "support_relation_1" for w in strong["warnings"]))


if __name__ == "__main__":
    unittest.main()
