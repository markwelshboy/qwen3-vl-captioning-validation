from __future__ import annotations

import unittest

from qwen_caption_validate.pose_semantics_10 import (
    _harden_probe_surface_gestures,
    _strict_reclining_support_evidence,
    _surface_support_economy,
)


class PoseSemantics10Tests(unittest.TestCase):
    def test_unrelated_scene_fabric_does_not_create_reclining_support(self) -> None:
        analysis = {
            "image_summary": "The subject bends forward while reaching toward a dark fabric item on the floor.",
            "target_subject": {
                "interactions": [],
                "visible_body_parts": [
                    {
                        "part": "torso",
                        "ownership": "target",
                        "support": "standing on both feet",
                        "contact": None,
                        "geometry": "bent forward at waist",
                        "confidence": 0.95,
                    }
                ],
            },
            "non_target_entities": [
                {
                    "description": "dark patterned fabric item on floor",
                    "contact": "being touched by left hand",
                    "support": "resting on floor",
                    "confidence": 0.90,
                }
            ],
        }
        self.assertEqual(_strict_reclining_support_evidence(analysis), [])

    def test_target_hand_resting_on_fabric_does_not_mean_target_body_is_supported(self) -> None:
        analysis = {
            "target_subject": {"interactions": [], "visible_body_parts": []},
            "non_target_entities": [
                {
                    "description": "dark patterned fabric draped over wooden table",
                    "contact": "held by target's left hand",
                    "support": "target's left hand resting on table",
                    "confidence": 0.95,
                }
            ],
        }
        self.assertEqual(_strict_reclining_support_evidence(analysis), [])

    def test_generic_surface_is_reclining_support_when_same_relation_names_bed_or_couch(self) -> None:
        analysis = {
            "target_subject": {
                "interactions": [
                    {
                        "type": "support",
                        "actor_part": "upper_torso",
                        "actor_ownership": "target",
                        "target": "gray surface",
                        "evidence_status": "observed",
                        "confidence": 0.95,
                        "notes": "torso resting on gray surface, likely couch or bed",
                    }
                ],
                "visible_body_parts": [],
            },
            "non_target_entities": [],
        }
        evidence = _strict_reclining_support_evidence(analysis)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["source"], "analysis.target_subject.interactions")

    def test_probe_only_surface_support_is_not_caption_usable(self) -> None:
        result = {
            "support_graph": {
                "body_support_edges": [
                    {
                        "actor_side": "right",
                        "actor_part": "hand",
                        "target_class": "surface",
                        "target": "surface",
                        "relation": "support",
                        "confidence": 0.90,
                        "authority": "pose_gestalt_observed_support",
                        "source": "pose_gestalt_v1.support_configuration",
                    }
                ],
                "support_chains": [],
            },
            "gestures": [
                {
                    "label": "right hand resting on the surface",
                    "caption_preferred": True,
                    "support": ["pose_gestalt_observed_support"],
                    "limitations": [],
                    "details": {
                        "class": "surface_support",
                        "actor_side": "right",
                        "part": "hand",
                        "surface": "surface",
                    },
                }
            ],
        }
        _harden_probe_surface_gestures(result)
        self.assertFalse(result["gestures"][0]["caption_preferred"])
        self.assertTrue(any("visibility does not independently verify" in value for value in result["gestures"][0]["limitations"]))

    def test_probe_surface_support_survives_matching_non_gestalt_relation(self) -> None:
        result = {
            "support_graph": {
                "body_support_edges": [
                    {
                        "actor_side": "left",
                        "actor_part": "forearm",
                        "target_class": "surface",
                        "target": "table",
                        "relation": "support",
                        "confidence": 0.95,
                        "authority": "pose_gestalt_observed_support",
                        "source": "pose_gestalt_v1.support_configuration",
                    },
                    {
                        "actor_side": "left",
                        "actor_part": "forearm",
                        "target_class": "surface",
                        "target": "table",
                        "relation": "support",
                        "confidence": 0.90,
                        "authority": "analyze_observed_body_part_support",
                        "source": "analysis.target_subject.visible_body_parts",
                    },
                ],
                "support_chains": [],
            },
            "gestures": [
                {
                    "label": "left forearm resting on the table",
                    "caption_preferred": True,
                    "support": ["pose_gestalt_observed_support"],
                    "limitations": [],
                    "details": {
                        "class": "surface_support",
                        "actor_side": "left",
                        "part": "forearm",
                        "surface": "table",
                    },
                }
            ],
        }
        _harden_probe_surface_gestures(result)
        self.assertTrue(result["gestures"][0]["caption_preferred"])
        self.assertIn("v10_independent_surface_support_corroboration", result["gestures"][0]["support"])

    def test_same_surface_component_supports_collapse_to_one_clause(self) -> None:
        def gesture(label: str, side: str, part: str, support: str) -> dict:
            return {
                "label": label,
                "caption_preferred": True,
                "support_score": 0.90,
                "support": [support],
                "limitations": [],
                "details": {
                    "class": "surface_support",
                    "actor_side": side,
                    "part": part,
                    "surface": "table",
                },
            }

        result = {
            "gestures": [
                gesture("left hand resting on the table", "left", "hand", "governed_fusion_body_surface_relation"),
                gesture("left arm resting on the table", "left", "arm", "governed_fusion_body_part_support"),
                gesture("right hand resting on the table", "right", "hand", "analyze_observed_body_surface_relation"),
                gesture("right arm resting on the table", "right", "arm", "analyze_observed_body_part_support"),
            ]
        }
        _surface_support_economy(result)
        preferred = [item for item in result["gestures"] if item.get("caption_preferred")]
        self.assertEqual(len(preferred), 1)
        self.assertEqual(preferred[0]["label"], "left arm resting on the table")


if __name__ == "__main__":
    unittest.main()
