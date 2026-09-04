from __future__ import annotations

import json
import unittest

from qwen_caption_validate.runner import validate_analysis
from qwen_caption_validate.semantic_v3_analyze import (
    DEFAULT_PROMPT,
    DEFAULT_SCHEMA,
    build_analyze_evidence,
    build_prompt,
)
from qwen_caption_validate.semantic_v3_analyze_normalize import normalize_analyze_representation


class SemanticV3AnalyzeTests(unittest.TestCase):
    def _extract(self) -> dict:
        return {
            "schema_version": "visual-extract-3.0",
            "framing": {
                "shot_scale_candidate": "close_up",
                "visible_extent": "head_to_upper_torso",
                "frame_observations": ["no visible lower body or legs"],
            },
            "target_subject": {
                "visible_body_parts": [
                    {
                        "part": "hand_fragment",
                        "ownership_candidate": "unknown",
                        "visibility": "fragment",
                        "connectivity_to_target_chain": "unknown",
                    }
                ],
                "geometry_landmark_visibility": {
                    "left_hip": {"visibility": "not_visible"},
                    "right_hip": {"visibility": "not_visible"},
                },
                "interactions": [
                    {
                        "type": "contact",
                        "actor_part": "hand_fragment",
                        "actor_ownership_candidate": "unknown",
                        "target_ref": "target_subject",
                        "evidence_status": "hypothesis",
                    }
                ],
                "gaze": {"target_candidate": "camera_lens"},
            },
            "entities": [{"id": "entity_01", "class": "wall"}],
            "relations": [],
            "scene": {
                "environment_candidate": "indoor",
                "environment_confidence": 0.9,
                "environment_cues": ["wall"],
            },
            "hypotheses": {
                "posture": {"value": "seated", "confidence": 0.65},
                "actions": [{"value": "posing", "confidence": 0.9}],
                "support_context": [{"subject_relation": "seated_on", "target_description": "unknown_surface"}],
                "torso_orientation": {"orientation_band": "frontal", "confidence": 0.9},
                "head_orientation": {"yaw": "frontal", "confidence": 0.9},
                "camera": {"elevation": "eye_level", "pitch": "level"},
                "capture": {"mode": "handheld_selfie"},
            },
            "uncertainties": ["fragment ownership unresolved"],
        }

    def _valid_analyze(self) -> dict:
        return {
            "schema_version": "semantic-analyze-3.0",
            "posture": {
                "value": "unknown",
                "confidence": 0.25,
                "assessment": "weak",
                "evidence": [],
                "limitations": ["lower body absent from crop"],
            },
            "actions": [
                {
                    "value": "posing",
                    "confidence": 0.6,
                    "evidence_status": "inferred",
                    "evidence": ["portrait-directed gaze and gesture candidate"],
                    "limitations": [],
                }
            ],
            "interactions": [
                {
                    "type": "contact",
                    "actor_part": "hand_fragment",
                    "actor_ownership": "unknown",
                    "target_ref": "target_subject",
                    "target_text": None,
                    "interpretation": "fragment contacts target",
                    "evidence_status": "inferred",
                    "confidence": 0.5,
                    "evidence": ["supplied contact candidate"],
                    "limitations": ["ownership unresolved"],
                }
            ],
            "ownership_assessments": [
                {
                    "part": "hand_fragment",
                    "ownership": "unknown",
                    "confidence": 0.8,
                    "evidence": ["Extract ownership unknown"],
                    "limitations": [],
                }
            ],
            "support_context": [],
            "physical_summary": "Tight crop with unresolved fragment contact; broad posture is not established.",
            "uncertainties": ["fragment ownership unresolved"],
        }

    def test_projection_selects_only_analyze_hypotheses(self) -> None:
        evidence = build_analyze_evidence(self._extract())
        hypotheses = evidence["candidate_hypotheses"]
        self.assertEqual(set(hypotheses), {"posture", "actions", "support_context"})
        encoded = json.dumps(evidence)
        self.assertNotIn("handheld_selfie", encoded)
        self.assertNotIn("eye_level", encoded)
        self.assertNotIn('"orientation_band": "frontal"', encoded)

    def test_projection_preserves_crop_and_ownership_limits(self) -> None:
        evidence = build_analyze_evidence(self._extract())
        self.assertIn("no visible lower body", evidence["framing_context"]["frame_observations"][0])
        part = evidence["subject_evidence"]["visible_body_parts"][0]
        self.assertEqual(part["ownership_candidate"], "unknown")
        self.assertEqual(part["connectivity_to_target_chain"], "unknown")
        self.assertIn("ownership unresolved", evidence["uncertainties"][0])

    def test_projection_omits_duplicate_visual_extraction_fields(self) -> None:
        evidence = build_analyze_evidence(self._extract())
        encoded = json.dumps(evidence)
        self.assertNotIn("transient_appearance", encoded)
        self.assertNotIn("illumination", encoded)
        self.assertNotIn("background_structure", encoded)
        self.assertNotIn("composition_observations", encoded)
        self.assertNotIn("image_overview", encoded)

    def test_prompt_contains_posture_fragment_and_support_guards(self) -> None:
        prompt = DEFAULT_PROMPT.read_text(encoding="utf-8")
        self.assertIn("YOU DO NOT SEE THE IMAGE", prompt)
        self.assertIn("No legs visible", prompt)
        self.assertIn("Do not complete a fragment into a whole limb", prompt)
        self.assertIn("Broad posture and exact support are separate", prompt)
        self.assertIn("laptop resting across thighs", prompt)
        self.assertIn("camera elevation", prompt)

    def test_normalizer_maps_hypothesis_evidence_status_only(self) -> None:
        value = self._valid_analyze()
        value["interactions"][0]["evidence_status"] = "hypothesis"
        normalized, audit = normalize_analyze_representation(value)
        self.assertEqual(normalized["interactions"][0]["evidence_status"], "inferred")
        self.assertEqual(audit["action_count"], 1)
        self.assertEqual(normalized["posture"], value["posture"])

    def test_valid_example_matches_schema(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(self._valid_analyze(), schema), [])

    def test_built_prompt_marks_extract_as_data(self) -> None:
        prompt = build_prompt("BASE", build_analyze_evidence(self._extract()))
        self.assertTrue(prompt.startswith("BASE"))
        self.assertIn("VISUAL EXTRACT EVIDENCE JSON", prompt)
        self.assertIn('"projection_policy"', prompt)


if __name__ == "__main__":
    unittest.main()
