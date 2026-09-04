from __future__ import annotations

import json
import unittest

from qwen_caption_validate.semantic_v3_gestalt import (
    DEFAULT_PROMPT,
    DEFAULT_SCHEMA,
    build_gestalt_evidence,
    build_prompt,
    govern_gestalt,
)
from qwen_caption_validate.runner import validate_analysis


class SemanticV3GestaltTests(unittest.TestCase):
    def _extract(self) -> dict:
        return {
            "schema_version": "visual-extract-3.0",
            "image_overview": "must not be transported into Gestalt evidence",
            "framing": {"shot_scale": "close_up", "visible_extent": "head_to_upper_torso"},
            "target_subject": {
                "transient_appearance": {"clothing": ["black shirt"]},
                "visible_body_parts": [{"part": "right_hand", "visibility": "partial", "ownership": "unknown"}],
                "geometry_landmark_visibility": {"hips": "not_visible"},
                "orientation_cues": ["one shoulder appears deeper than the other"],
                "gaze": {"direction": "toward_camera"},
                "interactions": [{"actor_part": "hand_fragment", "ownership": "unknown", "relation": "touching_neck"}],
            },
            "entities": [{"id": "entity_01", "category": "wall"}],
            "relations": [{"subject": "target", "relation": "in_front_of", "object": "entity_01"}],
            "scene": {
                "environment_candidate": {"value": "indoor", "confidence": 0.9},
                "background_regions": [{"description": "plain wall", "frame_location": "spanning"}],
            },
            "composition_observations": [{"type": "foreground_occlusion", "description": "human fragment crosses lower face"}],
            "hypotheses": {
                "posture": {"value": "seated", "confidence": 0.35},
                "torso_orientation": {"value": "three_quarter", "confidence": 0.65},
                "head_orientation": {"value": "toward_camera", "confidence": 0.9},
                "head_body_relation": {"value": "turned_toward_camera", "confidence": 0.65},
                "camera": {"elevation": "eye_level", "confidence": 0.35},
                "capture": {"mode": "handheld_selfie", "confidence": 0.35},
                "support_context": [],
            },
            "uncertainties": ["foreground human-fragment ownership unresolved"],
        }

    def _gestalt(self) -> dict:
        return {
            "schema_version": "composition-gestalt-1.4",
            "camera": {"elevation": "unknown", "pitch": "unknown", "confidence": 0.2, "evidence": [], "counterevidence": []},
            "capture": {"mode": "unknown", "confidence": 0.2, "device_visibility": "not_visible", "evidence": []},
            "subject_orientation": {
                "body_orientation": "three_quarter",
                "body_faces_frame": "left",
                "body_confidence": 0.65,
                "torso_evidence_quality": "moderate",
                "body_evidence": ["shoulder depth cue"],
                "body_counterevidence": [],
                "head_relative_body": "turned_toward_camera",
                "head_confidence": 0.65,
                "head_evidence": ["head candidate differs from torso candidate"],
            },
            "framing": {"shot_scale": "close_up", "visible_extent": "head to upper torso", "subject_frame_fill": "tight", "confidence": 0.9},
            "environment": {"space": "indoor", "lighting_context": "unknown", "confidence": 0.8, "evidence": ["plain wall background"], "counterevidence": []},
            "background_regions": [],
            "support_context": [
                {
                    "subject_relation": "seated_on",
                    "target": "surface",
                    "target_description": "unidentified pale surface",
                    "target_ownership": "unknown",
                    "evidence_status": "inferred",
                    "confidence": 0.3,
                }
            ],
            "foreground_relations": [],
            "salient_body_configuration": [
                {"description": "right hand near face", "evidence_status": "observed", "confidence": 0.8}
            ],
            "composition_summary": "right hand near the face in a tight portrait",
            "uncertainties": [],
        }

    def test_projection_uses_canonical_extract_fields_only(self) -> None:
        evidence = build_gestalt_evidence(self._extract())
        self.assertEqual(evidence["source_schema_version"], "visual-extract-3.0")
        self.assertIn("framing", evidence)
        self.assertIn("subject_evidence", evidence)
        self.assertIn("candidate_hypotheses", evidence)
        self.assertNotIn("image_overview", evidence)
        self.assertNotIn("transient_appearance", json.dumps(evidence))
        self.assertNotIn("raw_response", json.dumps(evidence))

    def test_projection_preserves_fragment_ownership_uncertainty(self) -> None:
        evidence = build_gestalt_evidence(self._extract())
        interaction = evidence["subject_evidence"]["interactions"][0]
        self.assertEqual(interaction["actor_part"], "hand_fragment")
        self.assertEqual(interaction["ownership"], "unknown")
        self.assertIn("ownership unresolved", evidence["uncertainties"][0])

    def test_prompt_explicitly_disables_image_and_demotes_hypotheses(self) -> None:
        prompt = DEFAULT_PROMPT.read_text(encoding="utf-8")
        self.assertIn("YOU DO NOT SEE THE IMAGE", prompt)
        self.assertIn("Extract hypotheses are CANDIDATES, not ground truth", prompt)
        self.assertIn("Missing evidence is not negative evidence", prompt)
        self.assertIn("Anatomical laterality is NOT Gestalt authority", prompt)
        self.assertIn("Broad posture and exact support are separate", prompt)
        self.assertIn("Do not infer handheld_selfie solely from direct gaze", prompt)

    def test_built_prompt_contains_projection_as_data(self) -> None:
        prompt = build_prompt("BASE", build_gestalt_evidence(self._extract()))
        self.assertTrue(prompt.startswith("BASE"))
        self.assertIn("VISUAL EXTRACT EVIDENCE JSON", prompt)
        self.assertIn('"source_schema_version":"visual-extract-3.0"', prompt)

    def test_governance_scrubs_anatomical_laterality_from_free_text(self) -> None:
        governed, audit = govern_gestalt(self._gestalt())
        self.assertEqual(governed["salient_body_configuration"][0]["description"], "hand near face")
        self.assertEqual(governed["composition_summary"], "hand near the face in a tight portrait")
        self.assertEqual(len(audit["body_laterality_scrubbed"]), 2)
        self.assertEqual(governed["subject_orientation"]["body_faces_frame"], "left")

    def test_governance_does_not_promote_unknown_support(self) -> None:
        governed, audit = govern_gestalt(self._gestalt())
        self.assertEqual(governed["support_context"][0]["target_ownership"], "unknown")
        row = audit["support_context_audit"][0]
        self.assertFalse(row["external_support_candidate"])

    def test_governed_payload_remains_v14_schema_valid(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        governed, _ = govern_gestalt(self._gestalt())
        self.assertEqual(validate_analysis(governed, schema), [])


if __name__ == "__main__":
    unittest.main()
