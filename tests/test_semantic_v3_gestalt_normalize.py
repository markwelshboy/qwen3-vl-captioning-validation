from __future__ import annotations

import json
import unittest
from copy import deepcopy

from qwen_caption_validate.reprocess_semantic_v3_gestalt import reprocess_artifact
from qwen_caption_validate.semantic_v3_gestalt import DEFAULT_SCHEMA
from qwen_caption_validate.semantic_v3_gestalt_normalize import normalize_gestalt_representation
from qwen_caption_validate.semantic_v3_gestalt_runtime import (
    _validate_with_representation_normalization,
)


class SemanticV3GestaltNormalizeTests(unittest.TestCase):
    def _gestalt(self) -> dict:
        return {
            "schema_version": "composition-gestalt-1.4",
            "camera": {"elevation": "unknown", "pitch": "unknown", "confidence": 0.2, "evidence": [], "counterevidence": []},
            "capture": {"mode": "unknown", "confidence": 0.2, "device_visibility": "unknown", "evidence": []},
            "subject_orientation": {
                "body_orientation": "unknown",
                "body_faces_frame": "unknown",
                "body_confidence": 0.2,
                "torso_evidence_quality": "weak",
                "body_evidence": [],
                "body_counterevidence": [],
                "head_relative_body": "unknown",
                "head_confidence": 0.2,
                "head_evidence": [],
            },
            "framing": {"shot_scale": "close_up", "visible_extent": "head to upper torso", "subject_frame_fill": "tight", "confidence": 0.9},
            "environment": {"space": "indoor", "lighting_context": "unknown", "confidence": 0.7, "evidence": [], "counterevidence": []},
            "background_regions": [
                {"description": "dark background", "relation_to_subject": "behind_subject", "frame_location": "spanning", "evidence_status": "hypothesis", "confidence": 0.5}
            ],
            "support_context": [
                {"subject_relation": "seated_on", "target": "surface", "target_description": "unidentified support", "target_ownership": "unknown", "evidence_status": "hypothesis", "confidence": 0.3}
            ],
            "foreground_relations": [
                {"type": "foreground_occlusion", "description": "fragment near face", "evidence_status": "hypothesis", "confidence": 0.4}
            ],
            "salient_body_configuration": [
                {"description": "hand fragment near face", "evidence_status": "hypothesis", "confidence": 0.4}
            ],
            "composition_summary": "tight portrait with a fragment near the face",
            "uncertainties": ["exact support unknown"],
        }

    def test_hypothesis_maps_to_inferred_across_v14_status_fields(self) -> None:
        source = self._gestalt()
        normalized, audit = normalize_gestalt_representation(source)
        self.assertEqual(audit["action_count"], 4)
        self.assertEqual(normalized["background_regions"][0]["evidence_status"], "inferred")
        self.assertEqual(normalized["support_context"][0]["evidence_status"], "inferred")
        self.assertEqual(normalized["foreground_relations"][0]["evidence_status"], "inferred")
        self.assertEqual(normalized["salient_body_configuration"][0]["evidence_status"], "inferred")

    def test_normalizer_does_not_mutate_raw_model_payload(self) -> None:
        source = self._gestalt()
        before = deepcopy(source)
        normalize_gestalt_representation(source)
        self.assertEqual(source, before)
        self.assertEqual(source["support_context"][0]["evidence_status"], "hypothesis")

    def test_runtime_validation_accepts_canonicalized_representation(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(_validate_with_representation_normalization(self._gestalt(), schema), [])

    def test_zero_gpu_reprocessor_preserves_raw_and_writes_canonical_gestalt(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        model_output = self._gestalt()
        artifact = {
            "image_key": "control",
            "gestalt_model_output": deepcopy(model_output),
            "raw_response": "RAW JSON STRING",
            "schema_valid": False,
            "schema_errors": ["old error"],
        }
        updated, valid, actions = reprocess_artifact(artifact, schema)
        self.assertTrue(valid)
        self.assertEqual(actions, 4)
        self.assertEqual(updated["raw_response"], "RAW JSON STRING")
        self.assertEqual(updated["gestalt_model_output"], model_output)
        self.assertEqual(updated["gestalt_model_output"]["support_context"][0]["evidence_status"], "hypothesis")
        self.assertEqual(updated["gestalt"]["support_context"][0]["evidence_status"], "inferred")
        self.assertEqual(updated["representation_normalization"]["action_count"], 4)
        self.assertEqual(updated["schema_errors"], [])


if __name__ == "__main__":
    unittest.main()
