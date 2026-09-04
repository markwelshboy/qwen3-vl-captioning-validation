from __future__ import annotations

import json
import unittest

from qwen_caption_validate.runner import validate_analysis
from qwen_caption_validate.semantic_v3_gestalt import DEFAULT_SCHEMA
from qwen_caption_validate.semantic_v3_gestalt_source_authority import apply_source_authority


class SemanticV3GestaltSourceAuthorityTests(unittest.TestCase):
    def _gestalt(self) -> dict:
        return {
            "schema_version": "composition-gestalt-1.4",
            "camera": {
                "elevation": "eye_level",
                "pitch": "level",
                "confidence": 0.8,
                "evidence": ["centered face"],
                "counterevidence": [],
            },
            "capture": {
                "mode": "external_camera",
                "confidence": 0.7,
                "device_visibility": "not_visible",
                "evidence": ["no phone or mirror"],
            },
            "subject_orientation": {
                "body_orientation": "frontal",
                "body_faces_frame": "unknown",
                "body_confidence": 0.8,
                "torso_evidence_quality": "strong",
                "body_evidence": ["shoulders visible"],
                "body_counterevidence": [],
                "head_relative_body": "aligned",
                "head_confidence": 0.8,
                "head_evidence": ["face centered"],
            },
            "framing": {
                "shot_scale": "medium",
                "visible_extent": "upper body",
                "subject_frame_fill": "tight",
                "confidence": 0.9,
            },
            "environment": {
                "space": "indoor",
                "lighting_context": "unknown",
                "confidence": 0.8,
                "evidence": ["wall"],
                "counterevidence": [],
            },
            "background_regions": [],
            "support_context": [],
            "foreground_relations": [
                {
                    "type": "object_near_lens",
                    "description": "laptop in foreground",
                    "evidence_status": "observed",
                    "confidence": 0.9,
                }
            ],
            "salient_body_configuration": [],
            "composition_summary": "frontal medium shot with an eye-level external camera",
            "uncertainties": [],
        }

    def _evidence(self) -> dict:
        return {
            "projection_policy": "observation_only",
            "subject_evidence": {"orientation_cues": []},
            "entities": [{"class": "laptop"}],
            "relations": [],
            "composition_observations": [],
        }

    def test_withholds_priors_when_required_source_channels_are_absent(self) -> None:
        governed, audit = apply_source_authority(self._gestalt(), self._evidence())
        self.assertEqual(governed["camera"]["elevation"], "unknown")
        self.assertEqual(governed["camera"]["pitch"], "unknown")
        self.assertEqual(governed["capture"]["mode"], "unknown")
        self.assertEqual(governed["subject_orientation"]["body_orientation"], "unknown")
        self.assertEqual(governed["subject_orientation"]["head_relative_body"], "unknown")
        self.assertEqual(governed["foreground_relations"], [])
        self.assertIsNone(governed["composition_summary"])
        self.assertFalse(audit["authority_surface"]["orientation"])
        self.assertFalse(audit["authority_surface"]["camera_perspective"])
        self.assertFalse(audit["authority_surface"]["capture"])
        self.assertFalse(audit["authority_surface"]["near_lens"])

    def test_positive_authority_channels_preserve_claims(self) -> None:
        evidence = self._evidence()
        evidence["subject_evidence"]["orientation_cues"] = ["one shoulder nearer than the other"]
        evidence["composition_observations"] = [
            {"type": "camera_angle", "description": "low_angle perspective with upward view toward subject"},
            {"type": "object_near_lens", "description": "object very near lens"},
        ]
        evidence["entities"].append({"class": "camera"})
        governed, audit = apply_source_authority(self._gestalt(), evidence)
        self.assertEqual(governed["camera"]["elevation"], "eye_level")
        self.assertEqual(governed["capture"]["mode"], "external_camera")
        self.assertEqual(governed["subject_orientation"]["body_orientation"], "frontal")
        self.assertEqual(len(governed["foreground_relations"]), 1)
        self.assertTrue(audit["authority_surface"]["orientation"])
        self.assertTrue(audit["authority_surface"]["camera_perspective"])
        self.assertTrue(audit["authority_surface"]["capture"])
        self.assertTrue(audit["authority_surface"]["near_lens"])

    def test_authority_output_remains_gestalt_v14_schema_valid(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        governed, _ = apply_source_authority(self._gestalt(), self._evidence())
        self.assertEqual(validate_analysis(governed, schema), [])


if __name__ == "__main__":
    unittest.main()
