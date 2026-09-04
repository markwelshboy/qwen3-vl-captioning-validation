from __future__ import annotations

import json
import unittest

from qwen_caption_validate.semantic_v3_gestalt_v02_runtime import (
    DEFAULT_PROMPT_V02,
    OUTPUT_SUBDIR_V02,
    OUTPUT_VERSION_V02,
    build_gestalt_evidence_v02,
)


class SemanticV3GestaltV02Tests(unittest.TestCase):
    def _extract(self) -> dict:
        return {
            "schema_version": "visual-extract-3.0",
            "framing": {
                "shot_scale_candidate": "close_up",
                "visible_extent": "head_to_upper_torso",
                "frame_observations": ["head and upper torso visible"],
            },
            "target_subject": {
                "visible_body_parts": [
                    {
                        "part": "head",
                        "ownership_candidate": "target",
                        "visibility": "full",
                        "geometry_cues": ["forward-facing"],
                    }
                ],
                "geometry_landmark_visibility": {"left_hip": {"visibility": "not_visible"}},
                "orientation_cues": [],
                "gaze": {"target_candidate": "camera_lens", "confidence": 0.9},
                "interactions": [],
            },
            "entities": [{"id": "entity_01", "class": "wall"}],
            "relations": [],
            "scene": {
                "environment_candidate": "indoor",
                "background_regions": [{"description": "wall", "evidence_status": "hypothesis"}],
            },
            "composition_observations": [],
            "hypotheses": {
                "posture": {"value": "seated", "confidence": 0.9},
                "torso_orientation": {"orientation_band": "frontal", "confidence": 0.9},
                "camera": {"elevation": "eye_level", "pitch": "level", "confidence": 0.9},
                "capture": {"mode": "handheld_selfie", "confidence": 0.9},
                "support_context": [{"subject_relation": "seated_on", "target_description": "unknown_surface"}],
            },
            "uncertainties": ["lower body not visible"],
        }

    def test_projection_omits_all_extract_hypotheses(self) -> None:
        evidence = build_gestalt_evidence_v02(self._extract())
        encoded = json.dumps(evidence)
        self.assertNotIn("candidate_hypotheses", evidence)
        self.assertNotIn('"hypotheses"', encoded)
        self.assertNotIn("handheld_selfie", encoded)
        self.assertNotIn("unknown_surface", encoded)
        self.assertNotIn('"seated"', encoded)

    def test_projection_retains_observation_bearing_fields(self) -> None:
        evidence = build_gestalt_evidence_v02(self._extract())
        self.assertEqual(evidence["source_schema_version"], "visual-extract-3.0")
        self.assertEqual(evidence["framing"]["shot_scale_candidate"], "close_up")
        self.assertEqual(evidence["subject_evidence"]["orientation_cues"], [])
        self.assertEqual(evidence["scene"]["environment_candidate"], "indoor")
        self.assertIn("observation_only", evidence["projection_policy"])

    def test_prompt_has_hard_camera_support_foreground_and_lighting_gates(self) -> None:
        prompt = DEFAULT_PROMPT_V02.read_text(encoding="utf-8")
        self.assertIn("semantic hypotheses are intentionally NOT supplied", prompt)
        self.assertIn("camera.elevation and camera.pitch MUST be unknown", prompt)
        self.assertIn("body_orientation MUST be unknown", prompt)
        self.assertIn("support_context MUST be []", prompt)
        self.assertIn("depth_band=foreground object alone is insufficient", prompt)
        self.assertIn("Absence of a visible window or natural-light source does NOT establish artificial lighting", prompt)

    def test_v02_is_version_isolated(self) -> None:
        self.assertEqual(OUTPUT_VERSION_V02, "semantic-v3-gestalt-from-extract-0.2")
        self.assertEqual(OUTPUT_SUBDIR_V02, "gestalt-from-extract-v0.2")


if __name__ == "__main__":
    unittest.main()
