from __future__ import annotations

import json
import unittest

from qwen_caption_validate.extract_v3 import DEFAULT_PROMPT, DEFAULT_SCHEMA
from qwen_caption_validate.extract_v3_contract import audit_extract_contract
from qwen_caption_validate.runner import validate_analysis


class ExtractV3Tests(unittest.TestCase):
    def _sample(self) -> dict:
        landmark = {"visibility": "visible", "confidence": 0.9, "evidence": "region represented"}
        return {
            "schema_version": "visual-extract-3.0",
            "image_overview": "Person in a room with a red car visible through a window.",
            "framing": {
                "shot_scale_candidate": "medium_close_up",
                "visible_extent": "head through upper torso",
                "subject_frame_coverage": "large",
                "frame_observations": ["lower torso exits bottom crop"],
            },
            "target_subject": {
                "entity_ref": "target_subject",
                "transient_appearance": {
                    "clothing": [{"id": "appearance_01", "category": "shirt", "descriptors": ["dark"], "frame_location": "center", "visibility": "partial", "confidence": 0.9}],
                    "accessories": [{"id": "appearance_02", "category": "watch_strap", "descriptors": ["white"], "frame_location": "lower center", "visibility": "partial", "confidence": 0.9}],
                    "hair_state": [],
                    "expression_state": ["slight smile"],
                },
                "visible_body_parts": [],
                "geometry_landmark_visibility": {name: dict(landmark) for name in ["head", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]},
                "orientation_cues": {"torso": ["shoulders depth-staggered"], "head": ["face more camera-facing than torso"], "image_plane_body_axis": ["torso axis near upright"]},
                "gaze": {"target_candidate": "camera_lens", "image_direction": "image_center", "confidence": 0.8, "cues": ["eyes directed near lens"]},
                "interactions": [],
            },
            "entities": [
                {"id": "entity_01", "type": "vehicle", "class": "car", "descriptors": ["red", "blurred"], "visibility": "blurred", "frame_location": "background right", "depth_band": "through_opening", "confidence": 0.9},
                {"id": "entity_02", "type": "architecture", "class": "window", "descriptors": [], "visibility": "partial", "frame_location": "right", "depth_band": "background", "confidence": 0.95},
            ],
            "relations": [{"subject_ref": "entity_01", "predicate": "visible_through", "object_ref": "entity_02", "object_text": None, "evidence_status": "observed", "confidence": 0.9, "cues": ["window frame surrounds car"]}],
            "scene": {
                "environment_candidate": "indoor", "environment_confidence": 0.9,
                "environment_cues": ["interior wall and window"], "environment_counterevidence": [],
                "illumination": {"type": "mixed", "directionality": "mixed", "contrast": "medium", "observations": []},
                "background_structure": {"texture_complexity": "low", "structural_complexity": "medium", "specular_reflective": "low", "repeated_geometry": False, "strong_lines_or_angles": "medium", "reflections_present": False, "observations": ["window frame"]},
                "background_regions": [{"description": "window and exterior view", "relation_to_subject": "behind_subject", "frame_location": "right", "evidence_status": "observed", "confidence": 0.95}],
                "nuisance_regions": [],
            },
            "composition_observations": {"subject_dominance": "dominant", "foreground_relations": [], "visual_thrust_cues": ["subject fills center of frame"]},
            "hypotheses": {
                "posture": {"value": "unknown", "confidence": 0.3, "cues": [], "limitations": ["lower body cropped"]},
                "torso_orientation": {"orientation_band": "three_quarter", "body_faces_frame": "left", "confidence": 0.8, "cues": ["shoulder depth asymmetry"], "limitations": []},
                "head_orientation": {"yaw": "frontal", "pitch": "neutral", "roll": "neutral", "confidence": 0.8, "cues": ["face near camera-facing"], "limitations": []},
                "head_body_relation": {"value": "turned_toward_camera", "confidence": 0.8, "cues": ["face more frontal than torso"], "limitations": []},
                "camera": {"elevation": "unknown", "pitch": "unknown", "confidence": 0.3, "cues": [], "counterevidence": ["height relationship ambiguous"]},
                "capture": {"mode": "external_camera", "confidence": 0.6, "cues": []},
                "support_context": [],
                "actions": [],
            },
            "uncertainties": ["lower-body posture unresolved"],
        }

    def test_schema_accepts_representative_extract(self) -> None:
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(validate_analysis(self._sample(), schema), [])

    def test_extract_satisfies_analyze_and_gestalt_contracts(self) -> None:
        audit = audit_extract_contract(self._sample())
        self.assertTrue(audit["analyze_reconstructable"])
        self.assertTrue(audit["gestalt_reconstructable"])
        self.assertEqual(audit["analyze_missing_paths"], [])
        self.assertEqual(audit["gestalt_missing_paths"], [])

    def test_missing_hypotheses_are_reported(self) -> None:
        sample = self._sample()
        del sample["hypotheses"]["camera"]
        audit = audit_extract_contract(sample)
        self.assertTrue(audit["analyze_reconstructable"] is False)
        self.assertTrue(audit["gestalt_reconstructable"] is False)
        self.assertIn("hypotheses.camera", audit["analyze_missing_paths"])
        self.assertIn("hypotheses.camera", audit["gestalt_missing_paths"])

    def test_prompt_encodes_observe_once_and_specificity_persistence(self) -> None:
        prompt = DEFAULT_PROMPT.read_text(encoding="utf-8")
        self.assertIn("OBSERVE ONCE, REASON MANY TIMES", prompt)
        self.assertIn("Preserve specificity", prompt)
        self.assertIn("Separate OBSERVATIONS from HYPOTHESES", prompt)
        self.assertIn("Later Analyze, Gestalt", prompt)
        self.assertIn("WITHOUT seeing the image again", prompt)


if __name__ == "__main__":
    unittest.main()
