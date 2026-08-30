from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_caption_validate.composition_gestalt_probe_13 import _govern_output_file, _scrub_body_laterality


class CompositionGestaltProbe13Tests(unittest.TestCase):
    def test_scrubs_only_body_part_laterality(self) -> None:
        text = "right forearm across the lap with a window to the left of the person"
        clean, changed = _scrub_body_laterality(text)
        self.assertTrue(changed)
        self.assertEqual(clean, "forearm across the lap with a window to the left of the person")

    def test_governance_marks_ambiguous_support_ineligible(self) -> None:
        payload = {
            "gestalt": {
                "salient_body_configuration": [
                    {
                        "description": "right forearm braced on a surface",
                        "evidence_status": "observed",
                        "confidence": 0.9,
                    }
                ],
                "composition_summary": "close portrait with the right fist under the chin",
                "support_context": [
                    {
                        "subject_relation": "resting_on",
                        "target": "surface",
                        "target_description": "pale curved region",
                        "target_ownership": "unknown",
                        "evidence_status": "observed",
                        "confidence": 0.9,
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.composition_gestalt.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _govern_output_file(path)
            out = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            out["gestalt"]["salient_body_configuration"][0]["description"],
            "forearm braced on a surface",
        )
        self.assertEqual(
            out["gestalt"]["composition_summary"],
            "close portrait with the fist under the chin",
        )
        audit = out["governance_v13"]["support_context_audit"][0]
        self.assertFalse(audit["external_support_candidate"])

    def test_external_observed_support_can_remain_candidate(self) -> None:
        payload = {
            "gestalt": {
                "salient_body_configuration": [],
                "composition_summary": None,
                "support_context": [
                    {
                        "subject_relation": "resting_on",
                        "target": "table",
                        "target_description": "visible wooden tabletop edge",
                        "target_ownership": "external_scene",
                        "evidence_status": "observed",
                        "confidence": 0.95,
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.composition_gestalt.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _govern_output_file(path)
            out = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(out["governance_v13"]["support_context_audit"][0]["external_support_candidate"])


if __name__ == "__main__":
    unittest.main()
