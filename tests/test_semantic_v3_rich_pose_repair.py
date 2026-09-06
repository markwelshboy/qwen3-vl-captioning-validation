from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_repair import (
    _expanded_authorized_laterality,
    _failure_lines,
    _repairable,
    build_repair_input,
    quality_audit,
)


class RichPoseRepairTests(unittest.TestCase):
    def test_governed_fist_authorizes_same_side_hand_only(self) -> None:
        pose = {
            "components": {
                "relations": [
                    {"side": "left", "phrase": "Head resting on the left fist."}
                ]
            }
        }
        authorized = _expanded_authorized_laterality(pose)
        self.assertIn(("left", "fist"), authorized)
        self.assertIn(("left", "hand"), authorized)
        self.assertNotIn(("left", "wrist"), authorized)
        self.assertNotIn(("right", "hand"), authorized)

    def test_left_hand_under_governed_left_fist_passes_laterality_gate(self) -> None:
        pose = {
            "caption_ready_phrases": ["Head resting on the left fist."],
            "components": {
                "relations": [
                    {"side": "left", "phrase": "Head resting on the left fist."}
                ]
            },
        }
        audit = quality_audit(
            "A woman rests her head on one hand.",
            "Her head rests on the left fist. Her left hand is raised under her jaw.",
            pose,
        )
        self.assertEqual(audit["unauthorized_anatomical_laterality"], [])
        self.assertNotIn("unsupported_anatomical_laterality", audit["warnings"])

    def test_right_hand_still_fails_under_left_fist_authority(self) -> None:
        pose = {
            "caption_ready_phrases": ["Head resting on the left fist."],
            "components": {
                "relations": [
                    {"side": "left", "phrase": "Head resting on the left fist."}
                ]
            },
        }
        audit = quality_audit(
            "A woman rests her head on one hand.",
            "Her head rests on the left fist while her right hand is raised.",
            pose,
        )
        self.assertIn("right hand", [v.lower() for v in audit["unauthorized_anatomical_laterality"]])
        self.assertIn("unsupported_anatomical_laterality", audit["warnings"])

    def test_generic_identity_evasion_is_repair_candidate(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        # Keep draft/output lengths comparable so this fixture isolates the semantic
        # identity-evasion gate rather than also tripping the independent expansion gate.
        audit = quality_audit(
            (
                "A man smiles toward the camera in a close portrait, with shoulder-length "
                "wavy hair framing his face and a dark shirt visible below."
            ),
            (
                "A man smiles toward the camera in a close portrait, with hair showing a "
                "natural texture and length that falls to the upper back above a dark shirt."
            ),
            pose,
        )
        self.assertIn("generic_identity_paraphrase", audit["warnings"])
        self.assertNotIn("edited_caption_expanded_substantially", audit["warnings"])
        self.assertTrue(_repairable(audit))
        lines = _failure_lines(audit)
        self.assertTrue(any("natural texture" in line.lower() for line in lines))

    def test_length_warning_is_not_auto_repairable(self) -> None:
        audit = {
            "warnings": ["edited_caption_overcompressed"],
            "identity_leaks": [],
            "generic_identity_paraphrase_leaks": [],
            "hair_dye_detail_leaks": [],
            "meta_redaction_language": [],
            "unauthorized_anatomical_laterality": [],
            "conflicting_pose_wording": [],
        }
        self.assertFalse(_repairable(audit))

    def test_repair_prompt_preserves_only_explicit_transient_hair_state(self) -> None:
        pose = {
            "caption_ready_phrases": ["Crouching with the torso partly turned sideways to the camera."],
            "components": {"relations": []},
        }
        editor_input = {
            "mandatory_redactions": {
                "transient_hair_mentions_to_preserve": [
                    "hair falls forward, partially obscuring her face"
                ]
            }
        }
        audit = {
            "warnings": ["generic_identity_paraphrase"],
            "identity_leaks": [],
            "generic_identity_paraphrase_leaks": ["natural texture"],
            "hair_dye_detail_leaks": [],
            "meta_redaction_language": [],
            "unauthorized_anatomical_laterality": [],
            "conflicting_pose_wording": [],
        }
        template = (
            "CAPTION={{CURRENT_CAPTION}}\nPOSE={{POSE_CORRECTIONS}}\n"
            "HAIR={{TRANSIENT_HAIR}}\nFAIL={{FAILURES}}"
        )
        built = build_repair_input(
            edited_caption="Her hair falls forward with a natural texture.",
            pose=pose,
            editor_input=editor_input,
            audit=audit,
            prompt_template=template,
        )
        self.assertIn("hair falls forward, partially obscuring her face", built["repair_prompt"])
        self.assertIn("natural texture", built["repair_prompt"])
        self.assertTrue(built["repairable"])


if __name__ == "__main__":
    unittest.main()
