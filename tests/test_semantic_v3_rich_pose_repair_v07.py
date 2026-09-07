from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_repair import _repairable
from qwen_caption_validate.semantic_v3_rich_pose_repair_v07 import (
    _failure_lines,
    _primary_subject_age_proxy_leaks,
    quality_audit,
)


class RichPoseRepairV07Tests(unittest.TestCase):
    def test_fine_lines_are_repairable_primary_subject_age_proxy(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        text = (
            "A woman wears glasses and a dark top while smiling toward the camera, with visible fine lines "
            "around the eyes and forehead. A window and lamp remain visible behind her."
        )
        audit = quality_audit(text, text, pose)
        self.assertIn("age_proxy_identity_leakage", audit["warnings"])
        self.assertIn(
            "visible fine lines around the eyes and forehead",
            [value.lower() for value in audit["age_proxy_identity_leaks"]],
        )
        self.assertTrue(_repairable(audit))

    def test_crows_feet_and_facial_lines_are_both_detected(self) -> None:
        text = (
            "A woman smiles broadly, with visible crow’s feet at the corners of her eyes and natural facial lines."
        )
        leaks = [value.lower() for value in _primary_subject_age_proxy_leaks(text)]
        self.assertTrue(any("crow’s feet" in value for value in leaks), leaks)
        self.assertIn("natural facial lines", leaks)

    def test_plain_natural_skin_texture_is_not_an_age_proxy(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        text = (
            "A close portrait shows natural skin texture under soft window light while the subject wears glasses."
        )
        audit = quality_audit(text, text, pose)
        self.assertNotIn("age_proxy_identity_leakage", audit["warnings"])
        self.assertEqual(audit["age_proxy_identity_leaks"], [])

    def test_background_person_crows_feet_are_scene_content(self) -> None:
        text = (
            "A woman sits at a cafe table. In the background, another woman with visible crow's feet smiles near a window."
        )
        self.assertEqual(_primary_subject_age_proxy_leaks(text), [])

    def test_depicted_portrait_fine_lines_are_scene_content(self) -> None:
        text = (
            "A man stands beside a framed portrait showing an older face with fine lines around the eyes and forehead."
        )
        self.assertEqual(_primary_subject_age_proxy_leaks(text), [])

    def test_failure_lines_name_age_proxy_without_broad_skin_instruction(self) -> None:
        audit = {
            "warnings": ["age_proxy_identity_leakage"],
            "age_proxy_identity_leaks": ["fine lines around the eyes"],
            "awkward_haircut_residue": [],
            "identity_leaks": [],
            "generic_identity_paraphrase_leaks": [],
            "hair_dye_detail_leaks": [],
            "meta_redaction_language": [],
            "unauthorized_anatomical_laterality": [],
            "conflicting_pose_wording": [],
        }
        lines = _failure_lines(audit)
        joined = "\n".join(lines).lower()
        self.assertIn("fine lines around the eyes", joined)
        self.assertIn("apparent-age proxy", joined)
        self.assertNotIn("skin texture", joined)


if __name__ == "__main__":
    unittest.main()
