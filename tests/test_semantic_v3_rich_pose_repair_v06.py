from __future__ import annotations

import unittest

from qwen_caption_validate.semantic_v3_rich_pose_repair import _repairable
from qwen_caption_validate.semantic_v3_rich_pose_repair_v06 import (
    _awkward_haircut_residues,
    _failure_lines,
    quality_audit,
)


class RichPoseRepairV06Tests(unittest.TestCase):
    def test_hair_styled_in_a_cut_is_repair_candidate(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        draft = (
            "A woman sits in an airplane cabin wearing glasses and a patterned top. Loose strands frame her face "
            "and sweep to the side. Rows of seats and overhead bins are visible behind her."
        )
        edited = (
            "A woman with hair styled in a cut, with loose strands framing her face and swept slightly to the side, "
            "sits in an airplane cabin wearing glasses and a patterned top. Rows of seats and overhead bins are "
            "visible behind her."
        )
        audit = quality_audit(draft, edited, pose)
        self.assertIn("hair styled in a cut", [value.lower() for value in audit["awkward_haircut_residue"]])
        self.assertIn("awkward_haircut_residue", audit["warnings"])
        self.assertTrue(_repairable(audit))

    def test_hair_styled_in_a_cut_featuring_is_detected(self) -> None:
        text = (
            "A woman with tousled hair styled in a cut featuring loose strands framing her face smiles at the camera."
        )
        self.assertEqual(
            [value.lower() for value in _awkward_haircut_residues(text)],
            ["hair styled in a cut featuring"],
        )

    def test_failure_lines_name_the_local_residue(self) -> None:
        audit = {
            "warnings": ["awkward_haircut_residue"],
            "awkward_haircut_residue": ["hair styled in a cut"],
            "identity_leaks": [],
            "generic_identity_paraphrase_leaks": [],
            "hair_dye_detail_leaks": [],
            "meta_redaction_language": [],
            "unauthorized_anatomical_laterality": [],
            "conflicting_pose_wording": [],
        }
        lines = _failure_lines(audit)
        self.assertTrue(any("hair styled in a cut" in line for line in lines))

    def test_clean_transient_hair_clause_still_passes(self) -> None:
        pose = {"caption_ready_phrases": [], "components": {"relations": []}}
        text = "A woman with loose strands framing her face and swept slightly to the side wears glasses."
        audit = quality_audit(text, text, pose)
        self.assertNotIn("awkward_haircut_residue", audit["warnings"])


if __name__ == "__main__":
    unittest.main()
