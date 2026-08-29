from __future__ import annotations

import unittest

from qwen_caption_validate.laterality_paired_interaction_guard_237 import (
    guard_cross_field_paired_distinct_interactions,
)


def _dwpose(*, complete: bool = True) -> dict:
    return {
        "derived": {
            "target": {
                "connectivity": {
                    "left_arm": {"complete": complete, "visible_count": 3 if complete else 2},
                    "right_arm": {"complete": complete, "visible_count": 3 if complete else 2},
                }
            }
        }
    }


def _state(source_side: str, qualified_side: str | None, *, interaction: bool = False) -> dict:
    if interaction:
        return {
            "qualified_actor_ownership": "target",
            "selection_usable": True,
            "source_actor_anatomical_side": source_side,
            "qualified_actor_anatomical_side": qualified_side or "unknown",
            "laterality_selection_usable": qualified_side in {"left", "right"},
            "laterality_authority": "dwpose_sam_correlated" if qualified_side else "unresolved_interaction_entity",
            "laterality_reasons": [],
        }
    return {
        "qualified_ownership": "target",
        "selection_usable": True,
        "source_anatomical_side": source_side,
        "qualified_anatomical_side": qualified_side or "unknown",
        "laterality_selection_usable": qualified_side in {"left", "right"},
        "laterality_authority": "dwpose_sam_correlated" if qualified_side else "unresolved_entity_association",
        "laterality_reasons": [],
    }


def _part(source_side: str, target: str, qualified_side: str | None) -> dict:
    return {
        "part": f"{qualified_side} arm" if qualified_side else "arm",
        "source_part": f"{source_side}_arm",
        "anatomical_side": source_side,
        "ownership": "target",
        "visible_subparts": ["shoulder", "elbow", "forearm", "hand"],
        "contact": f"touching {target}",
        "support": f"resting on {target}" if target == "table" else None,
        "confidence": 0.95,
        "fusion_v2": _state(source_side, qualified_side),
    }


def _interaction(
    source_side: str,
    target: str,
    qualified_side: str | None,
    *,
    kind: str = "contact",
    restate_side: bool = False,
) -> dict:
    notes = (
        f"{source_side} hand touching {target}"
        if restate_side
        else f"hand touching {target}"
    )
    return {
        "type": kind,
        "actor_part": f"{qualified_side} hand" if qualified_side else "hand",
        "source_actor_part": f"{source_side} hand",
        "actor_ownership": "target",
        "target": target,
        "evidence_status": "observed",
        "confidence": 0.95,
        "notes": notes,
        "fusion_v2": _state(source_side, qualified_side, interaction=True),
    }


def _paired_payload(
    *,
    same_target: bool = False,
    correct: bool = False,
    left_restate: bool = True,
    right_restate: bool = True,
) -> dict:
    left_target = "smartphone" if same_target else "fabric"
    right_target = "smartphone" if same_target else "table"
    left_q = "left" if correct else None
    right_q = "right" if correct else "left"
    return {
        "fusion": {
            "schema_version": "analysis-fusion-2.3.5",
            "qualified_body_parts": [
                _part("left", left_target, left_q),
                _part("right", right_target, right_q),
            ],
            "qualified_interactions": [
                _interaction(
                    "left",
                    left_target,
                    left_q,
                    kind="holding",
                    restate_side=left_restate,
                ),
                _interaction(
                    "right",
                    right_target,
                    right_q,
                    kind="contact",
                    restate_side=right_restate,
                ),
            ],
        }
    }


class PairedInteractionGuard237Tests(unittest.TestCase):
    def test_cross_field_repeated_distinct_target_conflict_is_withheld(self) -> None:
        out = guard_cross_field_paired_distinct_interactions(_paired_payload(), _dwpose())
        fusion = out["fusion"]
        parts = fusion["qualified_body_parts"]
        interactions = fusion["qualified_interactions"]

        self.assertEqual(fusion["schema_version"], "analysis-fusion-2.3.7")
        self.assertEqual([item["actor_part"] for item in interactions], ["hand", "hand"])
        self.assertEqual([item["part"] for item in parts], ["arm", "arm"])
        self.assertEqual(len(fusion["paired_distinct_interaction_laterality_audit"]["pairs_applied"]), 1)
        self.assertEqual(
            interactions[1]["fusion_v2"]["laterality_authority"],
            "paired_distinct_interaction_cross_field_conflict_withheld",
        )

    def test_no_side_restatement_does_not_veto_deterministic_correction(self) -> None:
        payload = _paired_payload(left_restate=False, right_restate=False)
        out = guard_cross_field_paired_distinct_interactions(payload, _dwpose())
        interactions = out["fusion"]["qualified_interactions"]
        parts = out["fusion"]["qualified_body_parts"]

        self.assertEqual(interactions[1]["actor_part"], "left hand")
        self.assertEqual(parts[1]["part"], "left arm")
        audit = out["fusion"]["paired_distinct_interaction_laterality_audit"]
        self.assertEqual(audit["pairs_applied"], [])
        self.assertEqual(
            audit["pairs_considered"][0]["source_side_restatement_in_notes"],
            {"left": False, "right": False},
        )

    def test_only_one_side_restatement_is_insufficient(self) -> None:
        payload = _paired_payload(left_restate=True, right_restate=False)
        out = guard_cross_field_paired_distinct_interactions(payload, _dwpose())
        fusion = out["fusion"]
        interactions = fusion["qualified_interactions"]
        parts = fusion["qualified_body_parts"]
        audit = fusion["paired_distinct_interaction_laterality_audit"]

        self.assertEqual(audit["pairs_applied"], [])
        self.assertEqual(len(audit["pairs_considered"]), 1)
        self.assertEqual(
            audit["pairs_considered"][0]["source_side_restatement_in_notes"],
            {"left": True, "right": False},
        )
        # One repeated source-side claim is deliberately insufficient to veto the
        # deterministic correction. The corrected right-source branch stays left.
        self.assertEqual(interactions[1]["actor_part"], "left hand")
        self.assertEqual(interactions[1]["fusion_v2"]["qualified_actor_anatomical_side"], "left")
        self.assertEqual(parts[1]["part"], "left arm")
        self.assertEqual(parts[1]["fusion_v2"]["qualified_anatomical_side"], "left")

    def test_same_target_pair_is_untouched_even_with_restatements(self) -> None:
        payload = _paired_payload(same_target=True)
        out = guard_cross_field_paired_distinct_interactions(payload, _dwpose())
        self.assertEqual(out["fusion"]["paired_distinct_interaction_laterality_audit"]["pairs_applied"], [])

    def test_incomplete_bilateral_arm_chains_do_not_apply_guard(self) -> None:
        out = guard_cross_field_paired_distinct_interactions(_paired_payload(), _dwpose(complete=False))
        audit = out["fusion"]["paired_distinct_interaction_laterality_audit"]
        self.assertFalse(audit["bilateral_arm_chains_complete"])
        self.assertEqual(audit["pairs_applied"], [])

    def test_already_consistent_pair_is_untouched(self) -> None:
        out = guard_cross_field_paired_distinct_interactions(_paired_payload(correct=True), _dwpose())
        interactions = out["fusion"]["qualified_interactions"]
        self.assertEqual([item["actor_part"] for item in interactions], ["left hand", "right hand"])
        self.assertEqual(out["fusion"]["paired_distinct_interaction_laterality_audit"]["pairs_applied"], [])


if __name__ == "__main__":
    unittest.main()
