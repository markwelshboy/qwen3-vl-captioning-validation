from __future__ import annotations

import unittest

from qwen_caption_validate.laterality_paired_interaction_guard_236 import (
    guard_paired_distinct_interactions,
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


def _interaction(source_side: str, target: str, qualified_side: str | None, kind: str = "contact") -> dict:
    return {
        "type": kind,
        "actor_part": f"{qualified_side} hand" if qualified_side else "hand",
        "source_actor_part": f"{source_side} hand",
        "actor_ownership": "target",
        "target": target,
        "evidence_status": "observed",
        "confidence": 0.95,
        "fusion_v2": _state(source_side, qualified_side, interaction=True),
    }


def _paired_payload(*, same_target: bool = False, correct: bool = False) -> dict:
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
                _interaction("left", left_target, left_q, "holding"),
                _interaction("right", right_target, right_q, "contact"),
            ],
        }
    }


class PairedInteractionGuard236Tests(unittest.TestCase):
    def test_distinct_target_pair_with_greedy_side_conflict_is_withheld(self) -> None:
        out = guard_paired_distinct_interactions(_paired_payload(), _dwpose())
        fusion = out["fusion"]
        parts = fusion["qualified_body_parts"]
        interactions = fusion["qualified_interactions"]

        self.assertEqual(fusion["schema_version"], "analysis-fusion-2.3.6")
        self.assertEqual([item["actor_part"] for item in interactions], ["hand", "hand"])
        self.assertEqual([item["part"] for item in parts], ["arm", "arm"])
        self.assertTrue(all(not item["fusion_v2"]["laterality_selection_usable"] for item in interactions))
        self.assertTrue(all(not item["fusion_v2"]["laterality_selection_usable"] for item in parts))
        self.assertEqual(
            interactions[1]["fusion_v2"]["laterality_authority"],
            "paired_distinct_interaction_conflict_withheld",
        )
        audit = fusion["paired_distinct_interaction_laterality_audit"]
        self.assertEqual(len(audit["pairs_applied"]), 1)

    def test_single_corrected_hand_interaction_is_untouched(self) -> None:
        payload = {
            "fusion": {
                "schema_version": "analysis-fusion-2.3.5",
                "qualified_body_parts": [_part("right", "chin", "left")],
                "qualified_interactions": [_interaction("right", "chin", "left", "support")],
            }
        }
        out = guard_paired_distinct_interactions(payload, _dwpose())
        interaction = out["fusion"]["qualified_interactions"][0]
        self.assertEqual(interaction["actor_part"], "left hand")
        self.assertEqual(interaction["fusion_v2"]["qualified_actor_anatomical_side"], "left")
        self.assertEqual(out["fusion"]["paired_distinct_interaction_laterality_audit"]["pairs_applied"], [])

    def test_bilateral_same_target_pair_is_untouched(self) -> None:
        payload = _paired_payload(same_target=True)
        out = guard_paired_distinct_interactions(payload, _dwpose())
        interaction = out["fusion"]["qualified_interactions"][1]
        self.assertEqual(interaction["actor_part"], "left hand")
        self.assertEqual(interaction["fusion_v2"]["qualified_actor_anatomical_side"], "left")
        self.assertEqual(out["fusion"]["paired_distinct_interaction_laterality_audit"]["pairs_applied"], [])

    def test_incomplete_bilateral_arm_chains_do_not_apply_guard(self) -> None:
        payload = _paired_payload()
        out = guard_paired_distinct_interactions(payload, _dwpose(complete=False))
        interaction = out["fusion"]["qualified_interactions"][1]
        self.assertEqual(interaction["actor_part"], "left hand")
        audit = out["fusion"]["paired_distinct_interaction_laterality_audit"]
        self.assertFalse(audit["bilateral_arm_chains_complete"])
        self.assertEqual(audit["pairs_applied"], [])

    def test_already_consistent_distinct_target_pair_is_untouched(self) -> None:
        payload = _paired_payload(correct=True)
        out = guard_paired_distinct_interactions(payload, _dwpose())
        interactions = out["fusion"]["qualified_interactions"]
        self.assertEqual([item["actor_part"] for item in interactions], ["left hand", "right hand"])
        self.assertEqual(out["fusion"]["paired_distinct_interaction_laterality_audit"]["pairs_applied"], [])


if __name__ == "__main__":
    unittest.main()
