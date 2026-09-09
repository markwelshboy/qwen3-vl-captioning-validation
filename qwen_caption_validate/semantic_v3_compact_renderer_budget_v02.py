from __future__ import annotations

import re
from typing import Any

from . import semantic_v3_compact_renderer_budget_v01 as budget_v01


ARTIFACT_VERSION = "semantic-v3-compact-renderer-budget-0.2"
RUN_VERSION = "semantic-v3-compact-renderer-budget-0.2-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-budget-v0.2"


# v0.2 changes the optimization target from additive sentence value to semantic-family
# coverage. This mirrors the identity-LoRA renderer policy: after trigger/governed
# pose, preserve transient appearance, salient held-object interaction, and a useful
# scene anchor before spending words on repeated framing/action/lighting detail.
_CLOTHING_RE = re.compile(
    r"\b(wear|wears|wearing|shirt|t-shirt|top|jacket|cardigan|sweatshirt|hoodie|"
    r"pullover|jeans|trousers|pants|shorts|dress|skirt|sneakers|shoes|boots|socks|"
    r"fleece|knit|crewneck|swimsuit|coat|blouse|sweater)\b",
    re.IGNORECASE,
)
_HELD_OBJECT_RE = re.compile(
    r"\b(hold|holds|holding|carry|carries|carrying|grip|grips|gripping)\b",
    re.IGNORECASE,
)
_INTERACTION_RE = re.compile(
    r"\b(clasped|supports?|hands?\s+(?:rest(?:s|ing)?\s+)?on\s+(?:their\s+)?hips|"
    r"arms?\s+(?:are\s+)?bent|hands?\s+(?:are\s+)?clasped|"
    r"(?:head|chin|hand|hands|arm|arms|forearm|forearms|fist)\s+(?:rests?|resting)|"
    r"fist)\b",
    re.IGNORECASE,
)
_SCENE_ANCHOR_RE = re.compile(
    r"\b(window|car|vehicle|trees?|woodland|forest|beach|shore|water|waves|hedge|"
    r"fence|basketball hoop|bed|headboard|nightstand|tapestry|kitchen|doorway|bench|"
    r"cafe|café|table|brick wall|black-and-white portrait|buildings?|stools?)\b",
    re.IGNORECASE,
)
_ACCESSORY_RE = re.compile(
    r"\b(glasses|sunglasses|earrings?|earbuds?|ring|rings|watch|smartwatch|bracelet|"
    r"tattoo|headband|necklace|nail polish|resistance band)\b",
    re.IGNORECASE,
)
_HAIR_STATE_RE = re.compile(
    r"\b(hair|strands?|tousled|damp|slicked|pulled back|gathered|tucked|windblown)\b",
    re.IGNORECASE,
)
_EXPRESSION_RE = re.compile(
    r"\b(smile|smiles|smiling|gaze|gazing|expression|teeth|thoughtful|calm|neutral|"
    r"warm|engaging)\b",
    re.IGNORECASE,
)
_FRAMING_RE = re.compile(
    r"\b(shot|close-up|close up|medium(?:-full)?|crop|foreground|foreshorten(?:ing)?|"
    r"low angle|low-angle|selfie|composition|lens|framed from|"
    r"filling most of (?:the )?frame|fills? most of (?:the )?frame)\b",
    re.IGNORECASE,
)
_GENERIC_SCENE_RE = re.compile(
    r"\b(background|wall|room|floor|indoor|outdoor|sky|surface|panelled|paneled)\b",
    re.IGNORECASE,
)
_LIGHTING_RE = re.compile(
    r"\b(lighting|overcast|sunlit|bright|shadows?|diffused|daylight|natural light|"
    r"soft light|frontal light)\b",
    re.IGNORECASE,
)
_LOW_VALUE_RE = re.compile(
    r"\b(logo|brand|stitch|stitching|residue|seam|drawstrings?|beads?|metallic hinges|"
    r'white "N"|New Balance)\b',
    re.IGNORECASE,
)

_CATEGORY_WEIGHTS: dict[str, float] = {
    "clothing": 120.0,
    "held_object": 110.0,
    "scene_anchor": 95.0,
    "interaction": 75.0,
    "accessory": 42.0,
    "expression": 35.0,
    "hair_state": 28.0,
    "framing": 24.0,
    "generic_scene": 7.0,
    "lighting": 3.0,
}

_PRIORITY_COVERAGE_BONUSES: dict[str, float] = {
    "clothing": 35.0,
    "held_object": 30.0,
    "scene_anchor": 25.0,
    "interaction": 18.0,
    "accessory": 12.0,
}

_CRITICAL_COVERAGE = ("clothing", "held_object", "scene_anchor")
_REPEAT_FRACTION = 0.025
_COMPACT_VARIANT_PENALTY = 1.5


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _split_sentences(text: str) -> list[str]:
    return budget_v01._split_sentences(text)


def _sentence_categories(sentence: str) -> list[str]:
    checks = (
        ("clothing", _CLOTHING_RE),
        ("held_object", _HELD_OBJECT_RE),
        ("interaction", _INTERACTION_RE),
        ("scene_anchor", _SCENE_ANCHOR_RE),
        ("accessory", _ACCESSORY_RE),
        ("hair_state", _HAIR_STATE_RE),
        ("expression", _EXPRESSION_RE),
        ("framing", _FRAMING_RE),
        ("generic_scene", _GENERIC_SCENE_RE),
        ("lighting", _LIGHTING_RE),
    )
    return [name for name, pattern in checks if pattern.search(sentence)]


def _sentence_score(sentence: str, index: int) -> tuple[float, list[str]]:
    categories = _sentence_categories(sentence)
    if index == 0:
        return 1000.0, categories
    score = sum(_CATEGORY_WEIGHTS[value] for value in categories)
    if _LOW_VALUE_RE.search(sentence):
        score -= 5.0
    return score, categories


def _category_mask(categories: list[str]) -> int:
    mask = 0
    for index, name in enumerate(_CATEGORY_WEIGHTS):
        if name in categories:
            mask |= 1 << index
    return mask


def _mask_categories(mask: int) -> set[str]:
    return {
        name
        for index, name in enumerate(_CATEGORY_WEIGHTS)
        if mask & (1 << index)
    }


def _variant_options(sentence: str, index: int) -> list[dict[str, Any]]:
    """Return conservative deterministic variants for one renderer sentence.

    v0.2 still never rewrites arbitrary prose. The only compaction currently allowed
    is removal of a trailing `, while ...` adjunct from a scene sentence. This is the
    exact shape that lets 00044 preserve glasses/clothing and the large window/red-car
    anchor without exceeding 90 words, while dropping only the low-priority foreground
    tail. The first trigger/pose sentence is never compacted.
    """
    options = [
        {
            "kind": "full",
            "text": sentence,
            "words": _word_count(sentence),
            "categories": _sentence_categories(sentence),
        }
    ]
    if index == 0:
        return options

    source_categories = set(options[0]["categories"])
    if not ({"scene_anchor", "generic_scene", "lighting"} & source_categories):
        return options

    marker = re.search(r",\s+while\s+", sentence, re.IGNORECASE)
    if marker is None:
        return options

    prefix = sentence[: marker.start()].rstrip(" ,;:")
    if not prefix:
        return options
    if prefix[-1] not in ".!?":
        prefix += "."
    if _word_count(prefix) < 6:
        return options

    prefix_categories = _sentence_categories(prefix)
    if "scene_anchor" in source_categories and "scene_anchor" not in prefix_categories:
        return options

    options.append(
        {
            "kind": "tail_clause_shave",
            "text": prefix,
            "words": _word_count(prefix),
            "categories": prefix_categories,
        }
    )
    return options


def _selection_score_increment(covered_mask: int, categories: list[str]) -> float:
    added = 0.0
    for index, name in enumerate(_CATEGORY_WEIGHTS):
        if name not in categories:
            continue
        weight = _CATEGORY_WEIGHTS[name]
        bit = 1 << index
        added += weight * (_REPEAT_FRACTION if covered_mask & bit else 1.0)
    return added


def _select_budget_sentences(
    caption: str,
    *,
    min_words: int,
    max_words: int,
) -> dict[str, Any]:
    """Select semantic-family coverage under the hard renderer word budget.

    Sentence zero is mandatory because it owns trigger binding and the primary governed
    pose. Tail sentences compete on new semantic-family coverage, not additive raw
    sentence scores. This prevents repeated action/framing language from outvoting the
    only clothing, held-object, or distinctive-scene sentence.
    """
    sentences = _split_sentences(caption)
    if not sentences:
        raise ValueError("Caption contains no sentences")

    analyses: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        score, categories = _sentence_score(sentence, index)
        analyses.append(
            {
                "index": index,
                "words": _word_count(sentence),
                "score": round(score, 3),
                "categories": categories,
                "selected": False,
                "selected_variant": None,
                "selected_text": None,
                "text": sentence,
            }
        )

    first_words = analyses[0]["words"]
    first_categories = analyses[0]["categories"]
    if first_words > max_words:
        analyses[0]["selected"] = True
        analyses[0]["selected_variant"] = "full"
        analyses[0]["selected_text"] = sentences[0]
        all_scene = [
            item["index"] for item in analyses if "scene_anchor" in item["categories"]
        ]
        return {
            "caption": sentences[0],
            "selected_sentence_indices": [0],
            "dropped_sentence_indices": list(range(1, len(sentences))),
            "compacted_sentence_indices": [],
            "sentence_analysis": analyses,
            "scene_sentence_indices": all_scene,
            "selected_scene_sentence_indices": [0] if 0 in all_scene else [],
            "selection_status": "first_sentence_exceeds_budget",
        }

    all_categories = {
        category
        for item in analyses
        for category in item["categories"]
    }
    source_mask = _category_mask(sorted(all_categories))
    first_mask = _category_mask(first_categories)

    # State: (used_words, covered_category_mask) -> (score, choices)
    # choices contains (sentence_index, variant_kind, variant_text).
    dp: dict[tuple[int, int], tuple[float, list[tuple[int, str, str]]]] = {
        (first_words, first_mask): (0.0, [(0, "full", sentences[0])])
    }

    for index in range(1, len(sentences)):
        variants = _variant_options(sentences[index], index)
        next_dp = dict(dp)  # Dropping this sentence is always allowed.
        for (used_words, covered_mask), (current_score, choices) in dp.items():
            for variant in variants:
                new_words = used_words + int(variant["words"])
                if new_words > max_words:
                    continue
                categories = list(variant["categories"])
                category_mask = _category_mask(categories)
                added_score = _selection_score_increment(covered_mask, categories)
                if _LOW_VALUE_RE.search(str(variant["text"])):
                    added_score -= 5.0
                if variant["kind"] != "full":
                    added_score -= _COMPACT_VARIANT_PENALTY
                # Tiny fill value breaks otherwise identical semantic ties without
                # making verbosity itself the objective.
                added_score += 0.005 * int(variant["words"])
                new_mask = covered_mask | category_mask
                key = (new_words, new_mask)
                candidate = (
                    current_score + added_score,
                    choices + [(index, str(variant["kind"]), str(variant["text"]))],
                )
                prior = next_dp.get(key)
                if prior is None or candidate[0] > prior[0]:
                    next_dp[key] = candidate
        dp = next_dp

    preferred_mid = (min_words + max_words) / 2.0
    candidates: list[tuple[float, float, int, int, list[tuple[int, str, str]]]] = []
    for (used_words, covered_mask), (raw_score, choices) in dp.items():
        score = raw_score
        if used_words < min_words:
            score -= 10.0 * (min_words - used_words)
        covered_categories = _mask_categories(covered_mask)
        source_categories = _mask_categories(source_mask)
        for category, bonus in _PRIORITY_COVERAGE_BONUSES.items():
            if category in source_categories and category in covered_categories:
                score += bonus
        # Prefer the center of the profile only after semantic coverage has decided
        # the meaningful content.
        preference_distance = abs(used_words - preferred_mid)
        score -= 0.01 * preference_distance
        candidates.append(
            (score, -preference_distance, used_words, covered_mask, choices)
        )

    if not candidates:
        raise RuntimeError("No budget-selection candidate was produced")

    _, _, _, selected_mask, selected_choices = max(
        candidates,
        key=lambda value: (value[0], value[1], value[2]),
    )
    selected_choices = sorted(selected_choices, key=lambda value: value[0])
    selected_indices = [value[0] for value in selected_choices]
    selected_set = set(selected_indices)
    compacted_indices = [
        index for index, kind, _ in selected_choices if kind != "full"
    ]
    selected_text_by_index = {
        index: (kind, text) for index, kind, text in selected_choices
    }

    for item in analyses:
        index = int(item["index"])
        item["selected"] = index in selected_set
        if index in selected_text_by_index:
            kind, text = selected_text_by_index[index]
            item["selected_variant"] = kind
            item["selected_text"] = text

    selected_caption = " ".join(text for _, _, text in selected_choices)
    dropped = [index for index in range(len(sentences)) if index not in selected_set]
    all_scene = [
        item["index"] for item in analyses if "scene_anchor" in item["categories"]
    ]
    selected_scene = [
        index
        for index, _, text in selected_choices
        if "scene_anchor" in _sentence_categories(text)
    ]

    return {
        "caption": selected_caption,
        "selected_sentence_indices": selected_indices,
        "dropped_sentence_indices": dropped,
        "compacted_sentence_indices": compacted_indices,
        "sentence_analysis": analyses,
        "scene_sentence_indices": all_scene,
        "selected_scene_sentence_indices": selected_scene,
        "source_semantic_categories": sorted(_mask_categories(source_mask)),
        "selected_semantic_categories": sorted(_mask_categories(selected_mask)),
        "selection_status": "selected_semantic_family_coverage",
    }


_original_budget_renderer_record = budget_v01.budget_renderer_record


def budget_renderer_record(
    *,
    renderer_record: dict[str, Any],
    pose: dict[str, Any],
) -> dict[str, Any]:
    result = _original_budget_renderer_record(
        renderer_record=renderer_record,
        pose=pose,
    )

    selection = result.get("selection") or {}
    analyses = selection.get("sentence_analysis") or []
    source_categories = {
        category
        for item in analyses
        for category in (item.get("categories") or [])
    }
    selected_categories: set[str] = set()
    for item in analyses:
        if not item.get("selected"):
            continue
        selected_text = str(item.get("selected_text") or item.get("text") or "")
        selected_categories.update(_sentence_categories(selected_text))

    coverage = {
        category: {
            "present_in_renderer": category in source_categories,
            "selected": category in selected_categories,
        }
        for category in _CATEGORY_WEIGHTS
    }
    dropped_critical = [
        category
        for category in _CRITICAL_COVERAGE
        if coverage[category]["present_in_renderer"] and not coverage[category]["selected"]
    ]
    review_warnings = list(result.get("review_warnings") or [])
    # v0.1 used a broad `scene` regex and could warn on a plain gray wall. Replace
    # that with v0.2's distinctive scene-anchor coverage check.
    review_warnings = [
        value for value in review_warnings if value != "scene_anchor_sentence_dropped"
    ]
    review_warnings.extend(
        f"priority_{category}_coverage_dropped" for category in dropped_critical
    )
    result["review_warnings"] = list(dict.fromkeys(review_warnings))
    result["priority_coverage"] = coverage
    result["dropped_critical_semantic_categories"] = dropped_critical
    return result


def main() -> int:
    # Reuse the mature v0.1 workspace/index writer while swapping only the versioned
    # deterministic selection policy. v0.1's record builder resolves these helpers
    # from its module globals at call time, so this remains a thin compatibility layer.
    budget_v01.ARTIFACT_VERSION = ARTIFACT_VERSION
    budget_v01.RUN_VERSION = RUN_VERSION
    budget_v01.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    budget_v01._sentence_categories = _sentence_categories
    budget_v01._sentence_score = _sentence_score
    budget_v01._select_budget_sentences = _select_budget_sentences
    budget_v01.budget_renderer_record = budget_renderer_record
    return budget_v01.main()


if __name__ == "__main__":
    raise SystemExit(main())
