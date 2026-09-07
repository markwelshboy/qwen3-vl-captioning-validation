from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from . import semantic_v3_compact_renderer_v01 as renderer_v01
from . import semantic_v3_compact_renderer_finish_v01 as finisher_v01
from .runner import model_slug, resolve_model_id


ARTIFACT_VERSION = "semantic-v3-compact-renderer-budget-0.1"
RUN_VERSION = "semantic-v3-compact-renderer-budget-0.1-run"
DEFAULT_OUTPUT_SUBDIR = "compact-renderer-budget-v0.1"


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SCENE_RE = re.compile(
    r"\b(background|behind|window|wall|room|cafe|café|forest|woodland|beach|shore|"
    r"sand|path|road|hedge|tree|fence|street|building|bed|headboard|nightstand|"
    r"floor|indoor|outdoor|sky|water|waves|panelled|paneled|brick|vehicle|car)\b",
    re.IGNORECASE,
)
_CLOTHING_RE = re.compile(
    r"\b(wear|wears|wearing|shirt|t-shirt|top|jacket|cardigan|sweatshirt|hoodie|"
    r"pullover|jeans|trousers|pants|shorts|dress|skirt|sneakers|shoes|boots|socks|"
    r"fleece|knit|crewneck|sleeves?|athletic)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(hold|holds|holding|hand|hands|clasped|rests?|resting|leans?|leaning|"
    r"sits?|seated|stands?|standing|squat|crouch|reclin|gaze|gazing|smil|faces?|"
    r"facing|bent|raised|support)\w*\b",
    re.IGNORECASE,
)
_FRAMING_RE = re.compile(
    r"\b(camera|shot|close-up|close up|medium|crop|frame|foreground|foreshorten|"
    r"low angle|selfie|portrait|composition|lens)\b",
    re.IGNORECASE,
)
_HAIR_RE = re.compile(
    r"\b(hair|strands?|tousled|damp|slicked|pulled back|gathered|tucked|windblown)\b",
    re.IGNORECASE,
)
_ACCESSORY_RE = re.compile(
    r"\b(glasses|sunglasses|earrings?|earbuds?|ring|rings|watch|bracelet|mug|cup|"
    r"tattoo|headband|necklace|band)\b",
    re.IGNORECASE,
)
_LIGHTING_RE = re.compile(
    r"\b(light|lighting|overcast|sunlit|bright|shadow|diffused|frontal|daylight)\b",
    re.IGNORECASE,
)
_LOW_VALUE_RE = re.compile(
    r"\b(logo|brand|stitch|stitching|residue|seam|drawstrings?|beads?|metallic hinges|"
    r'white "N"|New Balance)\b',
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def _split_sentences(text: str) -> list[str]:
    return [value.strip() for value in _SENTENCE_SPLIT_RE.split(text.strip()) if value.strip()]


def _sentence_categories(sentence: str) -> list[str]:
    categories: list[str] = []
    checks = (
        ("clothing", _CLOTHING_RE),
        ("action", _ACTION_RE),
        ("scene", _SCENE_RE),
        ("framing", _FRAMING_RE),
        ("hair", _HAIR_RE),
        ("accessory", _ACCESSORY_RE),
        ("lighting", _LIGHTING_RE),
    )
    for name, pattern in checks:
        if pattern.search(sentence):
            categories.append(name)
    return categories


_CATEGORY_WEIGHTS = {
    "clothing": 50.0,
    "action": 28.0,
    "scene": 50.0,
    "framing": 22.0,
    "hair": 12.0,
    "accessory": 10.0,
    "lighting": 4.0,
}


def _sentence_score(sentence: str, index: int) -> tuple[float, list[str]]:
    categories = _sentence_categories(sentence)
    if index == 0:
        return 1000.0, categories
    score = sum(_CATEGORY_WEIGHTS[value] for value in categories)
    if _LOW_VALUE_RE.search(sentence):
        score -= 5.0
    return score, categories


def _select_budget_sentences(
    caption: str,
    *,
    min_words: int,
    max_words: int,
) -> dict[str, Any]:
    """Select complete source sentences under a hard word budget.

    Sentence zero is mandatory because it carries trigger binding / primary pose.
    Remaining sentences are selected with a tiny dynamic program that maximizes
    semantic coverage rather than blindly retaining a prefix. Original sentence
    order is preserved and sentence text is never edited.
    """
    sentences = _split_sentences(caption)
    if not sentences:
        raise ValueError("Caption contains no sentences")

    lengths = [_word_count(value) for value in sentences]
    analyses: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        score, categories = _sentence_score(sentence, index)
        analyses.append(
            {
                "index": index,
                "words": lengths[index],
                "score": round(score, 3),
                "categories": categories,
                "selected": False,
                "text": sentence,
            }
        )

    if lengths[0] > max_words:
        analyses[0]["selected"] = True
        return {
            "caption": sentences[0],
            "selected_sentence_indices": [0],
            "dropped_sentence_indices": list(range(1, len(sentences))),
            "sentence_analysis": analyses,
            "scene_sentence_indices": [
                item["index"] for item in analyses if "scene" in item["categories"]
            ],
            "selected_scene_sentence_indices": [
                0 if "scene" in analyses[0]["categories"] else None
            ][:
                1 if "scene" in analyses[0]["categories"] else 0
            ],
            "selection_status": "first_sentence_exceeds_budget",
        }

    remaining_budget = max_words - lengths[0]
    # map used_words -> (semantic_score, selected_indices)
    dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}

    for index in range(1, len(sentences)):
        words = lengths[index]
        score = float(analyses[index]["score"])
        next_dp = dict(dp)
        for used, (current_score, selected) in dp.items():
            new_used = used + words
            if new_used > remaining_budget:
                continue
            # Very small fill bonus favors using available capacity only after
            # semantic category value has done the real ranking.
            candidate_score = current_score + score + 0.08 * words
            prior = next_dp.get(new_used)
            if prior is None or candidate_score > prior[0]:
                next_dp[new_used] = (candidate_score, selected + [index])
        dp = next_dp

    candidates: list[tuple[float, int, list[int]]] = []
    for used, (score, selected) in dp.items():
        total_words = lengths[0] + used
        candidates.append((score, total_words, selected))

    in_range = [value for value in candidates if value[1] >= min_words]
    pool = in_range if in_range else candidates
    _, _, selected_tail = max(pool, key=lambda value: (value[0], value[1]))

    selected_indices = sorted([0] + selected_tail)
    selected_set = set(selected_indices)
    for item in analyses:
        item["selected"] = item["index"] in selected_set

    selected_caption = " ".join(sentences[index] for index in selected_indices)
    dropped = [index for index in range(len(sentences)) if index not in selected_set]

    all_scene = [item["index"] for item in analyses if "scene" in item["categories"]]
    selected_scene = [index for index in all_scene if index in selected_set]

    return {
        "caption": selected_caption,
        "selected_sentence_indices": selected_indices,
        "dropped_sentence_indices": dropped,
        "sentence_analysis": analyses,
        "scene_sentence_indices": all_scene,
        "selected_scene_sentence_indices": selected_scene,
        "selection_status": "selected",
    }


def budget_renderer_record(
    *,
    renderer_record: dict[str, Any],
    pose: dict[str, Any],
) -> dict[str, Any]:
    profile = str(renderer_record.get("profile") or "")
    if profile not in renderer_v01.PROFILE_BUDGETS:
        raise ValueError(f"Unknown profile: {profile}")

    trigger = str(renderer_record.get("trigger") or "").strip()
    semantic_caption = str(renderer_record.get("semantic_caption") or "").strip()
    initial_caption = str(renderer_record.get("rendered_caption") or "").strip()
    if not trigger or not semantic_caption or not initial_caption:
        raise ValueError("Renderer record missing trigger, semantic_caption, or rendered_caption")

    min_words, max_words = renderer_v01.PROFILE_BUDGETS[profile]
    initial_audit = finisher_v01.quality_audit(
        semantic_caption=semantic_caption,
        rendered_caption=initial_caption,
        pose=pose,
        trigger=trigger,
        profile=profile,
    )

    if _word_count(initial_caption) <= max_words:
        sentence_analysis = []
        for index, sentence in enumerate(_split_sentences(initial_caption)):
            score, categories = _sentence_score(sentence, index)
            sentence_analysis.append(
                {
                    "index": index,
                    "words": _word_count(sentence),
                    "score": round(score, 3),
                    "categories": categories,
                    "selected": True,
                    "text": sentence,
                }
            )
        selection = {
            "caption": initial_caption,
            "selected_sentence_indices": list(range(len(sentence_analysis))),
            "dropped_sentence_indices": [],
            "sentence_analysis": sentence_analysis,
            "scene_sentence_indices": [
                item["index"] for item in sentence_analysis if "scene" in item["categories"]
            ],
            "selected_scene_sentence_indices": [
                item["index"] for item in sentence_analysis if "scene" in item["categories"]
            ],
            "selection_status": "passthrough_within_budget",
        }
        final_caption = initial_caption
        action = "passthrough"
    else:
        selection = _select_budget_sentences(
            initial_caption,
            min_words=min_words,
            max_words=max_words,
        )
        final_caption = str(selection["caption"])
        action = "sentence_budget_select"

    final_audit = finisher_v01.quality_audit(
        semantic_caption=semantic_caption,
        rendered_caption=final_caption,
        pose=pose,
        trigger=trigger,
        profile=profile,
    )
    hard_warnings = [str(value) for value in (final_audit.get("warnings") or [])]
    length_warnings = {
        "rendered_caption_above_target_words",
        "rendered_caption_below_target_words",
    }
    semantic_warnings = [value for value in hard_warnings if value not in length_warnings]

    if not hard_warnings:
        status = "pass"
    elif semantic_warnings:
        status = "semantic_repair_required"
    else:
        status = "budget_unresolved"

    review_warnings: list[str] = []
    if selection.get("scene_sentence_indices") and not selection.get("selected_scene_sentence_indices"):
        review_warnings.append("scene_anchor_sentence_dropped")

    return {
        "schema_version": ARTIFACT_VERSION,
        "image_key": renderer_record.get("image_key"),
        "profile": profile,
        "trigger": renderer_record.get("trigger"),
        "grammar_profile": renderer_record.get("grammar_profile"),
        "renderer_model": renderer_record.get("model"),
        "renderer_backend": renderer_record.get("backend"),
        "semantic_caption": semantic_caption,
        "pose_corrections": renderer_record.get("pose_corrections") or [],
        "initial_caption": initial_caption,
        "budgeted_caption": final_caption,
        "budget_action": action,
        "selection": selection,
        "initial_quality_audit": initial_audit,
        "quality_audit": final_audit,
        "semantic_repair_required": bool(semantic_warnings),
        "remaining_hard_warnings": hard_warnings,
        "review_warnings": review_warnings,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically enforce Semantic V3 compact/medium word budgets by "
            "selecting complete renderer sentences. No model/GPU load occurs."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--renderer-dir", type=Path)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-model", default="32b-fp8")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(renderer_v01.PROFILE_BUDGETS),
        default=["compact", "medium"],
    )
    parser.add_argument("--only", nargs="+", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    source_slug = model_slug(resolve_model_id(args.source_model))
    renderer_dir = (
        args.renderer_dir
        or (run_dir / "semantic-v3" / "compact-renderer-v0.2" / source_slug)
    ).expanduser().resolve()
    pose_dir = (
        args.pose_dir
        or (run_dir / "semantic-v3" / "pose-language-v0.1")
    ).expanduser().resolve()
    output_dir = (
        args.output_dir
        or (run_dir / "semantic-v3" / DEFAULT_OUTPUT_SUBDIR / source_slug)
    ).expanduser().resolve()

    for label, path in (("renderer", renderer_dir), ("pose", pose_dir)):
        if not path.is_dir():
            print(f"{label.capitalize()} directory not found: {path}", file=sys.stderr)
            return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_keys = set(args.only)
    selected_profiles = set(args.profiles)
    source_paths = sorted(renderer_dir.glob("*.rendered_caption.json"))
    records: list[dict[str, Any]] = []

    for source_path in source_paths:
        source = _read_json(source_path)
        profile = str(source.get("profile") or "")
        key = str(source.get("image_key") or "")
        if profile not in selected_profiles:
            continue
        if selected_keys and key not in selected_keys:
            continue

        pose_path = pose_dir / f"{key}.pose_language.json"
        if not pose_path.is_file():
            print(f"Missing Pose language: {pose_path}", file=sys.stderr)
            return 2
        pose = _read_json(pose_path)
        payload = budget_renderer_record(renderer_record=source, pose=pose)
        payload["renderer_source"] = str(source_path)
        payload["pose_source"] = str(pose_path)

        output_path = output_dir / f"{key}.{profile}.budgeted_caption.json"
        if args.overwrite or not output_path.exists():
            _write_json(output_path, payload)
        else:
            payload = _read_json(output_path)
        records.append(payload)

        print(
            f"SEMANTIC_V3_RENDERER_BUDGET image={key} profile={profile} "
            f"words={payload['quality_audit']['rendered_word_count']} "
            f"status={payload['status']} action={payload['budget_action']} "
            f"review={','.join(payload.get('review_warnings') or []) or 'none'}"
        )

    if not records:
        print("No matching renderer artifacts found.", file=sys.stderr)
        return 2

    profile_summary: dict[str, Any] = {}
    for profile in args.profiles:
        values = [value for value in records if value.get("profile") == profile]
        if not values:
            continue
        words = [int(value["quality_audit"]["rendered_word_count"]) for value in values]
        profile_summary[profile] = {
            "record_count": len(values),
            "pass_count": sum(value.get("status") == "pass" for value in values),
            "semantic_repair_required_count": sum(
                bool(value.get("semantic_repair_required")) for value in values
            ),
            "review_warning_count": sum(bool(value.get("review_warnings")) for value in values),
            "word_count_min": min(words),
            "word_count_median": statistics.median(words),
            "word_count_max": max(words),
        }

    index = {
        "schema_version": RUN_VERSION,
        "run_dir": str(run_dir),
        "renderer_dir": str(renderer_dir),
        "pose_dir": str(pose_dir),
        "output_dir": str(output_dir),
        "profiles": args.profiles,
        "record_count": len(records),
        "profile_summary": profile_summary,
        "records": [
            {
                "image_key": value.get("image_key"),
                "profile": value.get("profile"),
                "status": value.get("status"),
                "budget_action": value.get("budget_action"),
                "word_count": value["quality_audit"]["rendered_word_count"],
                "remaining_hard_warnings": value.get("remaining_hard_warnings") or [],
                "review_warnings": value.get("review_warnings") or [],
            }
            for value in records
        ],
    }
    _write_json(output_dir / "budget.index.json", index)

    summary_parts = []
    for profile, summary in profile_summary.items():
        summary_parts.append(
            f"{profile}:pass={summary['pass_count']}/{summary['record_count']} "
            f"words={summary['word_count_min']}/{summary['word_count_median']}/{summary['word_count_max']} "
            f"semantic_repair={summary['semantic_repair_required_count']}"
        )
    print(f"Deterministic training-caption budgets: {output_dir}")
    print(f"Records: {len(records)}; " + "; ".join(summary_parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
