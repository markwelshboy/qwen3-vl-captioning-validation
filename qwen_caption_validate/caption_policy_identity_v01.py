from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.2.2"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
SCHEMA_VERSION = "caption-fact-sheet-0.3"
PROFILE = "character_identity_v0.1"

# Caption-policy goal for character/identity training:
# describe transient factors so they do not become entangled with identity,
# while withholding persistent identity traits as conditioning concepts.
HAIR_COLOR_RE = re.compile(
    r"\b(?:blond(?:e)?|brunette|brown|black|red|auburn|ginger|gray|grey|silver|white|"
    r"dark[- ]brown|light[- ]brown|dark[- ]blond(?:e)?|light[- ]blond(?:e)?)\b",
    re.I,
)
HAIR_LENGTH_RE = re.compile(
    r"\b(?:very\s+)?(?:short|long|medium(?:[- ]length)?|shoulder[- ]length|chin[- ]length|"
    r"jaw[- ]length|waist[- ]length|hip[- ]length|cropped)\b",
    re.I,
)
HAIR_TEXTURE_RE = re.compile(r"\b(?:straight|wavy|curly|coily)\b", re.I)
HAIR_STATE_RE = re.compile(
    r"\b(?:ponytail|pigtails?|bun|topknot|braid(?:ed|s)?|updo|tied\s+back|pulled\s+back|"
    r"swept\s+back|slicked\s+back|tucked\s+behind|loose\s+strands?|face[- ]framing\s+strands?|"
    r"messy|styled|half[- ]up|half\s+up)\b",
    re.I,
)
EYE_COLOR_RE = re.compile(
    r"\b(?:blue|green|brown|hazel|gray|grey|amber)\s+eyes?\b|\beyes?\s+(?:are\s+)?(?:blue|green|brown|hazel|gray|grey|amber)\b",
    re.I,
)
SKIN_TONE_RE = re.compile(
    r"\b(?:fair|pale|light|medium|olive|tan(?:ned)?|brown|dark|deep)\s+(?:skin|complexion)\b|"
    r"\b(?:skin|complexion)\s+(?:is\s+)?(?:fair|pale|light|medium|olive|tan(?:ned)?|brown|dark|deep)\b",
    re.I,
)
AGE_RE = re.compile(
    r"\b(?:young|youthful|middle[- ]aged|older|elderly|senior)\s+(?:woman|man|person|adult)\b|"
    r"\b(?:woman|man|person|adult)\s+in\s+(?:his|her|their)\s+(?:20s|30s|40s|50s|60s|70s|80s)\b",
    re.I,
)
FACE_STRUCTURE_RE = re.compile(
    r"\b(?:round|oval|square|heart[- ]shaped|long)\s+face\b|"
    r"\b(?:high|prominent)\s+cheekbones\b|\b(?:strong|defined|square)\s+jaw(?:line)?\b",
    re.I,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _clean_hair_phrase(text: str) -> str | None:
    """Keep transient hairstyle/state while removing persistent identity cues.

    Examples:
      blonde shoulder-length hair -> None
      blonde hair pulled back -> hair pulled back
      blonde hair in a messy ponytail with loose strands -> hair in a messy ponytail with loose strands
      curly brown hair -> None
    """
    if not re.search(r"\bhair\b", text, re.I):
        return text

    had_intrinsic = bool(HAIR_COLOR_RE.search(text) or HAIR_LENGTH_RE.search(text) or HAIR_TEXTURE_RE.search(text))
    has_transient_state = bool(HAIR_STATE_RE.search(text))

    out = HAIR_COLOR_RE.sub("", text)
    out = HAIR_LENGTH_RE.sub("", out)
    out = HAIR_TEXTURE_RE.sub("", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;])", r"\1", out)
    out = re.sub(r"\b(?:with|and)\s*$", "", out, flags=re.I)
    out = out.strip(" ,.;-")

    # An unstyled residual like just "hair" is not useful training context.
    if had_intrinsic and not has_transient_state:
        return None
    if out.casefold() in {"hair", "the hair", "their hair", "his hair", "her hair"}:
        return None
    return out or None


def _intrinsic_only_reason(text: str) -> str | None:
    if EYE_COLOR_RE.search(text):
        return "eye_color"
    if SKIN_TONE_RE.search(text):
        return "skin_tone"
    if AGE_RE.search(text):
        return "apparent_age"
    if FACE_STRUCTURE_RE.search(text):
        return "facial_structure"
    return None


def _apply_appearance_item(item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    out = copy.deepcopy(item)
    composer_text = _clean(out.get("composer_text"))
    if composer_text is None:
        return out, None

    protected_reason = _intrinsic_only_reason(composer_text)
    if protected_reason:
        out["composer_text"] = None
        out["promotion_status"] = "protected_intrinsic_identity"
        out["caption_policy"] = {
            "profile": PROFILE,
            "decision": "withhold",
            "protected_trait": protected_reason,
        }
        out["note_caption_policy"] = (
            "Persistent identity semantics are retained in the fact sheet for audit but withheld from character/identity caption composition."
        )
        return out, protected_reason

    if re.search(r"\bhair\b", composer_text, re.I):
        normalized = _clean_hair_phrase(composer_text)
        if normalized != composer_text:
            if normalized:
                out["composer_text"] = normalized
                out["normalized_caption_text"] = normalized
                out["promotion_status"] = "accepted_transient_appearance"
                out["caption_policy"] = {
                    "profile": PROFILE,
                    "decision": "strip_intrinsic_keep_transient",
                    "protected_trait": "hair_identity",
                }
                out["note_caption_policy"] = (
                    "Natural hair color, baseline length, and texture are protected; transient hairstyle/state remains captionable."
                )
                return out, "hair_identity_partially_withheld"
            out["composer_text"] = None
            out["promotion_status"] = "protected_intrinsic_identity"
            out["caption_policy"] = {
                "profile": PROFILE,
                "decision": "withhold",
                "protected_trait": "hair_identity",
            }
            out["note_caption_policy"] = (
                "Natural hair color, baseline length, and texture are protected identity semantics for character training."
            )
            return out, "hair_identity"

    return out, None


def apply_character_identity_policy(sheet: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(sheet)
    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    visual = facts.get("visual") if isinstance(facts.get("visual"), dict) else {}
    appearance = visual.get("appearance") if isinstance(visual.get("appearance"), list) else []

    reasons: list[str] = []
    normalized: list[dict[str, Any]] = []
    for item in appearance:
        if not isinstance(item, dict):
            continue
        updated, reason = _apply_appearance_item(item)
        normalized.append(updated)
        if reason:
            reasons.append(reason)
    visual["appearance"] = normalized

    out["caption_policy"] = {
        "profile": PROFILE,
        "training_objective": "character_identity",
        "principle": "describe_transient_factors_protect_persistent_identity_traits",
        "protected_identity_traits": [
            "natural_hair_color",
            "baseline_hair_length",
            "baseline_hair_texture",
            "eye_color",
            "skin_tone",
            "apparent_age",
            "facial_structure",
        ],
        "transient_appearance_remains_captionable": True,
    }

    audit = out.get("audit") if isinstance(out.get("audit"), dict) else {}
    warnings = [str(x) for x in (audit.get("warnings") or []) if x]
    if reasons:
        warnings.append("caption_policy_intrinsic_identity_traits_withheld")
    audit["warnings"] = sorted(set(warnings))
    audit["phase"] = "4C"
    invariants = audit.get("invariants") if isinstance(audit.get("invariants"), dict) else {}
    invariants.update(
        appearance_observation_is_preserved_for_audit=True,
        persistent_identity_traits_are_not_composer_visible=True,
        transient_appearance_remains_composer_visible=True,
        mixed_hair_phrases_keep_transient_state_when_possible=True,
    )
    audit["invariants"] = invariants
    out["audit"] = audit
    out["schema_version"] = SCHEMA_VERSION
    return out


def _input_files(directory: Path, only: set[str]) -> list[Path]:
    paths = sorted(directory.glob("*.fact_sheet.json"))
    return [p for p in paths if not only or p.name.removesuffix(".fact_sheet.json") in only]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-4C character/identity caption policy over governed Phase-4B.2 fact sheets.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else run_dir / DEFAULT_INPUT_SUBDIR
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / DEFAULT_OUTPUT_SUBDIR
    if not input_dir.is_dir():
        print(f"Phase-4B.2 fact-sheet directory not found: {input_dir}", file=sys.stderr)
        return 2

    paths = _input_files(input_dir, set(args.only))
    if not paths:
        print(f"No matching fact sheets found in {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for input_path in paths:
        key = input_path.name.removesuffix(".fact_sheet.json")
        out_path = output_dir / input_path.name
        if out_path.is_file() and not args.overwrite:
            record = _read_json(out_path)
        else:
            source = _read_json(input_path)
            if source.get("schema_version") != "caption-fact-sheet-0.2.2":
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "image_key": key,
                    "error": f"expected caption-fact-sheet-0.2.2, got {source.get('schema_version')}",
                }
            else:
                record = apply_character_identity_policy(source)
            _write_json(out_path, record)
        records.append(record)
        print(f"{key}: {record.get('status')} | identity policy")

    records.sort(key=lambda r: str(r.get("image_key") or ""))
    counts = Counter(str(r.get("status") or "unknown") for r in records)
    index = {
        "schema_version": SCHEMA_VERSION + "-run",
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "profile": PROFILE,
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "invariants": {
            "no_model_calls": True,
            "phase4b2_is_input_not_recomputed": True,
            "appearance_observation_is_preserved": True,
            "persistent_identity_traits_are_withheld_from_composer": True,
            "transient_appearance_remains_captionable": True,
        },
        "records": records,
    }
    _write_json(output_dir / "caption_policy.index.json", index)
    print(f"Index: {output_dir / 'caption_policy.index.json'}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
