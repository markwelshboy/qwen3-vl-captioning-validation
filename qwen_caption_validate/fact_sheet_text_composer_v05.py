from __future__ import annotations

from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v04 as phase53

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v04.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.5"
SCHEMA_VERSION = "fact-sheet-text-composer-0.5"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3"

_BASE_PROJECTION = phase53._projection

# Re-export validated Phase-5.3 helpers for tests and downstream retry tooling.
_torso_fact = phase53._torso_fact
_gaze_fact = phase53._gaze_fact
_caption_audit = phase53._caption_audit


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _canonical_broad_pose(body: dict[str, Any]) -> str | None:
    """Return the post-specialist broad pose that the composer may publish.

    Phase-4B.9 can replace an upstream Qwen pose with a specialist-authoritative
    pose.  Prefer that explicit canonical adjudication.  Otherwise consume the
    pose candidate's composer-facing text, including accepted specialist
    promotion statuses, before falling back to the original text field.
    """
    adjudication = body.get("broad_pose_adjudication") if isinstance(body.get("broad_pose_adjudication"), dict) else {}
    if adjudication.get("composer_authoritative"):
        canonical = _clean(adjudication.get("canonical_pose_text"))
        if canonical:
            return canonical

    pose = body.get("pose_candidate") if isinstance(body.get("pose_candidate"), dict) else {}
    status = str(pose.get("promotion_status") or "").strip()
    accepted = status in {"candidate", "accepted", "resolved"} or status.startswith("accepted_")
    if not accepted:
        return None

    for key in ("composer_text", "normalized_text", "text"):
        text = _clean(pose.get(key))
        if text:
            return text
    return None


def _global_support_shape(body: dict[str, Any]) -> dict[str, Any] | None:
    raw = body.get("support_geometry") if isinstance(body.get("support_geometry"), dict) else {}
    if not raw.get("available") or not raw.get("composer_eligible"):
        return None
    if raw.get("overall_shape") != "mostly_upright_over_support_leg":
        return None

    out: dict[str, Any] = {
        "overall_shape": "mostly_upright_over_support_leg",
    }
    for key in ("support_side", "elevated_side", "elevated_leg_height"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    angle = raw.get("support_axis_angle_from_vertical_deg")
    if isinstance(angle, (int, float)):
        out["support_axis_angle_from_vertical_deg"] = round(float(angle), 1)
    return out


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # v04 ultimately delegates to the v01 projection engine.  Its legacy
    # broad-pose helper read raw `text` and rejected `accepted_*` specialist
    # statuses.  Patch only that helper for this composer generation so the
    # canonical Phase-4B.9 pose actually reaches the text-only model.
    phase53.engine._broad_pose = _canonical_broad_pose

    projection, audit = _BASE_PROJECTION(
        sheet,
        trigger_token=trigger_token,
        subject_class=subject_class,
    )

    facts = sheet.get("facts") if isinstance(sheet.get("facts"), dict) else {}
    source_body = facts.get("body") if isinstance(facts.get("body"), dict) else {}
    support = _global_support_shape(source_body)
    if support:
        authoritative = projection.get("authoritative_facts") if isinstance(projection.get("authoritative_facts"), dict) else {}
        body = authoritative.get("body") if isinstance(authoritative.get("body"), dict) else {}
        body["global_support_shape"] = support
        authoritative["body"] = body
        projection["authoritative_facts"] = authoritative

    projected_body = (
        projection.get("authoritative_facts", {}).get("body", {})
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    audit["global_support_shape_projected"] = bool(support)
    audit["canonical_broad_pose_projected"] = _clean(projected_body.get("broad_pose")) if isinstance(projected_body, dict) else None
    return projection, audit


def main() -> int:
    # Keep every validated Phase-5.3 behavior (trigger/pronouns, gaze semantics,
    # signed torso direction, specialist laterality) and add one compact global
    # support-shape fact plus the canonical Phase-4B.9 broad pose for the
    # text-only composer.
    phase53.engine._broad_pose = _canonical_broad_pose
    phase53._projection = _projection
    phase53.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase53.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase53.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase53.SCHEMA_VERSION = SCHEMA_VERSION
    phase53.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase53.main()


if __name__ == "__main__":
    raise SystemExit(main())
