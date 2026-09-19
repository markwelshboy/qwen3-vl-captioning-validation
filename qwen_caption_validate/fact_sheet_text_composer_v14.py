from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from . import fact_sheet_text_composer_v13 as phase512

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_SUBDIR = Path("semantic-v3") / "caption-fact-sheet-v0.3.2"
DEFAULT_PROMPT = PACKAGE_ROOT / "prompts" / "fact_sheet_text_composer_v10.txt"
DEFAULT_OUTPUT_SUBDIR = Path("semantic-v3") / "text-composer-v0.14"
SCHEMA_VERSION = "fact-sheet-text-composer-0.14"
EXPECTED_FACT_SHEET_SCHEMA = "caption-fact-sheet-0.3.2"

# Freeze the validated Phase-5.12 stack before main() patches module globals.
_BASE_PROJECTION = phase512._projection
_BASE_CAPTION_AUDIT = phase512._caption_audit
_BASE_RETRY_PROMPT = phase512._retry_prompt

_CAMERA_HOLD_SUFFIX_RE = re.compile(
    r"(?:\s*,\s*|\s+)(?:while\s+)?"
    r"(?:holding|gripping|carrying)\s+(?:a\s+|the\s+)?camera\b.*$",
    re.I,
)
_CAMERA_CAPTURE_MECHANISM_RE = re.compile(
    r"\b(?:hold(?:s|ing)?|grip(?:s|ping)?|carry(?:ing|ies)?)\s+(?:a\s+|the\s+)?camera\b"
    r"|\b(?:tak(?:e|es|ing)|captur(?:e|es|ing))\s+(?:a\s+|the\s+)?"
    r"(?:photo|photograph|picture|selfie)\b"
    r"|\bphotograph(?:s|ing)?\s+(?:herself|himself|themself|themselves)\b",
    re.I,
)
_NEGATIVE_BODY_GEOMETRY_RE = re.compile(
    r"\bno\s+(?:visible\s+)?(?:body\s+|torso\s+)?(?:turn|tilt|lean|bend)\b"
    r"|\bno\s+(?:visible\s+)?(?:turn|tilt)(?:\s+or\s+(?:turn|tilt))?\b"
    r"|\bwithout\s+(?:any\s+)?(?:visible\s+)?(?:body\s+|torso\s+)?(?:turn|tilt|lean|bend)\b",
    re.I,
)
_NEGATIVE_VISIBILITY_CONTACT_RE = re.compile(
    r"\b(?:the\s+|her\s+|his\s+|their\s+)?(?:left\s+|right\s+)?"
    r"(?:hand|fist|palm|fingers?|arm|forearm|wrist|elbow|leg|knee|foot|feet)\s+"
    r"(?:is\s+|are\s+)?not\s+visible"
    r"(?:\s+in\s+(?:the\s+)?(?:crop|frame|image))?\b"
    r"|\b(?:there\s+is\s+)?no\s+(?:visible\s+)?contact\s+(?:with|between)\s+[^.!?;,]{1,90}"
    r"|\bno\s+(?:left\s+|right\s+)?"
    r"(?:hand|fist|palm|fingers?|arm|forearm)"
    r"(?:\s+(?:or|nor)\s+(?:left\s+|right\s+)?(?:hand|fist|palm|fingers?|arm|forearm))*\s+"
    r"(?:makes?|has|maintains?)\s+(?:any\s+)?contact\s+with\s+[^.!?;,]{1,90}"
    r"|\bno\s+(?:left\s+|right\s+)?"
    r"(?:hand|fist|palm|fingers?|arm|forearm)"
    r"(?:\s+(?:or|nor)\s+(?:left\s+|right\s+)?(?:hand|fist|palm|fingers?|arm|forearm))*\s+"
    r"(?:touches?|supports?)\b[^.!?;,]{0,90}"
    r"|\b(?:hand|fist|palm|fingers?|arm|forearm)\s+(?:does|do)\s+not\s+"
    r"(?:touch|contact|support)\b"
    r"|\b(?:hand|fist|palm|fingers?|arm|forearm)\s+(?:is|are)\s+not\s+"
    r"(?:touching|supporting)\b"
    r"|\bwithout\s+(?:any\s+)?contact\s+with\s+[^.!?;,]{1,90}",
    re.I,
)
_NEGATIVE_VISIBILITY_CONTACT_SUFFIX_RE = re.compile(
    r"(?:,\s*|\s+(?:and|but|though|while)\s+)"
    r"(?:"
    r"(?:the\s+|her\s+|his\s+|their\s+)?(?:left\s+|right\s+)?"
    r"(?:hand|fist|palm|fingers?|arm|forearm|wrist|elbow|leg|knee|foot|feet)\s+"
    r"(?:is\s+|are\s+)?not\s+visible"
    r"(?:\s+in\s+(?:the\s+)?(?:crop|frame|image))?"
    r"|(?:there\s+is\s+)?no\s+(?:visible\s+)?contact\s+(?:with|between)\s+[^.!?;,]{1,90}"
    r"|no\s+(?:left\s+|right\s+)?"
    r"(?:hand|fist|palm|fingers?|arm|forearm)"
    r"(?:\s+(?:or|nor)\s+(?:left\s+|right\s+)?(?:hand|fist|palm|fingers?|arm|forearm))*\s+"
    r"(?:makes?|has|maintains?)\s+(?:any\s+)?contact\s+with\s+[^.!?;,]{1,90}"
    r"|no\s+(?:left\s+|right\s+)?"
    r"(?:hand|fist|palm|fingers?|arm|forearm)"
    r"(?:\s+(?:or|nor)\s+(?:left\s+|right\s+)?(?:hand|fist|palm|fingers?|arm|forearm))*\s+"
    r"(?:touches?|supports?)\b[^.!?;,]*"
    r"|(?:hand|fist|palm|fingers?|arm|forearm)\s+(?:does|do)\s+not\s+"
    r"(?:touch|contact|support)\b[^.!?;,]*"
    r"|(?:hand|fist|palm|fingers?|arm|forearm)\s+(?:is|are)\s+not\s+"
    r"(?:touching|supporting)\b[^.!?;,]*"
    r")"
    r".*$",
    re.I,
)
_MIRROR_TAUTOLOGY_RE = re.compile(
    r"\b(?:the\s+)?mirror(?:\s+surface)?\s+reflects?\s+(?:the\s+)?scene\b",
    re.I,
)
_META_COMPOSITION_RE = re.compile(
    r"\b(?:the\s+)?(?:mirror\s+selfie\s+)?composition\s+"
    r"(?:frames?|focuses?|emphasiz(?:es|ing)|highlights?|captures?)\b"
    r"|\b(?:the\s+)?framing\s+(?:focuses?|emphasiz(?:es|ing)|highlights?|captures?)\b",
    re.I,
)
_INTERPRETIVE_CAUSAL_RE = re.compile(
    r"\b(?:contribut(?:e|es|ed|ing)|add(?:s|ed|ing)?|giv(?:e|es|ing)|"
    r"creat(?:e|es|ed|ing)|evok(?:e|es|ed|ing)|suggest(?:s|ed|ing)?)\b"
    r"[^.!?]{0,120}\b(?:mood|expression|feeling|feel|atmosphere|vibe|tone)\b",
    re.I,
)
_SELF_CONTAINED_MOMENT_RE = re.compile(r"\bself[- ]contained\s+moment\b", re.I)

_PRIMARY_POSE_PATTERNS: dict[str, re.Pattern[str]] = {
    "standing": re.compile(r"\b(?:stand|stands|standing)\b", re.I),
    "seated": re.compile(r"\b(?:seated|sit|sits|sitting)\b", re.I),
    "lying": re.compile(r"\b(?:lie|lies|lying|reclined|reclining)\b", re.I),
    "crouching": re.compile(r"\b(?:crouch|crouches|crouched|crouching)\b", re.I),
    "kneeling": re.compile(r"\b(?:kneel|kneels|kneeled|kneeling)\b", re.I),
    "squatting": re.compile(r"\b(?:squat|squats|squatted|squatting)\b", re.I),
}
_SECONDARY_HUMAN_NOUN_RE = re.compile(
    r"\b(?:person|people|child|children|man|men|woman|women|figure|figures|boy|girl)\b",
    re.I,
)
_INTERVENING_NONPRIMARY_CLAUSE_SUBJECT_RE = re.compile(
    r"(?:[,;]\s*|\b)(?:and|but|while|whereas)\s+"
    r"(?:a|an|the|this|that|these|those|another|one)\s+"
    r"(?:[A-Za-z][A-Za-z'’-]*\s+){0,5}[A-Za-z][A-Za-z'’-]*\s*$",
    re.I,
)

_MIRROR_PHONE_HARDWARE_RE = re.compile(
    r"\b(?:triple|multiple|three)\s+(?:rear[- ]?)?camera"
    r"(?:s|\s+module|\s+lenses?)?\b"
    r"|\brear[- ]facing\s+camera(?:s|\s+module)?\b"
    r"|\brear\s+camera(?:s|\s+module)?\b"
    r"|\bcamera\s+array\b",
    re.I,
)
_MIRROR_PHONE_HARDWARE_CLAUSE_RE = re.compile(
    r"\s+(?:with|featuring|features|showing|has|having)\s+(?:a\s+)?"
    r"(?:"
    r"(?:triple|multiple|three)\s+(?:rear[- ]?)?camera(?:s|\s+module|\s+lenses?)?"
    r"|rear[- ]facing\s+camera(?:s|\s+module)?"
    r"|rear\s+camera(?:s|\s+module)?"
    r"|camera\s+array"
    r")\b",
    re.I,
)


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    return text or None


def _capture_authorized(projection_or_authoritative: dict[str, Any]) -> bool:
    if not isinstance(projection_or_authoritative, dict):
        return False

    authoritative = projection_or_authoritative
    if isinstance(projection_or_authoritative.get("authoritative_facts"), dict):
        authoritative = projection_or_authoritative["authoritative_facts"]

    capture = (
        authoritative.get("capture")
        if isinstance(authoritative.get("capture"), dict)
        else {}
    )
    return capture.get("subtype") in {"direct_selfie", "mirror_selfie"}


def _strip_unauthorized_capture_mechanism(text: str) -> str | None:
    """Remove capture-mechanism semantics while preserving local body geometry."""
    value = " ".join(str(text or "").split())
    if not value:
        return None

    # The common useful form is "arm extended forward, holding the camera".
    # Keep the local arm geometry and remove only the capture claim.
    value = _CAMERA_HOLD_SUFFIX_RE.sub("", value).strip(" ,.;")

    # If the remaining phrase still states a capture mechanism, the relation is
    # capture-owned and must be withheld entirely.
    if _CAMERA_CAPTURE_MECHANISM_RE.search(value):
        return None

    return value or None



def _strip_negative_visibility_contact(text: str) -> str | None:
    """Keep positive visible geometry; drop negative visibility/contact bookkeeping."""
    value = " ".join(str(text or "").split())
    if not value:
        return None

    value = _NEGATIVE_VISIBILITY_CONTACT_SUFFIX_RE.sub("", value).strip(" ,.;")
    if not value:
        return None

    if _NEGATIVE_VISIBILITY_CONTACT_RE.search(value):
        return None

    return value or None


def _sanitize_negative_visibility_contact_configuration(
    values: Any,
) -> tuple[list[Any], list[str]]:
    if not isinstance(values, list):
        return [], []

    out: list[Any] = []
    sanitized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            out.append(copy.deepcopy(item))
            continue

        normalized = " ".join(item.split())
        cleaned = _strip_negative_visibility_contact(item)
        if cleaned != normalized:
            sanitized.append(item)
        if cleaned:
            out.append(cleaned)

    return out, sanitized


def _capture_safe_configuration(
    values: Any,
    *,
    capture_authorized: bool,
) -> tuple[list[Any], list[str]]:
    if not isinstance(values, list):
        return [], []

    if capture_authorized:
        return copy.deepcopy(values), []

    out: list[Any] = []
    withheld: list[str] = []
    for item in values:
        if not isinstance(item, str):
            out.append(copy.deepcopy(item))
            continue

        cleaned = _strip_unauthorized_capture_mechanism(item)
        if cleaned != " ".join(item.split()):
            withheld.append(item)
        if cleaned:
            out.append(cleaned)

    return out, withheld



def _strip_mirror_phone_hardware_anywhere(text: str) -> str | None:
    value = " ".join(str(text or "").split())
    if not value or not _MIRROR_PHONE_HARDWARE_RE.search(value):
        return value or None

    value = _MIRROR_PHONE_HARDWARE_CLAUSE_RE.sub("", value)
    value = _MIRROR_PHONE_HARDWARE_RE.sub("", value)
    value = re.sub(r"\s{2,}", " ", value)
    value = re.sub(r"\s+([,.;])", r"\1", value)
    value = re.sub(r"\b(?:with|featuring|features|showing|has|having)\s+(?:a\s+)?(?=[,.;]|$)", "", value, flags=re.I)
    value = value.strip(" ,.;")
    return value or None


def _mirror_phone_hardware_surface(
    value: Any,
    removed: list[str],
) -> Any:
    if isinstance(value, str):
        cleaned = _strip_mirror_phone_hardware_anywhere(value)
        if cleaned != " ".join(value.split()):
            removed.append(value)
        return cleaned
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            cleaned = _mirror_phone_hardware_surface(item, removed)
            if cleaned not in (None, "", [], {}):
                out.append(cleaned)
        return out
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _mirror_phone_hardware_surface(item, removed)
            if cleaned not in (None, "", [], {}):
                out[key] = cleaned
        return out
    return copy.deepcopy(value)


def _allowed_primary_pose_groups(projection: dict[str, Any]) -> set[str]:
    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    body = (
        authoritative.get("body")
        if isinstance(authoritative.get("body"), dict)
        else {}
    )
    texts = [
        _clean(body.get("broad_pose")),
        _clean(body.get("canonical_broad_pose")),
    ]
    allowed: set[str] = set()
    for text in texts:
        if not text:
            continue
        for name, pattern in _PRIMARY_POSE_PATTERNS.items():
            if pattern.search(text):
                allowed.add(name)
    return allowed


def _primary_subject_pose_groups(
    caption: str,
    projection: dict[str, Any],
) -> set[str]:
    subject = (
        projection.get("subject")
        if isinstance(projection.get("subject"), dict)
        else {}
    )
    trigger = _clean(subject.get("trigger_token"))
    pronoun = _clean(subject.get("subject_pronoun"))
    possessive = _clean(subject.get("possessive_pronoun"))

    ref_patterns: list[re.Pattern[str]] = []
    if trigger:
        ref_patterns.append(re.compile(rf"(?<![A-Za-z0-9_]){re.escape(trigger)}(?![A-Za-z0-9_])"))
    if pronoun:
        ref_patterns.append(re.compile(rf"\b{re.escape(pronoun)}\b", re.I))

    possessive_body_re = (
        re.compile(
            rf"\b{re.escape(possessive)}\s+(?:body|posture|stance)\b",
            re.I,
        )
        if possessive
        else None
    )

    used: set[str] = set()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption) if s.strip()]
    for sentence in sentences:
        refs: list[re.Match[str]] = []
        for pattern in ref_patterns:
            refs.extend(pattern.finditer(sentence))
        refs.sort(key=lambda m: m.start())

        for name, pattern in _PRIMARY_POSE_PATTERNS.items():
            for pose_match in pattern.finditer(sentence):
                after = sentence[pose_match.end():pose_match.end() + 28]
                # "a seated person" / "standing man" belongs to the secondary
                # noun that immediately follows the posture adjective.
                if _SECONDARY_HUMAN_NOUN_RE.search(after):
                    continue

                primary_bound = False
                for ref in refs:
                    if ref.end() <= pose_match.start():
                        between = sentence[ref.end():pose_match.start()]
                        if (
                            not _SECONDARY_HUMAN_NOUN_RE.search(between)
                            and not _INTERVENING_NONPRIMARY_CLAUSE_SUBJECT_RE.search(between)
                        ):
                            primary_bound = True
                    elif pose_match.end() <= ref.start():
                        # Fronted participle: "Seated near the window, sH1VX ..."
                        between = sentence[pose_match.end():ref.start()]
                        if len(between) <= 90 and not _SECONDARY_HUMAN_NOUN_RE.search(between):
                            primary_bound = True
                    if primary_bound:
                        break

                if not primary_bound and possessive_body_re:
                    prefix = sentence[:pose_match.start()]
                    poss = list(possessive_body_re.finditer(prefix))
                    if poss:
                        between = sentence[poss[-1].end():pose_match.start()]
                        primary_bound = not _SECONDARY_HUMAN_NOUN_RE.search(between)

                if primary_bound:
                    used.add(name)
    return used


def _suppress_mirror_neutral_torso(body: dict[str, Any]) -> bool:
    torso = (
        body.get("torso_orientation")
        if isinstance(body.get("torso_orientation"), dict)
        else {}
    )
    if torso.get("camera_orientation") != "frontal":
        return False

    body.pop("torso_orientation", None)
    return True


def _projection(
    sheet: dict[str, Any],
    *,
    trigger_token: str | None = None,
    subject_class: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    projection, audit = _BASE_PROJECTION(
        sheet,
        trigger_token=trigger_token,
        subject_class=subject_class,
    )

    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    authoritative = copy.deepcopy(authoritative)

    capture = (
        authoritative.get("capture")
        if isinstance(authoritative.get("capture"), dict)
        else {}
    )
    subtype = _clean(capture.get("subtype"))
    capture_ok = subtype in {"direct_selfie", "mirror_selfie"}
    mirror_mode = subtype == "mirror_selfie"

    body = (
        authoritative.get("body")
        if isinstance(authoritative.get("body"), dict)
        else {}
    )
    body = copy.deepcopy(body)

    original_configuration = (
        body.get("configuration")
        if isinstance(body.get("configuration"), list)
        else []
    )
    configuration, withheld_capture_relations = _capture_safe_configuration(
        original_configuration,
        capture_authorized=capture_ok,
    )
    configuration, sanitized_negative_relations = (
        _sanitize_negative_visibility_contact_configuration(configuration)
    )
    if original_configuration:
        if configuration:
            body["configuration"] = configuration
        else:
            body.pop("configuration", None)

    mirror_frontal_torso_suppressed = False
    mirror_phone_hardware_sanitized: list[str] = []
    if mirror_mode:
        mirror_frontal_torso_suppressed = _suppress_mirror_neutral_torso(body)
        authoritative = _mirror_phone_hardware_surface(
            authoritative,
            mirror_phone_hardware_sanitized,
        )
        body = (
            authoritative.get("body")
            if isinstance(authoritative.get("body"), dict)
            else {}
        )

    if body:
        authoritative["body"] = body
    else:
        authoritative.pop("body", None)

    holistic = _clean(projection.get("holistic_context_non_authoritative"))
    holistic_capture_withheld = bool(
        not capture_ok
        and holistic
        and _CAMERA_CAPTURE_MECHANISM_RE.search(holistic)
    )
    if holistic_capture_withheld:
        projection.pop("holistic_context_non_authoritative", None)

    projection["authoritative_facts"] = authoritative
    audit["capture_mechanism_domain_firewall_enforced"] = True
    audit["capture_mechanism_relations_sanitized"] = withheld_capture_relations
    audit["negative_visibility_contact_relations_sanitized"] = sanitized_negative_relations
    audit["holistic_capture_mechanism_withheld"] = holistic_capture_withheld
    audit["mirror_neutral_frontal_torso_suppressed"] = (
        mirror_frontal_torso_suppressed
    )
    audit["mirror_phone_hardware_sanitized"] = mirror_phone_hardware_sanitized
    return projection, audit


def _caption_audit(caption: str, projection: dict[str, Any]) -> dict[str, Any]:
    audit = _BASE_CAPTION_AUDIT(caption, projection)
    violations = [
        violation
        for violation in list(audit.get("violations") or [])
        if not str(violation).startswith("unauthorized_broad_pose:")
    ]

    allowed_primary_pose = _allowed_primary_pose_groups(projection)
    used_primary_pose = _primary_subject_pose_groups(caption, projection)
    unauthorized_primary_pose = sorted(used_primary_pose - allowed_primary_pose)
    if unauthorized_primary_pose:
        violations.append(
            "unauthorized_broad_pose:" + ",".join(unauthorized_primary_pose)
        )

    capture_ok = _capture_authorized(projection)
    capture_mechanism_used = bool(_CAMERA_CAPTURE_MECHANISM_RE.search(caption))
    if capture_mechanism_used and not capture_ok:
        violations.append("capture_mechanism_language_without_capture_authority")

    negative_geometry = [
        " ".join(m.group(0).split())
        for m in _NEGATIVE_BODY_GEOMETRY_RE.finditer(caption)
    ]
    if negative_geometry:
        violations.append("unsupported_negative_body_geometry_language")

    negative_visibility_contact = [
        " ".join(m.group(0).split())
        for m in _NEGATIVE_VISIBILITY_CONTACT_RE.finditer(caption)
    ]
    if negative_visibility_contact:
        violations.append("negative_visibility_or_contact_language")

    mirror_tautology = [
        " ".join(m.group(0).split())
        for m in _MIRROR_TAUTOLOGY_RE.finditer(caption)
    ]
    if mirror_tautology:
        violations.append("mirror_reflection_tautology")

    meta_composition = [
        " ".join(m.group(0).split())
        for m in _META_COMPOSITION_RE.finditer(caption)
    ]
    if meta_composition:
        violations.append("meta_composition_narration")

    interpretive_causal = [
        " ".join(m.group(0).split())
        for m in _INTERPRETIVE_CAUSAL_RE.finditer(caption)
    ]
    if interpretive_causal:
        violations.append("unsupported_setting_to_mood_causality")

    self_contained = [
        " ".join(m.group(0).split())
        for m in _SELF_CONTAINED_MOMENT_RE.finditer(caption)
    ]
    if self_contained:
        violations.append("interpretive_moment_narration")

    authoritative = (
        projection.get("authoritative_facts")
        if isinstance(projection.get("authoritative_facts"), dict)
        else {}
    )
    capture = (
        authoritative.get("capture")
        if isinstance(authoritative.get("capture"), dict)
        else {}
    )
    mirror_phone_hardware = [
        " ".join(m.group(0).split())
        for m in _MIRROR_PHONE_HARDWARE_RE.finditer(caption)
    ]
    if capture.get("subtype") == "mirror_selfie" and mirror_phone_hardware:
        violations.append("mirror_phone_hardware_leak")

    audit["violations"] = sorted(set(violations))
    audit["capture_mechanism_language_used"] = capture_mechanism_used
    audit["capture_mechanism_authorized"] = capture_ok
    audit["negative_body_geometry_phrases"] = negative_geometry
    audit["negative_visibility_contact_phrases"] = negative_visibility_contact
    audit["mirror_reflection_tautology_phrases"] = mirror_tautology
    audit["meta_composition_phrases"] = meta_composition
    audit["setting_to_mood_causality_phrases"] = interpretive_causal
    audit["interpretive_moment_phrases"] = self_contained
    audit["primary_subject_allowed_pose_groups"] = sorted(allowed_primary_pose)
    audit["primary_subject_used_pose_groups"] = sorted(used_primary_pose)
    audit["primary_subject_pose_scope_enforced"] = True
    audit["mirror_phone_hardware_phrases"] = mirror_phone_hardware
    audit["caption_surface_narration_guard_enforced"] = True
    return audit


def _retry_prompt(
    original_prompt: str,
    caption: str,
    violations: list[str],
) -> str:
    prompt = _BASE_RETRY_PROMPT(original_prompt, caption, violations)
    extra: list[str] = []

    unauthorized_pose = [
        violation.split(":", 1)[1]
        for violation in violations
        if violation.startswith("unauthorized_broad_pose:") and ":" in violation
    ]
    if unauthorized_pose:
        extra.append(
            "Remove the unsupported broad-pose/posture claim(s) "
            + ", ".join(unauthorized_pose)
            + " entirely. Do not replace them with another posture or with a negative/neutral posture statement. If a scene/object phrase depends grammatically on the unsupported posture (for example 'lying on a gray fabric surface'), remove that dependent relation too rather than preserving a dangling preposition."
        )

    if "capture_mechanism_language_without_capture_authority" in violations:
        extra.append(
            "Remove all claims that the subject is holding a camera, taking a photo, or otherwise operating the capture device. Preserve only local body geometry that is independently supplied, such as an arm extending forward."
        )
    if "unsupported_negative_body_geometry_language" in violations:
        extra.append(
            "Remove negative/neutral completion such as 'no visible turn or tilt'. Absence of a turn, tilt, lean, or bend is not a caption fact."
        )
    if "negative_visibility_or_contact_language" in violations:
        extra.append(
            "Remove negative visibility/contact bookkeeping such as 'the hand is not visible in the crop', 'there is no contact with her face or chin', 'the hand does not touch/support the face', or similar veto language. Preserve only independently supplied positive visible geometry, such as 'one arm extends outward with the elbow bent'."
        )
    if "mirror_reflection_tautology" in violations:
        extra.append(
            "Remove tautological mirror narration such as 'the mirror reflects the scene'. The authoritative 'mirror selfie' phrase already carries that capture information."
        )
    if "meta_composition_narration" in violations:
        extra.append(
            "Remove meta-caption narration about what the composition or framing focuses on, emphasizes, highlights, captures, or frames. State concrete visible facts directly."
        )
    if "unsupported_setting_to_mood_causality" in violations:
        extra.append(
            "Remove any causal claim that the setting, lighting, or background creates or contributes to the subject's mood, expression, feeling, feel, atmosphere, vibe, or tone. Keep the visible setting and visible expression as separate facts."
        )
    if "mirror_phone_hardware_leak" in violations:
        extra.append(
            "Remove rear/triple/multiple-camera hardware detail from the mirror-selfie caption. Keep the visible phone itself, but do not mention the camera array, rear camera module, triple camera, or number of camera lenses."
        )
    if "interpretive_moment_narration" in violations:
        extra.append(
            "Remove interpretive phrases such as 'self-contained moment'. End on concrete visible facts instead."
        )

    if not extra:
        return prompt
    return prompt + "\nCAPTION-SURFACE REVISION:\n- " + "\n- ".join(extra)


def main() -> int:
    # Preserve the full Phase-5.12 semantic-surface + Phase-5.11 selfie/mirror
    # stack. Extend only capture-domain ownership, mirror neutral geometry, and
    # anti-narration surface discipline.
    phase512._projection = _projection
    phase512._caption_audit = _caption_audit
    phase512._retry_prompt = _retry_prompt
    phase512.DEFAULT_INPUT_SUBDIR = DEFAULT_INPUT_SUBDIR
    phase512.DEFAULT_PROMPT = DEFAULT_PROMPT
    phase512.DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_SUBDIR
    phase512.SCHEMA_VERSION = SCHEMA_VERSION
    phase512.EXPECTED_FACT_SHEET_SCHEMA = EXPECTED_FACT_SHEET_SCHEMA
    return phase512.main()


if __name__ == "__main__":
    raise SystemExit(main())
