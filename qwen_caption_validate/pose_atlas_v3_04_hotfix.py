from __future__ import annotations

"""Compatibility hotfixes for Pose Atlas v0.4.

v0.4 calls ``v03._resolve_mesh`` but the helper actually lives in the v0.2
module, which v0.3 imports as ``v03.v02``. It also originally imported the
v0.2 relational profiler directly. Patch both aliases before delegating so
atlas review uses the current v0.3 profile semantics without rewriting the
large diagnostic renderer.
"""

from . import pose_atlas_v3_04 as v04
from . import sam3d_relational_pose_profile_03 as relprof03


v04.v03._resolve_mesh = v04.v03.v02._resolve_mesh
v04.relprof = relprof03

_original_make_card = v04._make_card
_original_html_index = v04._html_index


def _make_card_v03(*args, **kwargs):
    card, record = _original_make_card(*args, **kwargs)
    policy = record.get("interpretation_policy") or {}
    policy.pop("waving_candidate_requires_vlm_confirmation", None)
    policy["action_semantics_reserved_for_fusion_caption"] = True
    policy["open_closed_hand_and_wrist_height_are_geometry_primitives"] = True
    record["interpretation_policy"] = policy
    return card, record


def _html_index_v03(records):
    text = _original_html_index(records)
    return text.replace(
        "Raised-open-hand geometry is only a waving candidate and requires VLM confirmation.",
        "Open/closed hand shape and wrist height remain geometry primitives; action semantics are reserved for Fusion/Caption.",
    )


v04._make_card = _make_card_v03
v04._html_index = _html_index_v03


def main() -> int:
    return v04.main()


if __name__ == "__main__":
    raise SystemExit(main())
